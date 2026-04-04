#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Combined Portfolio Strategy - Policy Gap Retail Reversal + Institutional Flow Divergence

This example demonstrates how to combine two strategies using the PortfolioStrategy system
with order netting. When Strategy A wants to buy and Strategy B wants to sell the same
symbol, the orders are netted and only the difference is executed.

Strategies:
1. PolicyGapRetailReversal (Gap Fade): Contrarian - buys on overnight gaps down
2. InstitutionalFlowDivergence (Momentum): Trend-following - buys on breakouts

These strategies have different alpha sources and may generate conflicting signals,
making them ideal candidates for portfolio combination with order netting.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from lumibot.strategies import Strategy
from lumibot.entities import Asset, Data
from lumibot.backtesting import PandasDataBacktesting, QMTBridgeDataBacktesting
from lumibot.strategies.portfolio_strategy import (
    PortfolioStrategy,
    SubStrategy,
    TradingSignal,
)
from lumibot.credentials import IS_BACKTESTING

# Import strategy-specific parameters and constants
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ── Shared Constants ──────────────────────────────────────────────────────────

# Load QMT Bridge environment
QMT_BRIDGE_ENV_PATH = "/home/quant_volumn/docker/data/qmt-bridge/.env"
load_dotenv(QMT_BRIDGE_ENV_PATH)

# Sector mapping for Policy Gap strategy
SECTOR_MAP = {
    '300750.SZ': 'Renewable', '002129.SZ': 'Renewable', '601865.SH': 'Renewable',
    '002594.SZ': 'Renewable', '300274.SZ': 'Renewable', '600900.SH': 'Renewable',
    '601615.SH': 'Renewable', '002202.SZ': 'Renewable',
    '002475.SZ': 'Tech', '002415.SZ': 'Tech', '300059.SZ': 'Tech',
    '600588.SH': 'Tech', '002230.SZ': 'Tech', '002049.SZ': 'Tech',
    '600519.SH': 'Consumption', '000858.SZ': 'Consumption', '600887.SH': 'Consumption',
    '000333.SZ': 'Consumption', '000651.SZ': 'Consumption', '601888.SH': 'Consumption',
}

DEFAULT_SYMBOLS = [
    '300750.SZ', '002129.SZ', '601865.SH', '002594.SZ',
    '300274.SZ', '600900.SH', '601615.SH', '002202.SZ',
    '002475.SZ', '002415.SZ', '300059.SZ', '600588.SH',
    '002230.SZ', '002049.SZ',
    '600519.SH', '000858.SZ', '600887.SH', '000333.SZ',
    '000651.SZ', '601888.SH',
]


# ── Sub-Strategy 1: Policy Gap Retail Reversal ─────────────────────────────────

class PolicyGapRetailReversalSub(SubStrategy):
    """
    Gap Fade sub-strategy for portfolio.
    Generates signals based on overnight gap patterns.
    """

    def initialize(self,
                   gap_min=0.02,
                   gap_max=0.10,
                   volume_multiple=1.2,
                   position_size=100):
        self.gap_min = gap_min
        self.gap_max = gap_max
        self.volume_multiple = volume_multiple
        self.position_size = position_size
        self.symbols = self.portfolio.main_strategy.symbols

    def on_trading_iteration(self):
        """Generate signals based on gap fade logic."""
        for symbol in self.symbols:
            try:
                # Get price history from main strategy
                asset = Asset(symbol=symbol, asset_type=Asset.AssetType.STOCK)
                bars = self.portfolio.main_strategy.get_historical_prices(asset, 60, timestep="day")

                if bars is None or bars.df is None or len(bars.df) < 22:
                    continue

                df = bars.df
                close = df['close']
                open_prices = df.get('open', close)
                volume = df.get('volume', pd.Series([1] * len(df)))

                # Today's open (execution price)
                today_open = open_prices.iloc[-1]
                # Yesterday's close (last available data at 9:30 AM)
                yesterday_close = close.iloc[-2]

                # Calculate overnight gap
                gap = (today_open - yesterday_close) / yesterday_close

                # Check: Gap DOWN 2-10%
                if gap >= -self.gap_min or gap <= -self.gap_max:
                    continue

                # Check: Yesterday's volume above average
                yesterday_volume = volume.iloc[-2]
                avg_volume = volume.iloc[-22:-2].mean()

                if avg_volume <= 0:
                    continue
                vol_ratio = yesterday_volume / avg_volume

                if vol_ratio < self.volume_multiple:
                    continue

                # Check: Policy-supported sector
                sector = SECTOR_MAP.get(symbol, 'Other')
                if sector == 'Other':
                    continue

                # Generate BUY signal (gap fade - buy the dip)
                self.buy(symbol, self.position_size)

                print(f"[GapFade] {symbol}: BUY signal (gap={gap:.1%}, vol={vol_ratio:.1f}x)")

            except Exception as e:
                print(f"[GapFade] Error processing {symbol}: {e}")


# ── Sub-Strategy 2: Institutional Flow Divergence ──────────────────────────────

class InstitutionalFlowDivergenceSub(SubStrategy):
    """
    Momentum sub-strategy for portfolio.
    Generates signals based on breakout and volume patterns.
    """

    def initialize(self,
                   min_vol_ratio=1.3,
                   min_score_entry=55,
                   position_size=100):
        self.min_vol_ratio = min_vol_ratio
        self.min_score_entry = min_score_entry
        self.position_size = position_size
        self.symbols = self.portfolio.main_strategy.symbols

    def _compute_score(self, symbol, df):
        """Compute entry score using data available before market open."""
        try:
            close = df['close'].values
            volume = df['volume'].values
            high = df['high'].values
            low = df['low'].values
            n = len(close)
            if n < 30:
                return None

            curr = close[-1]

            # Component 1: Breakout (0-35)
            high_20d = np.max(high[-21:-1])
            low_20d = np.min(low[-21:-1])
            rng = high_20d - low_20d
            pos = (curr - low_20d) / rng if rng > 0 else 0.5

            if curr >= high_20d:
                breakout = 35
            elif pos > 0.95:
                breakout = 28
            elif pos > 0.90:
                breakout = 22
            else:
                breakout = max(0, (pos - 0.7) * 50)

            # Component 2: Volume surge (0-30, mandatory)
            vol_20d = volume[-21:-1]
            vol_avg = np.mean(vol_20d)
            vol_ratio = volume[-1] / vol_avg if vol_avg > 0 else 1

            if vol_ratio > 2.5:
                vol_score = 30
            elif vol_ratio > 2.0:
                vol_score = 25
            elif vol_ratio > 1.5:
                vol_score = 20
            elif vol_ratio > self.min_vol_ratio:
                vol_score = 12
            else:
                vol_score = 0
            if vol_score == 0:
                return 0

            # Component 3: Trend alignment (0-25)
            ma5 = np.mean(close[-6:-1])
            ma10 = np.mean(close[-11:-1])
            ma20 = np.mean(close[-21:-1])

            if curr > ma5 > ma10 > ma20:
                trend = 25
            elif curr > ma5 > ma10:
                trend = 18
            elif curr > ma5:
                trend = 10
            elif curr > ma10:
                trend = 4
            else:
                trend = 0

            # Component 4: Momentum (0-10)
            ret_3d = (close[-1] - close[-4]) / close[-4] if close[-4] > 0 else 0
            if ret_3d > 0.05:
                mom = 10
            elif ret_3d > 0.02:
                mom = 6
            elif ret_3d > 0:
                mom = 3
            else:
                mom = 0

            return breakout + vol_score + trend + mom

        except Exception:
            return None

    def on_trading_iteration(self):
        """Generate signals based on momentum logic."""
        for symbol in self.symbols:
            try:
                # Get price history
                asset = Asset(symbol=symbol, asset_type=Asset.AssetType.STOCK)
                bars = self.portfolio.main_strategy.get_historical_prices(asset, 60, timestep="day")

                if bars is None or bars.df is None or len(bars.df) < 30:
                    continue

                df = bars.df
                score = self._compute_score(symbol, df)

                if score is not None and score >= self.min_score_entry:
                    # Generate BUY signal (momentum breakout)
                    self.buy(symbol, self.position_size)
                    print(f"[Momentum] {symbol}: BUY signal (score={score:.0f})")

            except Exception as e:
                print(f"[Momentum] Error processing {symbol}: {e}")


# ── Main Portfolio Strategy ────────────────────────────────────────────────────

class CombinedPortfolioStrategy(Strategy):
    """
    Combined portfolio strategy that runs multiple sub-strategies with order netting.

    Order Netting Example:
    - GapFade wants to buy 000001.SZ: +100
    - Momentum wants to sell 000001.SZ: -100
    - Net result: 0 (no trade executed, saves fees)

    Partial Netting Example:
    - GapFade wants to buy 000001.SZ: +150
    - Momentum wants to sell 000001.SZ: -100
    - Net result: +50 (only 50 shares bought)
    """

    def initialize(self,
                   symbols=None,
                   lot_size=100,
                   max_positions=10,
                   # Sub-strategy parameters
                   gap_min=0.02,
                   gap_max=0.10,
                   volume_multiple=1.2,
                   min_vol_ratio=1.3,
                   min_score_entry=55,
                   # Exit parameters
                   profit_target=0.10,
                   stop_loss=0.04,
                   max_hold_days=10):
        """
        Initialize the combined portfolio strategy.
        """
        self.sleeptime = "1D"
        self.symbols = symbols or DEFAULT_SYMBOLS
        self.lot_size = lot_size
        self.max_positions = max_positions
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        self.max_hold_days = max_hold_days

        # Position tracking
        self.positions_dict = {}
        self.price_history = {}

        # Create portfolio manager
        self.portfolio = PortfolioStrategy(self, min_net_quantity=lot_size)

        # Add sub-strategies
        self.portfolio.add_sub_strategy(
            PolicyGapRetailReversalSub,
            "gap_fade",
            gap_min=gap_min,
            gap_max=gap_max,
            volume_multiple=volume_multiple,
            position_size=100
        )

        self.portfolio.add_sub_strategy(
            InstitutionalFlowDivergenceSub,
            "momentum",
            min_vol_ratio=min_vol_ratio,
            min_score_entry=min_score_entry,
            position_size=100
        )

        print(f"\n{'='*60}")
        print("Combined Portfolio Strategy Initialized")
        print(f"{'='*60}")
        print(f"Symbols: {len(self.symbols)}")
        print(f"Sub-strategies: gap_fade, momentum")
        print(f"Order netting enabled (min_qty={lot_size})")
        print(f"Exit: {profit_target:.0%} profit / {stop_loss:.0%} stop / {max_hold_days}d max")
        print(f"{'='*60}\n")

    def on_trading_iteration(self):
        """Main trading iteration."""
        current_date = self.get_datetime()
        print(f"\n{'='*60}")
        print(f"Trading Iteration: {current_date.strftime('%Y-%m-%d')}")
        print(f"{'='*60}")

        # 1. Check exit conditions for existing positions
        self._check_exit_conditions()

        # 2. Update price history
        self._update_price_history()

        # 3. Run all sub-strategies (collects signals)
        signals = self.portfolio.run_iteration()

        # 4. Get netted orders
        netted_orders = self.portfolio.get_netted_orders()

        # 5. Print netting summary
        summary = self.portfolio.get_netting_summary()
        print(f"\nNetting Summary:")
        for symbol, details in summary["netting_details"].items():
            net_qty = details["net_quantity"]
            strategies = details["strategies"]
            print(f"  {symbol}: net={net_qty:+.0f} (from {strategies})")

        # 6. Execute netted orders
        if netted_orders:
            print(f"\nExecuting {len(netted_orders)} netted orders:")
            self._execute_netted_orders(netted_orders)
        else:
            print(f"\nNo orders to execute (all netted to zero)")

        # 7. Print current positions
        self._print_positions()

        self.await_market_to_close()

    def _update_price_history(self):
        """Update price history for all symbols."""
        lookback = 60
        for symbol in self.symbols:
            try:
                asset = Asset(symbol=symbol, asset_type=Asset.AssetType.STOCK)
                bars = self.get_historical_prices(asset, lookback, timestep="day")
                if bars is not None and bars.df is not None and len(bars.df) >= 20:
                    self.price_history[symbol] = bars.df
            except Exception:
                pass

    def _execute_netted_orders(self, netted_orders):
        """Execute netted orders."""
        for netted in netted_orders:
            if not netted.should_execute:
                continue

            symbol = netted.symbol
            quantity = int(netted.abs_quantity)

            # Round to lot size
            quantity = (quantity // self.lot_size) * self.lot_size
            if quantity <= 0:
                continue

            try:
                asset = Asset(symbol=symbol, asset_type=Asset.AssetType.STOCK)
                last_price = self.get_last_price(asset)

                if last_price is None or last_price <= 0:
                    continue

                # Check position limit
                current_positions = len(self.positions_dict)
                if netted.is_buy and current_positions >= self.max_positions:
                    print(f"  SKIP {symbol}: max positions reached ({self.max_positions})")
                    continue

                # Execute order
                side = "buy" if netted.is_buy else "sell"
                order = self.create_order(asset, quantity, side)
                self.submit_order(order)

                # Update tracking
                if netted.is_buy:
                    self.positions_dict[symbol] = {
                        'quantity': quantity,
                        'entry_price': last_price,
                        'entry_date': self.get_datetime(),
                        'highest_price': last_price
                    }
                    print(f"  BUY {symbol} x {quantity} @ {last_price:.2f}")
                else:
                    if symbol in self.positions_dict:
                        del self.positions_dict[symbol]
                    print(f"  SELL {symbol} x {quantity} @ {last_price:.2f}")

                # Update portfolio position manager
                self.portfolio.update_positions_after_fill(
                    order, last_price, quantity if netted.is_buy else -quantity
                )

            except Exception as e:
                print(f"  ERROR executing {symbol}: {e}")

    def _check_exit_conditions(self):
        """Check exit conditions for existing positions."""
        for symbol in list(self.positions_dict.keys()):
            pos = self.positions_dict[symbol]

            try:
                asset = Asset(symbol=symbol, asset_type=Asset.AssetType.STOCK)
                current_price = self.get_last_price(asset)

                if current_price is None or current_price <= 0:
                    continue

                entry_price = pos['entry_price']
                entry_date = pos.get('entry_date', self.get_datetime())

                pnl_pct = (current_price - entry_price) / entry_price
                days_held = (self.get_datetime() - entry_date).days

                # Update highest price
                if current_price > pos['highest_price']:
                    pos['highest_price'] = current_price

                # Exit 1: Profit target
                if pnl_pct >= self.profit_target:
                    self._exit_position(symbol, f"Profit target ({pnl_pct:.1%})")
                    continue

                # Exit 2: Stop loss
                if pnl_pct <= -self.stop_loss:
                    self._exit_position(symbol, f"Stop loss ({pnl_pct:.1%})")
                    continue

                # Exit 3: Max hold days
                if days_held >= self.max_hold_days:
                    self._exit_position(symbol, f"Max hold ({days_held}d)")
                    continue

            except Exception:
                pass

    def _exit_position(self, symbol, reason):
        """Exit a position."""
        if symbol not in self.positions_dict:
            return

        pos = self.positions_dict[symbol]
        quantity = pos['quantity']

        try:
            asset = Asset(symbol=symbol, asset_type=Asset.AssetType.STOCK)
            order = self.create_order(asset, quantity, "sell")
            self.submit_order(order)

            current_price = self.get_last_price(asset)
            print(f"  SELL {symbol} x {quantity} @ {current_price:.2f} ({reason})")

            del self.positions_dict[symbol]

        except Exception as e:
            print(f"  ERROR selling {symbol}: {e}")

    def _print_positions(self):
        """Print current positions."""
        if not self.positions_dict:
            print(f"\nCurrent Positions: None")
            return

        print(f"\nCurrent Positions ({len(self.positions_dict)}):")
        total_value = 0
        for symbol, pos in self.positions_dict.items():
            try:
                asset = Asset(symbol=symbol, asset_type=Asset.AssetType.STOCK)
                current_price = self.get_last_price(asset)
                if current_price:
                    value = pos['quantity'] * current_price
                    pnl = (current_price - pos['entry_price']) / pos['entry_price'] * 100
                    total_value += value
                    print(f"  {symbol}: {pos['quantity']} @ {pos['entry_price']:.2f} "
                          f"(current: {current_price:.2f}, PnL: {pnl:+.1f}%)")
            except Exception:
                pass

        print(f"  Total Position Value: {total_value:,.0f}")

    def on_abrupt_closing(self):
        """Handle abrupt closing."""
        print(f"\n{'='*60}")
        print("Abrupt Closing - Selling All Positions")
        print(f"{'='*60}")
        self.sell_all()
        self.positions_dict.clear()


# ── Backtest Entry Point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    # Configuration
    qmt_host = os.getenv("QMT_BRIDGE_HOST", "localhost")
    qmt_port = int(os.getenv("QMT_BRIDGE_PORT", "8083"))
    qmt_api_key = os.getenv("QMT_BRIDGE_API_KEY", "")
    symbols_to_trade = DEFAULT_SYMBOLS

    strategy_params = {
        "symbols": symbols_to_trade,
        "lot_size": 100,
        "max_positions": 10,
        "gap_min": 0.02,
        "gap_max": 0.10,
        "volume_multiple": 1.2,
        "min_vol_ratio": 1.3,
        "min_score_entry": 55,
        "profit_target": 0.10,
        "stop_loss": 0.04,
        "max_hold_days": 10,
    }

    # Backtest configuration
    backtesting_start_date = '2022-01-01'
    backtesting_end_date = '2024-12-31'

    # Data loading with lookback
    lookback_period = 60
    data_loading_start = pd.to_datetime(backtesting_start_date) - pd.Timedelta(days=lookback_period + 50)
    data_loading_start_str = data_loading_start.strftime('%Y-%m-%d')

    test_date = datetime.now().strftime('%Y-%m-%d')
    quant_data_dir = "/home/quant_volumn/quant_data"
    execution_folder_path = f"{quant_data_dir}/html/backtest/{test_date}/combined_portfolio"
    Path(execution_folder_path).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    base_filename = f"{execution_folder_path}/combined_portfolio_{timestamp}"

    print("=" * 60)
    print("Combined Portfolio Strategy Backtest")
    print("=" * 60)
    print(f"Sub-strategies: GapFade + Momentum")
    print(f"Order Netting: Enabled")
    print(f"Symbols: {len(symbols_to_trade)}")
    print(f"Backtest period: {backtesting_start_date} to {backtesting_end_date}")
    print("=" * 60)

    # Run backtest
    results = CombinedPortfolioStrategy.backtest(
        QMTBridgeDataBacktesting,
        pd.to_datetime(backtesting_start_date),
        pd.to_datetime(backtesting_end_date),
        benchmark_asset="000001.SS",
        sleeptime="1D",
        logfile=f"{base_filename}_log.txt",
        stats_file=f"{base_filename}_stats.csv",
        config={
            "host": qmt_host,
            "port": qmt_port,
            "api_key": qmt_api_key,
            "symbols": symbols_to_trade,
            "dividend_type": "front"
        },
        parameters=strategy_params,
    )

    print(f"\nBacktest completed: {execution_folder_path}")

    if results:
        print("\n" + "=" * 60)
        print("Backtest Results Summary")
        print("=" * 60)
        print(f"Total Return: {results.get('total_return', 'N/A')}")
        print(f"CAGR: {results.get('cagr', 'N/A')}")
        print(f"Max Drawdown: {results.get('max_drawdown', 'N/A')}")
        print(f"Sharpe Ratio: {results.get('sharpe', 'N/A')}")
        print(f"Total Trades: {results.get('total_trades', 'N/A')}")
        print("=" * 60)
