#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Combined Portfolio Strategy - Policy Gap Retail Reversal + Institutional Flow Divergence

This example demonstrates how to combine signals from multiple strategies using
the PortfolioStrategyHelper with order netting.

Order netting example:
    - GapFade signals: BUY 000001.SZ x 100
    - Momentum signals: SELL 000001.SZ x 100
    - Net result: 0 trades (orders cancelled out, fees saved!)

The key is to separate SIGNAL GENERATION from ORDER EXECUTION:
1. Each strategy has a generate_signals() method that returns signals
2. Main strategy collects signals from all sub-strategies
3. PortfolioHelper nets the signals
4. Main strategy executes only the net difference
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict, Any

from lumibot.strategies import Strategy
from lumibot.entities import Asset
from lumibot.backtesting import QMTBridgeDataBacktesting
from lumibot.strategies.portfolio_strategy import (
    PortfolioStrategyHelper,
    TradingSignal,
    NettedOrder,
)

# ── Environment Setup ──────────────────────────────────────────────────────────

QMT_BRIDGE_ENV_PATH = "/home/quant_volumn/docker/data/qmt-bridge/.env"
load_dotenv(QMT_BRIDGE_ENV_PATH)

# Default symbols for the combined strategy
DEFAULT_SYMBOLS = [
    '300750.SZ', '002129.SZ', '601865.SH', '002594.SZ',
    '300274.SZ', '600900.SH', '601615.SH', '002202.SZ',
    '002475.SZ', '002415.SZ', '300059.SZ', '600588.SH',
    '002230.SZ', '002049.SZ',
    '600519.SH', '000858.SZ', '600887.SH', '000333.SZ',
    '000651.SZ', '601888.SH',
]

# Sector mapping
SECTOR_MAP = {
    '300750.SZ': 'Renewable', '002129.SZ': 'Renewable', '601865.SH': 'Renewable',
    '002594.SZ': 'Renewable', '300274.SZ': 'Renewable', '600900.SH': 'Renewable',
    '601615.SH': 'Renewable', '002202.SZ': 'Renewable',
    '002475.SZ': 'Tech', '002415.SZ': 'Tech', '300059.SZ': 'Tech',
    '600588.SH': 'Tech', '002230.SZ': 'Tech', '002049.SZ': 'Tech',
    '600519.SH': 'Consumption', '000858.SZ': 'Consumption', '600887.SH': 'Consumption',
    '000333.SZ': 'Consumption', '000651.SZ': 'Consumption', '601888.SH': 'Consumption',
}


# ── Signal Generator 1: Gap Fade ───────────────────────────────────────────────

class GapFadeSignalGenerator:
    """
    Contrarian gap fade signal generator.

    Generates BUY signals when overnight gap DOWN 2-10% with volume confirmation.
    This is a contrarian approach - buying the dip when others panic.
    """

    def __init__(self, strategy_id: str = "gap_fade",
                 gap_min: float = 0.02, gap_max: float = 0.10,
                 volume_multiple: float = 1.2, position_size: int = 100):
        self.strategy_id = strategy_id
        self.gap_min = gap_min
        self.gap_max = gap_max
        self.volume_multiple = volume_multiple
        self.position_size = position_size

    def generate_signals(self, strategy: Strategy, symbols: List[str],
                        price_history: Dict[str, pd.DataFrame]) -> List[TradingSignal]:
        """
        Generate trading signals based on gap fade logic.

        Args:
            strategy: The main strategy (for datetime access)
            symbols: List of symbols to analyze
            price_history: Dict mapping symbol -> DataFrame with OHLCV data

        Returns:
            List of TradingSignal objects
        """
        signals = []

        for symbol in symbols:
            if symbol not in price_history:
                continue

            df = price_history[symbol]
            if len(df) < 22:
                continue

            try:
                close = df['close']
                open_prices = df.get('open', close)
                volume = df.get('volume', pd.Series([1] * len(df)))

                # Today's open (execution price)
                today_open = open_prices.iloc[-1]
                # Yesterday's close
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

                # Generate BUY signal
                signal = TradingSignal(
                    strategy_id=self.strategy_id,
                    symbol=symbol,
                    quantity=self.position_size,  # Positive = buy
                    price=today_open,
                    metadata={'gap': gap, 'vol_ratio': vol_ratio, 'sector': sector}
                )
                signals.append(signal)

                print(f"[GapFade] {symbol}: BUY signal (gap={gap:.1%}, vol={vol_ratio:.1f}x)")

            except Exception as e:
                print(f"[GapFade] Error processing {symbol}: {e}")

        return signals


# ── Signal Generator 2: Momentum ───────────────────────────────────────────────

class MomentumSignalGenerator:
    """
    Momentum breakout signal generator.

    Generates BUY signals when price breaks out with volume confirmation.
    This is a trend-following approach - buying strength.
    """

    def __init__(self, strategy_id: str = "momentum",
                 min_vol_ratio: float = 1.3, min_score: int = 55,
                 position_size: int = 100):
        self.strategy_id = strategy_id
        self.min_vol_ratio = min_vol_ratio
        self.min_score = min_score
        self.position_size = position_size

    def _compute_score(self, df: pd.DataFrame) -> float:
        """Compute entry score using data available before market open."""
        try:
            close = df['close'].values
            volume = df['volume'].values
            high = df['high'].values
            low = df['low'].values
            n = len(close)
            if n < 30:
                return 0

            # Use yesterday's data (iloc[-1] is yesterday in our pre-loaded data)
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
            return 0

    def generate_signals(self, strategy: Strategy, symbols: List[str],
                        price_history: Dict[str, pd.DataFrame]) -> List[TradingSignal]:
        """Generate trading signals based on momentum logic."""
        signals = []

        for symbol in symbols:
            if symbol not in price_history:
                continue

            df = price_history[symbol]
            if len(df) < 30:
                continue

            try:
                score = self._compute_score(df)

                if score >= self.min_score:
                    # Generate BUY signal
                    signal = TradingSignal(
                        strategy_id=self.strategy_id,
                        symbol=symbol,
                        quantity=self.position_size,  # Positive = buy
                        price=df['close'].iloc[-1],
                        metadata={'score': score}
                    )
                    signals.append(signal)
                    print(f"[Momentum] {symbol}: BUY signal (score={score:.0f})")

            except Exception as e:
                print(f"[Momentum] Error processing {symbol}: {e}")

        return signals


# ── Combined Portfolio Strategy ────────────────────────────────────────────────

class CombinedPortfolioStrategy(Strategy):
    """
    Combined portfolio strategy using signal generators with order netting.

    This demonstrates:
    1. Separating signal generation from order execution
    2. Multiple signal generators with different logic
    3. Order netting to reduce unnecessary trades
    """

    def initialize(self,
                   symbols=None,
                   lot_size=100,
                   max_positions=12,
                   # Exit parameters
                   profit_target=0.10,
                   stop_loss=0.05,
                   max_hold_days=15,
                   # Gap Fade params
                   gap_min=0.02,
                   gap_max=0.10,
                   volume_multiple=1.2,
                   # Momentum params
                   min_vol_ratio=1.3,
                   min_score_entry=55):
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

        # Create portfolio helper
        self.portfolio = PortfolioStrategyHelper(min_net_quantity=lot_size)

        # Create signal generators
        self.gap_fade = GapFadeSignalGenerator(
            strategy_id="gap_fade",
            gap_min=gap_min,
            gap_max=gap_max,
            volume_multiple=volume_multiple,
            position_size=100
        )

        self.momentum = MomentumSignalGenerator(
            strategy_id="momentum",
            min_vol_ratio=min_vol_ratio,
            min_score=min_score_entry,
            position_size=100
        )

        print(f"\n{'='*70}")
        print("Combined Portfolio Strategy Initialized")
        print(f"{'='*70}")
        print(f"Signal Generators:")
        print(f"  1. GapFade (contrarian) - buys on gap down")
        print(f"  2. Momentum (trend) - buys on breakout")
        print(f"Order Netting: ENABLED")
        print(f"Symbols: {len(self.symbols)}")
        print(f"{'='*70}\n")

    def on_trading_iteration(self):
        """Main trading iteration."""
        current_date = self.get_datetime()
        print(f"\n{'='*70}")
        print(f"Trading Iteration: {current_date.strftime('%Y-%m-%d')}")
        print(f"{'='*70}")

        # 1. Check exit conditions
        self._check_exit_conditions()

        # 2. Update price history
        self._update_price_history()

        # 3. Start new iteration (clears old signals)
        self.portfolio.new_iteration()

        # 4. Collect signals from all generators
        print("\n--- Generating Signals ---")
        signals = []

        # Gap Fade signals
        gap_signals = self.gap_fade.generate_signals(
            self, self.symbols, self.price_history
        )
        signals.extend(gap_signals)
        print(f"GapFade: {len(gap_signals)} signals")

        # Momentum signals
        mom_signals = self.momentum.generate_signals(
            self, self.symbols, self.price_history
        )
        signals.extend(mom_signals)
        print(f"Momentum: {len(mom_signals)} signals")

        # 5. Add signals to netting engine
        self.portfolio.add_signals(signals)

        # 6. Get netted orders
        netted_orders = self.portfolio.get_netted_orders()

        # 7. Print netting summary
        summary = self.portfolio.get_netting_summary()
        print(f"\n--- Order Netting Summary ---")
        for symbol, details in summary["netting_details"].items():
            net_qty = details["net_quantity"]
            strategies = ", ".join(details["strategies"])
            signals_str = ", ".join([
                f"{s['strategy']}:{s['qty']:+.0f}"
                for s in details["signals"]
            ])
            action = "EXECUTE" if abs(net_qty) >= self.lot_size else "SKIP (netted)"
            print(f"  {symbol}:")
            print(f"    Signals: [{signals_str}]")
            print(f"    Net: {net_qty:+.0f} → {action}")

        # 8. Execute netted orders
        if netted_orders:
            print(f"\n--- Executing {len(netted_orders)} Netted Orders ---")
            self.portfolio.execute_netted_orders(
                netted_orders, self,
                lot_size=self.lot_size,
                max_positions=self.max_positions,
                position_dict=self.positions_dict
            )
        else:
            print(f"\nNo orders to execute (all netted to zero)")

        # 9. Print positions
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
                if 'highest_price' not in pos:
                    pos['highest_price'] = entry_price
                if current_price > pos['highest_price']:
                    pos['highest_price'] = current_price

                should_sell = False
                reason = ""

                # Exit 1: Profit target
                if pnl_pct >= self.profit_target:
                    should_sell = True
                    reason = f"Profit target ({pnl_pct:.1%})"

                # Exit 2: Stop loss
                elif pnl_pct <= -self.stop_loss:
                    should_sell = True
                    reason = f"Stop loss ({pnl_pct:.1%})"

                # Exit 3: Max hold days
                elif days_held >= self.max_hold_days:
                    should_sell = True
                    reason = f"Max hold ({days_held}d)"

                if should_sell:
                    quantity = pos['quantity']
                    order = self.create_order(asset, quantity, "sell")
                    self.submit_order(order)
                    print(f"  SELL {symbol} x {quantity} @ {current_price:.2f} ({reason})")
                    del self.positions_dict[symbol]

            except Exception:
                pass

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
        print(f"\n{'='*70}")
        print("Abrupt Closing - Selling All Positions")
        print(f"{'='*70}")
        self.sell_all()
        self.positions_dict.clear()


# ── Live Trading Mode ──────────────────────────────────────────────────────────

def run_live_trading(symbols, strategy_params):
    """Run live trading with QMT Bridge."""
    from lumibot.data_sources import QMTBridgeData
    from lumibot.brokers import QMTBridgeBroker
    from lumibot.traders import Trader

    qmt_host = os.getenv("QMT_BRIDGE_HOST", "localhost")
    qmt_port = int(os.getenv("QMT_BRIDGE_PORT", "8083"))
    qmt_api_key = os.getenv("QMT_BRIDGE_API_KEY", "")
    qmt_account_id = os.getenv("QMT_BRIDGE_TRADING_ACCOUNT_ID", "")

    if not qmt_account_id:
        print("ERROR: QMT_BRIDGE_TRADING_ACCOUNT_ID is required for live trading")
        print("Set it in your .env file or environment variables")
        sys.exit(1)

    print("=" * 70)
    print("Combined Portfolio Strategy - LIVE TRADING")
    print("=" * 70)
    print(f"QMT Bridge Host: {qmt_host}")
    print(f"QMT Bridge Port: {qmt_port}")
    print(f"Account ID: {qmt_account_id}")
    print(f"Symbols: {len(symbols)}")
    print("=" * 70)
    print("Signal Generators:")
    print("  1. GapFade (contrarian - buys on gap down)")
    print("  2. Momentum (trend - buys on breakout)")
    print("Order Netting: ENABLED")
    print("=" * 70)
    print("\nStarting live trading... (Ctrl+C to stop)\n")

    # Create data source for live market data
    data_source = QMTBridgeData(
        host=qmt_host,
        port=qmt_port,
        api_key=qmt_api_key,
    )

    # Create broker for live order execution
    broker = QMTBridgeBroker(
        host=qmt_host,
        port=qmt_port,
        api_key=qmt_api_key,
        account_id=qmt_account_id,
        data_source=data_source,
        connect_stream=True,
    )

    # Create strategy instance
    strategy = CombinedPortfolioStrategy(
        broker=broker,
        parameters=strategy_params,
    )

    # Run live trading
    trader = Trader(backtest=False)
    trader.add_strategy(strategy)
    trader.run_all()


# ── Backtest Mode ──────────────────────────────────────────────────────────────

def run_backtest(symbols, strategy_params):
    """Run backtest with QMT Bridge data."""
    qmt_host = os.getenv("QMT_BRIDGE_HOST", "localhost")
    qmt_port = int(os.getenv("QMT_BRIDGE_PORT", "8083"))
    qmt_api_key = os.getenv("QMT_BRIDGE_API_KEY", "")

    backtesting_start_date = '2022-01-01'
    backtesting_end_date = '2024-12-31'

    test_date = datetime.now().strftime('%Y-%m-%d')
    quant_data_dir = "/home/quant_volumn/quant_data"
    execution_folder_path = f"{quant_data_dir}/html/backtest/{test_date}/combined_portfolio"
    Path(execution_folder_path).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    base_filename = f"{execution_folder_path}/combined_portfolio_{timestamp}"

    print("=" * 70)
    print("Combined Portfolio Strategy - BACKTEST")
    print("=" * 70)
    print(f"Signal Generators:")
    print(f"  - GapFade (contrarian - buys on gap down)")
    print(f"  - Momentum (trend - buys on breakout)")
    print(f"Order Netting: ENABLED")
    print(f"Symbols: {len(symbols)}")
    print(f"Backtest: {backtesting_start_date} to {backtesting_end_date}")
    print("=" * 70)

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
            "symbols": symbols,
            "dividend_type": "front"
        },
        parameters=strategy_params,
    )

    print(f"\nBacktest completed: {execution_folder_path}")

    if results:
        print("\n" + "=" * 70)
        print("Backtest Results Summary")
        print("=" * 70)
        print(f"Total Return: {results.get('total_return', 'N/A')}")
        print(f"CAGR: {results.get('cagr', 'N/A')}")
        print(f"Max Drawdown: {results.get('max_drawdown', 'N/A')}")
        print(f"Sharpe Ratio: {results.get('sharpe', 'N/A')}")
        print(f"Total Trades: {results.get('total_trades', 'N/A')}")
        print("=" * 70)

    return results


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Combined Portfolio Strategy with Order Netting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run backtest (default)
  python combined_portfolio_strategy.py

  # Run live trading
  python combined_portfolio_strategy.py --live

  # Run live trading with custom symbols
  python combined_portfolio_strategy.py --live --symbols 000001.SZ,600519.SH,300750.SZ

  # Run backtest with custom parameters
  python combined_portfolio_strategy.py --gap-min 0.03 --profit-target 0.08
        """
    )

    # Mode selection
    parser.add_argument(
        "--live", action="store_true",
        help="Run live trading (default: backtest)"
    )

    # Symbol selection
    parser.add_argument(
        "--symbols", type=str, default=None,
        help="Comma-separated list of symbols (default: use DEFAULT_SYMBOLS)"
    )

    # Strategy parameters
    parser.add_argument("--lot-size", type=int, default=100, help="Lot size (default: 100)")
    parser.add_argument("--max-positions", type=int, default=12, help="Max positions (default: 12)")
    parser.add_argument("--profit-target", type=float, default=0.10, help="Profit target (default: 0.10)")
    parser.add_argument("--stop-loss", type=float, default=0.05, help="Stop loss (default: 0.05)")
    parser.add_argument("--max-hold-days", type=int, default=15, help="Max hold days (default: 15)")
    parser.add_argument("--gap-min", type=float, default=0.02, help="Min gap (default: 0.02)")
    parser.add_argument("--gap-max", type=float, default=0.10, help="Max gap (default: 0.10)")
    parser.add_argument("--volume-multiple", type=float, default=1.2, help="Volume multiple (default: 1.2)")
    parser.add_argument("--min-vol-ratio", type=float, default=1.3, help="Min volume ratio (default: 1.3)")
    parser.add_argument("--min-score", type=int, default=55, help="Min momentum score (default: 55)")

    args = parser.parse_args()

    # Parse symbols
    if args.symbols:
        symbols_to_trade = [s.strip() for s in args.symbols.split(",")]
    else:
        symbols_to_trade = DEFAULT_SYMBOLS

    # Build strategy parameters
    strategy_params = {
        "symbols": symbols_to_trade,
        "lot_size": args.lot_size,
        "max_positions": args.max_positions,
        "profit_target": args.profit_target,
        "stop_loss": args.stop_loss,
        "max_hold_days": args.max_hold_days,
        "gap_min": args.gap_min,
        "gap_max": args.gap_max,
        "volume_multiple": args.volume_multiple,
        "min_vol_ratio": args.min_vol_ratio,
        "min_score_entry": args.min_score,
    }

    # Run in appropriate mode
    if args.live:
        run_live_trading(symbols_to_trade, strategy_params)
    else:
        run_backtest(symbols_to_trade, strategy_params)
