#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Combined Portfolio Strategy - Policy Gap Retail Reversal + Institutional Flow Divergence

This example demonstrates how to combine two EXISTING strategies using the
PortfolioStrategy system with order netting.

The strategies are imported directly without modification:
- PolicyGapRetailReversal: Contrarian gap fade strategy
- InstitutionalFlowDivergence: Momentum breakout strategy

Order netting example:
    - GapFade signals: BUY 000001.SZ x 100
    - Momentum signals: SELL 000001.SZ x 100
    - Net result: 0 trades (orders cancelled out, fees saved!)
"""

import os
import sys
import pandas as pd
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from lumibot.strategies import Strategy
from lumibot.entities import Asset
from lumibot.backtesting import QMTBridgeDataBacktesting
from lumibot.strategies.portfolio_strategy import PortfolioStrategy
from lumibot.credentials import IS_BACKTESTING

# ── Import the actual strategies ──────────────────────────────────────────────
# Add paths for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import existing strategies
from policy_gap_retail.cn_policy_gap_retail_reversal_qmt_bridge_data import (
    PolicyGapRetailReversal,
    DEFAULT_STOCKS as GAP_FADE_SYMBOLS
)
from institutional_flow_divergence.csi300_institutional_flow_divergence import (
    InstitutionalFlowDivergence,
    StrategyParams as MomentumParams
)

# ── Environment Setup ──────────────────────────────────────────────────────────

QMT_BRIDGE_ENV_PATH = "/home/quant_volumn/docker/data/qmt-bridge/.env"
load_dotenv(QMT_BRIDGE_ENV_PATH)


# ── Combined Portfolio Strategy ────────────────────────────────────────────────

class CombinedPortfolioStrategy(Strategy):
    """
    Combined portfolio strategy that runs multiple existing strategies
    with order netting.

    This class:
    1. Imports existing strategies without modification
    2. Runs them through PortfolioStrategy wrapper
    3. Intercepts their orders and converts to signals
    4. Nets conflicting signals
    5. Executes only the net difference
    """

    def initialize(self,
                   symbols=None,
                   lot_size=100,
                   max_positions=12,
                   # Exit parameters (applied at portfolio level)
                   profit_target=0.10,
                   stop_loss=0.05,
                   max_hold_days=15,
                   # Gap Fade strategy params
                   gap_min=0.02,
                   gap_max=0.10,
                   volume_multiple=1.2,
                   # Momentum strategy params
                   min_vol_ratio=1.3,
                   min_score_entry=55):
        """
        Initialize the combined portfolio strategy.
        """
        self.sleeptime = "1D"

        # Merge symbols from both strategies
        self.symbols = symbols or list(set(GAP_FADE_SYMBOLS))

        self.lot_size = lot_size
        self.max_positions = max_positions
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        self.max_hold_days = max_hold_days

        # Position tracking
        self.positions_dict = {}

        # Create portfolio manager
        self.portfolio = PortfolioStrategy(self, min_net_quantity=lot_size)

        # Add existing strategies (they are imported, not modified)
        self.portfolio.add_strategy(
            PolicyGapRetailReversal,
            "gap_fade",
            symbols=self.symbols,
            gap_min=gap_min,
            gap_max=gap_max,
            volume_multiple=volume_multiple,
            max_positions=max_positions // 2,
        )

        self.portfolio.add_strategy(
            InstitutionalFlowDivergence,
            "momentum",
            symbols=self.symbols,
            params=MomentumParams(),
        )

        print(f"\n{'='*70}")
        print("Combined Portfolio Strategy Initialized")
        print(f"{'='*70}")
        print(f"Imported Strategies:")
        print(f"  1. PolicyGapRetailReversal (gap_fade)")
        print(f"     - Type: Contrarian / Gap Fade")
        print(f"     - Entry: Overnight gap DOWN {gap_min:.0%}-{gap_max:.0%}")
        print(f"  2. InstitutionalFlowDivergence (momentum)")
        print(f"     - Type: Momentum / Breakout")
        print(f"     - Entry: Score >= {min_score_entry}")
        print(f"")
        print(f"Portfolio Settings:")
        print(f"  - Symbols: {len(self.symbols)}")
        print(f"  - Order netting: Enabled (min_qty={lot_size})")
        print(f"  - Exit: {profit_target:.0%} profit / {stop_loss:.0%} stop / {max_hold_days}d max")
        print(f"{'='*70}\n")

    def on_trading_iteration(self):
        """Main trading iteration."""
        current_date = self.get_datetime()
        print(f"\n{'='*70}")
        print(f"Trading Iteration: {current_date.strftime('%Y-%m-%d')}")
        print(f"{'='*70}")

        # 1. Check exit conditions for existing positions
        self._check_exit_conditions()

        # 2. Run all imported strategies (collects their orders as signals)
        print("\nRunning imported strategies...")
        signals = self.portfolio.run_iteration()

        # 3. Print signal summary
        print(f"\nSignals generated: {len(signals)}")
        for sig in signals:
            side = "BUY" if sig.quantity > 0 else "SELL"
            print(f"  [{sig.strategy_id}] {sig.symbol}: {side} {abs(sig.quantity)}")

        # 4. Get netted orders
        netted_orders = self.portfolio.get_netted_orders()

        # 5. Print netting summary
        summary = self.portfolio.get_netting_summary()
        print(f"\n{'─'*70}")
        print("Order Netting Summary:")
        print(f"{'─'*70}")
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

        # 6. Execute netted orders
        if netted_orders:
            print(f"\nExecuting {len(netted_orders)} netted orders:")
            self._execute_netted_orders(netted_orders)
        else:
            print(f"\nNo orders to execute (all netted to zero or below threshold)")

        # 7. Print current positions
        self._print_positions()

        self.await_market_to_close()

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

                # Create and submit order through main strategy
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

                # Update portfolio positions
                fill_qty = quantity if netted.is_buy else -quantity
                self.portfolio.update_positions_after_fill(order, last_price, abs(fill_qty))

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
        print(f"\n{'='*70}")
        print("Abrupt Closing - Selling All Positions")
        print(f"{'='*70}")
        self.sell_all()
        self.positions_dict.clear()


# ── Backtest Entry Point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    # Configuration
    qmt_host = os.getenv("QMT_BRIDGE_HOST", "localhost")
    qmt_port = int(os.getenv("QMT_BRIDGE_PORT", "8083"))
    qmt_api_key = os.getenv("QMT_BRIDGE_API_KEY", "")
    symbols_to_trade = list(set(GAP_FADE_SYMBOLS))  # Use symbols from Gap Fade strategy

    strategy_params = {
        "symbols": symbols_to_trade,
        "lot_size": 100,
        "max_positions": 12,
        "gap_min": 0.02,
        "gap_max": 0.10,
        "volume_multiple": 1.2,
        "min_vol_ratio": 1.3,
        "min_score_entry": 55,
        "profit_target": 0.10,
        "stop_loss": 0.05,
        "max_hold_days": 15,
    }

    # Backtest configuration
    backtesting_start_date = '2022-01-01'
    backtesting_end_date = '2024-12-31'

    test_date = datetime.now().strftime('%Y-%m-%d')
    quant_data_dir = "/home/quant_volumn/quant_data"
    execution_folder_path = f"{quant_data_dir}/html/backtest/{test_date}/combined_portfolio"
    Path(execution_folder_path).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    base_filename = f"{execution_folder_path}/combined_portfolio_{timestamp}"

    print("=" * 70)
    print("Combined Portfolio Strategy Backtest")
    print("=" * 70)
    print(f"Imported Strategies:")
    print(f"  - PolicyGapRetailReversal (contrarian gap fade)")
    print(f"  - InstitutionalFlowDivergence (momentum breakout)")
    print(f"Order Netting: ENABLED")
    print(f"Symbols: {len(symbols_to_trade)}")
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
            "symbols": symbols_to_trade,
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
