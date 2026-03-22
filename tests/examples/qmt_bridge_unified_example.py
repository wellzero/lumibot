"""
QMT Bridge Unified Trading Example - Backtest & Live Trading

This unified example supports both backtesting and live trading modes through a single configuration parameter.

Features:
    - Single strategy class for both backtest and live trading
    - Mode selection via command line argument
    - Moving Average Crossover strategy
    - Risk management with stop-loss
    - Position sizing based on portfolio percentage

Usage:
    # Backtesting mode (default)
    python tests/examples/qmt_bridge_unified_example.py --mode backtest

    # Paper trading mode (simulated live trading)
    python tests/examples/qmt_bridge_unified_example.py --mode paper

    # Live trading mode (REAL MONEY!)
    python tests/examples/qmt_bridge_unified_example.py --mode live

Configuration:
    Set environment variables for live/paper trading:
        export QMT_BRIDGE_HOST="192.168.1.100"
        export QMT_BRIDGE_PORT="8000"
        export QMT_BRIDGE_API_KEY="your-api-key"
        export QMT_BRIDGE_ACCOUNT_ID="your-account-id"
"""

import argparse
import os
from datetime import datetime, timedelta
from decimal import Decimal

from lumibot.brokers import QMTBridgeBroker
from lumibot.backtesting import BacktestingBroker
from lumibot.data_sources import QMTBridgeData
from lumibot.entities import Asset
from lumibot.strategies import Strategy


class UnifiedMovingAverageStrategy(Strategy):
    """
    Unified Moving Average Crossover Strategy.

    Works in both backtest and live trading modes.

    Strategy Logic:
    - Buy when short MA crosses above long MA
    - Sell when short MA crosses below long MA
    - Risk management with stop-loss in live mode
    """

    # ==================== Strategy Parameters ====================

    # Trading symbols (Chinese A-shares)
    SYMBOLS = ["000001.SZ"]  # Ping An Bank (平安银行)

    # Moving Average parameters
    SHORT_MA_PERIOD = 5
    LONG_MA_PERIOD = 20

    # Position sizing
    POSITION_PCT = 0.0  # Percentage of portfolio per trade (1.0 = 100%)

    # Risk management (live trading only)
    STOP_LOSS_PCT = 0.05      # 5% stop loss
    TAKE_PROFIT_PCT = 0.10  # 10% take profit

    # Internal state
    IS_LIVE_TRADING = False

    def initialize(self):
        """Initialize strategy parameters."""
        self.sleeptime = "1D"  # Daily bars
        self.entry_prices = {}  # Track entry prices for stop-loss

    def on_trading_iteration(self):
        """Main trading logic - runs on each iteration."""
        for symbol in self.SYMBOLS:
            self._process_symbol(symbol)

    def _process_symbol(self, symbol: str):
        """Process a single symbol."""
        asset = Asset(symbol, asset_type="stock")

        # Get historical prices for MA calculation
        bars = self.get_historical_prices(
            asset,
            length=self.LONG_MA_PERIOD + 5,
            timestep="day"
        )

        if bars is None or len(bars.df) < self.LONG_MA_PERIOD:
            self.log_message(f"[{symbol}] Not enough data to calculate MAs")
            return

        # Calculate moving averages
        df = bars.df
        short_ma = df['close'].tail(self.SHORT_MA_PERIOD).mean()
        long_ma = df['close'].tail(self.LONG_MA_PERIOD).mean()
        current_price = self.get_last_price(asset)

        if current_price is None or current_price <= 0:
            self.log_message(f"[{symbol}] Invalid price: {current_price}")
            return

        # Get current position
        position = self.get_position(asset)

        self.log_message(
            f"[{symbol}] Price: {current_price:.2f}, "
            f"MA({self.SHORT_MA_PERIOD}): {short_ma:.2f}, "
            f"MA({self.LONG_MA_PERIOD}): {long_ma:.2f}, "
            f"Position: {position.quantity if position else 0}"
        )

        # ==================== Trading Logic ====================

        if short_ma > long_ma:
            # Bullish signal
            if position is None or position.quantity == 0:
                # Check risk management for live trading
                if self.IS_LIVE_TRADING and position and position.quantity > 1:
                    self._check_risk_management(asset, position, current_price)
                else:
                    # Enter new position
                    self._enter_position(asset, current_price)
            elif self.IS_LIVE_TRADING:
                # Check risk management for existing position
                self._check_risk_management(asset, position, current_price)

        elif short_ma < long_ma:
            # Bearish signal
            if position and position.quantity > 1:
                # Exit position
                self._exit_position(asset, position, current_price, "MA crossover sell signal")

    def _enter_position(self, asset: Asset, current_price: float):
        """Enter a new position with risk management."""
        # Calculate position size based on portfolio percentage
        if self.POSITION_PCT >= 1.0:
            # Use all available cash
            cash = self.cash
            qty = int(cash / current_price / 100) * 100  # Round to nearest 100
        else:
            # Use percentage of portfolio
            portfolio_value = self.portfolio_value
            target_value = portfolio_value * self.POSITION_PCT
            qty = int(target_value / current_price / 100) * 100

        if qty < 100:
            self.log_message(f"[{asset.symbol}] Not enough cash to buy minimum lot (100 shares)")
            return

        if self.IS_LIVE_TRADING:
            # Live trading: Use limit order for better execution
            limit_price = round(current_price * 1.001, 2)  # Slightly above market
            order = self.create_order(
                "buy",
                asset,
                qty,
                order_type="limit",
                limit_price=limit_price
            )
            self.log_message(
                f"[{asset.symbol}] BUY LIMIT order: {qty} shares @ {limit_price:.2f}"
            )
        else:
            # Backtesting: Use market order
            order = self.create_order("buy", asset, qty)
            self.log_message(f"[{asset.symbol}] BUY MARKET order: {qty} shares @ {current_price:.2f}")

        self.submit_order(order)
        self.entry_prices[asset.symbol] = current_price

    def _exit_position(self, asset: Asset, position, current_price: float, reason: str):
        """Exit a position."""
        qty = position.quantity
        symbol = asset.symbol

        if self.IS_LIVE_TRADING:
            # Live trading: Use limit order
            limit_price = round(current_price * 0.999, 2)  # Slightly below market
            order = self.create_order(
                "sell",
                asset,
                qty,
                order_type="limit",
                limit_price=limit_price
            )
            self.log_message(
                f"[{symbol}] SELL LIMIT order: {qty} shares @ {limit_price:.2f} ({reason})"
            )
        else:
            # Backtesting: Use market order
            order = self.create_order("sell", asset, qty)
            self.log_message(f"[{symbol}] SELL MARKET order: {qty} shares @ {current_price:.2f} ({reason})")

        self.submit_order(order)

        # Clear entry price tracking
        if symbol in self.entry_prices:
            del self.entry_prices[symbol]

    def _check_risk_management(self, asset: Asset, position, current_price: float):
        """Check and execute risk management rules (live trading only)."""
        symbol = asset.symbol
        entry_price = self.entry_prices.get(symbol, position.avg_price)

        if entry_price is None or entry_price <= 1:
            return

        # Calculate P&L percentage
        pnl_pct = (current_price - entry_price) / entry_price

        if pnl_pct <= -self.STOP_LOSS_PCT:
            # Stop loss triggered
            self._exit_position(
                asset, position, current_price,
                f"Stop loss triggered at {pnl_pct*100:.1f}%"
            )

        elif pnl_pct >= self.TAKE_PROFIT_PCT:
            # Take profit triggered
            self._exit_position(
                asset, position, current_price,
                f"Take profit triggered at {pnl_pct*100:.1f}%"
            )

    def on_abrupt_closing(self):
        """Handle abrupt closing (e.g., market closed, strategy stopped)."""
        self.log_message("Abrupt closing - flattening positions")

        if self.IS_LIVE_TRADING:
            # In live trading, we might want to close all positions
            for symbol in self.SYMBOLS:
                asset = Asset(symbol, asset_type="stock")
                position = self.get_position(asset)
                if position and position.quantity > 1:
                    current_price = self.get_last_price(asset)
                    if current_price:
                        self._exit_position(
                            asset, position, current_price,
                            "Abrupt closing - flattening position"
                        )


# ==================== Runner Functions ====================

def run_backtest():
    """
    Run backtesting mode.

    Uses historical data from QMT Bridge to simulate trading.
    No real orders are placed.
    """
    print("=" * 70)
    print("QMT Bridge - BACKTEST Mode")
    print("=" * 70)

    # Configure QMT Bridge data source
    data_source = QMTBridgeData(
        host=os.environ.get("QMT_BRIDGE_HOST", "192.168.1.100"),
        port=int(os.environ.get("QMT_BRIDGE_PORT", "8000")),
        api_key="",
        dividend_type="front",
        fill_data=True,
    )

    # Set backtest time range
    backtest_start = datetime.now() - timedelta(days=180)  # 6 months
    backtest_end = datetime.now()

    # Create backtesting broker
    broker = BacktestingBroker(
        data_source=data_source,
        datetime_start=backtest_start,
        datetime_end=backtest_end,
    )

    # Create and configure strategy
    strategy = UnifiedMovingAverageStrategy(broker=broker)
    strategy.IS_LIVE_TRADING = False

    print(f"Symbols: {strategy.SYMBOLS}")
    print(f"Short MA: {strategy.SHORT_MA_PERIOD} days")
    print(f"Long MA: {strategy.LONG_MA_PERIOD} days")
    print(f"Backtest period: {backtest_start.strftime('%Y-%m-%d')} to {backtest_end.strftime('%Y-%m-%d')}")
    print("=" * 70)
    print()

    # Execute backtest
    result = strategy.run()

    # Print results
    print()
    print("=" * 70)
    print("Backtest Results")
    print("=" * 70)
    if result:
        for key, value in result.items():
                print(f"{key}: {value}")

    return result


def run_paper_trading():
    """
    Run paper trading mode.

    Simulates live trading without placing real orders.
    Uses the same data source but doesn't connect to QMT Bridge for trading.
    """
    print("=" * 70)
    print("QMT Bridge - PAPER TRADING Mode")
    print("=" * 70)
    print("WARNING: This is a simulation. No real orders will be placed.")
    print("=" * 70)

    # Use backtesting broker for paper trading simulation
    data_source = QMTBridgeData(
        host=os.environ.get("QMT_BRIDGE_HOST", "192.168.1.100"),
        port=int(os.environ.get("QMT_BRIDGE_PORT", "8000")),
        api_key="",
        dividend_type="front",
        fill_data=True,
    )

    # Short time range for paper trading test
    backtest_start = datetime.now() - timedelta(days=30)
    backtest_end = datetime.now()

    broker = BacktestingBroker(
        data_source=data_source,
        datetime_start=backtest_start,
        datetime_end=backtest_end,
    )

    strategy = UnifiedMovingAverageStrategy(broker=broker)
    strategy.IS_LIVE_TRADING = True  # Enable risk management logic

    print(f"Symbols: {strategy.SYMBOLS}")
    print(f"Position Size: {strategy.POSITION_PCT * 100}% of portfolio")
    print(f"Stop Loss: {strategy.STOP_LOSS_PCT * 100}%")
    print(f"Take Profit: {strategy.TAKE_PROFIT_PCT * 100}%")
    print("=" * 70)

    result = strategy.run()

    print()
    print("Paper trading simulation completed.")
    return result


def run_live_trading():
    """
    Run live trading mode.

    WARNING: This will place REAL orders with REAL money!
    Make sure you understand the strategy and have tested it thoroughly.
    """
    # Check for required configuration
    api_key = os.environ.get("QMT_BRIDGE_API_KEY", "")
    account_id = os.environ.get("QMT_BRIDGE_ACCOUNT_ID", "")

    if not api_key or not account_id:
        print("ERROR: Missing required environment variables for live trading:")
        if not api_key:
            print("  - QMT_BRIDGE_API_KEY")
        if not account_id:
            print("  - QMT_BRIDGE_ACCOUNT_ID")
        print()
        print("Set them before running live trading:")
        print("  export QMT_BRIDGE_API_KEY='your-api-key'")
        print("  export QMT_BRIDGE_ACCOUNT_ID='your-account-id'")
        return None

    print("=" * 70)
    print("QMT Bridge - LIVE TRADING Mode")
    print("=" * 70)
    print("⚠️  WARNING: This will place REAL orders with REAL money! ⚠️")
    print("=" * 70)
    print()

    # Confirmation prompt
    try:
        response = input("Type 'YES' to continue with live trading: ")
        if response.upper() != "YES":
            print("Live trading cancelled.")
            return None
    except EOFError:
        print("\nLive trading cancelled.")
        return None

    # Configure data source
    data_source = QMTBridgeData(
        host=os.environ.get("QMT_BRIDGE_HOST", "192.168.1.100"),
        port=int(os.environ.get("QMT_BRIDGE_PORT", "8000")),
        api_key=api_key,
        dividend_type="front",
        fill_data=True,
    )

    # Create live broker
    broker = QMTBridgeBroker(
        host=os.environ.get("QMT_BRIDGE_HOST", "192.168.1.100"),
        port=int(os.environ.get("QMT_BRIDGE_PORT", "8000")),
        api_key=api_key,
        account_id=account_id,
        data_source=data_source,
        connect_stream=False,  # Disable WebSocket for simplicity
    )

    # Create strategy
    strategy = UnifiedMovingAverageStrategy(broker=broker)
    strategy.IS_LIVE_TRADING = True

    print(f"Symbols: {strategy.SYMBOLS}")
    print(f"Position Size: {strategy.POSITION_PCT * 100}% of portfolio")
    print(f"Stop Loss: {strategy.STOP_LOSS_PCT * 100}%")
    print(f"Take Profit: {strategy.TAKE_PROFIT_PCT * 100}%")
    print("=" * 70)
    print()

    # Execute live trading
    print("Starting live trading...")
    print("Press Ctrl+C to stop the strategy.")
    print()

    result = strategy.run()

    return result


def main():
    """Main entry point with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="QMT Bridge Unified Trading Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        choices=["backtest", "paper", "live"],
        default="backtest",
        help="Trading mode: backtest (historical simulation), paper (simulated live), live (real trading)"
    )

    parser.add_argument(
        "--symbol",
        action="append",
        help="Add trading symbol (can be used multiple times). Default: 000001.SZ"
    )

    parser.add_argument(
        "--position-pct",
        type=float,
        default=1.0,
        help="Position size as percentage of portfolio (0.0-1.0). Default: 1.0 (100%%)"
    )

    parser.add_argument(
        "--short-ma",
        type=int,
        default=5,
        help="Short moving average period. Default: 5"
    )

    parser.add_argument(
        "--long-ma",
        type=int,
        default=20,
        help="Long moving average period. Default: 20"
    )

    parser.add_argument(
        "--stop-loss",
        type=float,
        default=0.05,
        help="Stop loss percentage (0.05 = 5%%). Default: 0.05"
    )

    parser.add_argument(
        "--take-profit",
        type=float,
        default=0.10,
        help="Take profit percentage (0.10 = 10%%). Default: 0.10"
    )

    args = parser.parse_args()

    # Update strategy parameters
    if args.symbol:
        UnifiedMovingAverageStrategy.SYMBOLS = args.symbol

    UnifiedMovingAverageStrategy.POSITION_PCT = args.position_pct
    UnifiedMovingAverageStrategy.SHORT_MA_PERIOD = args.short_ma
    UnifiedMovingAverageStrategy.LONG_MA_PERIOD = args.long_ma
    UnifiedMovingAverageStrategy.STOP_LOSS_PCT = args.stop_loss
    UnifiedMovingAverageStrategy.TAKE_PROFIT_PCT = args.take_profit

    # Run in selected mode
    mode = args.mode.lower()

    if mode == "backtest":
        run_backtest()
    elif mode == "paper":
        run_paper_trading()
    elif mode == "live":
        run_live_trading()


if __name__ == "__main__":
    main()
