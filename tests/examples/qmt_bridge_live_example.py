"""
QMT Bridge Live Trading Example

This example demonstrates how to use QMT Bridge for live trading Chinese A-shares.

Features demonstrated:
1. Setting up QMTBridgeBroker with API authentication
2. Creating a live trading strategy
3. Order placement and position management
4. Real-time market data handling

IMPORTANT WARNINGS:
- This example places REAL orders with REAL money when connected to a live account
- Always test with a paper/simulated account first
- Never expose your API key in version control

Run with:
    python tests/examples/qmt_bridge_live_example.py
"""

import os
from datetime import datetime, time

from lumibot.brokers import QMTBridgeBroker
from lumibot.data_sources import QMTBridgeData
from lumibot.entities import Asset, Order
from lumibot.strategies import Strategy


class LiveTradingStrategy(Strategy):
    """
    Live Trading Strategy for Chinese A-shares.

    This strategy demonstrates:
    - Market order execution
    - Limit order execution
    - Position management
    - Risk management with stop-loss

    Strategy Logic:
    - Buy at market open if price is above yesterday's close
    - Use limit orders for better execution
    - Implement stop-loss for risk management
    """

    # Strategy parameters
    SYMBOL = "000001.SZ"  # Ping An Bank (平安银行)
    MAX_POSITION = 10000  # Maximum position value in CNY
    STOP_LOSS_PCT = 0.05  # 5% stop loss
    TAKE_PROFIT_PCT = 0.10  # 10% take profit

    # Trading hours (Chinese stock market)
    MARKET_OPEN = time(9, 30)
    MARKET_CLOSE = time(15, 0)
    LUNCH_START = time(11, 30)
    LUNCH_END = time(13, 0)

    def initialize(self):
        """Initialize strategy."""
        self.sleeptime = "1M"  # Check every minute
        self.entry_price = None
        self.order_placed_today = False

    def is_trading_hours(self):
        """Check if within trading hours."""
        now = self.get_datetime().time()
        is_morning = self.MARKET_OPEN <= now <= self.LUNCH_START
        is_afternoon = self.LUNCH_END <= now <= self.MARKET_CLOSE
        return is_morning or is_afternoon

    def on_trading_iteration(self):
        """Main trading logic - runs every iteration."""
        # Check trading hours
        if not self.is_trading_hours():
            self.log_message("Outside trading hours, waiting...")
            return

        # Reset daily flag at market open
        current_time = self.get_datetime().time()
        if current_time <= time(9, 35):
            self.order_placed_today = False

        asset = Asset(self.SYMBOL, asset_type="stock")
        current_price = self.get_last_price(asset)
        position = self.get_position(asset)

        if current_price is None:
            self.log_message("Unable to get current price")
            return

        self.log_message(
            f"Time: {self.get_datetime().strftime('%H:%M:%S')}, "
            f"Price: {current_price:.2f}, "
            f"Position: {position.quantity if position else 0}"
        )

        # Check for stop-loss or take-profit if we have a position
        if position and position.quantity > 0 and self.entry_price:
            pnl_pct = (current_price - self.entry_price) / self.entry_price

            # Stop-loss check
            if pnl_pct <= -self.STOP_LOSS_PCT:
                self.log_message(f"Stop-loss triggered! PnL: {pnl_pct:.2%}")
                order = self.create_order("sell", asset, position.quantity)
                self.submit_order(order)
                self.entry_price = None
                return

            # Take-profit check
            if pnl_pct >= self.TAKE_PROFIT_PCT:
                self.log_message(f"Take-profit triggered! PnL: {pnl_pct:.2%}")
                order = self.create_order("sell", asset, position.quantity)
                self.submit_order(order)
                self.entry_price = None
                return

        # Entry logic - only place one order per day
        if not self.order_placed_today and (not position or position.quantity == 0):
            # Check if we should enter a position
            if self.should_enter_position(asset, current_price):
                self.enter_position(asset, current_price)

    def should_enter_position(self, asset, current_price):
        """Determine if we should enter a position."""
        # Get yesterday's close price
        bars = self.get_historical_prices(asset, length=2, timestep="day")

        if bars is None or len(bars.df) < 2:
            return False

        yesterday_close = bars.df['close'].iloc[-2]

        # Enter if current price is above yesterday's close
        if current_price > yesterday_close:
            self.log_message(
                f"Entry signal: Price {current_price:.2f} > "
                f"Yesterday close {yesterday_close:.2f}"
            )
            return True

        return False

    def enter_position(self, asset, current_price):
        """Enter a position with limit order."""
        # Calculate position size
        cash = self.cash
        position_value = min(cash * 0.95, self.MAX_POSITION)  # Use 95% of cash, max 10k

        if current_price <= 0:
            return

        # Calculate shares (round to nearest 100 for A-shares)
        shares = int(position_value / current_price / 100) * 100

        if shares < 100:
            self.log_message("Not enough cash for minimum position")
            return

        # Place limit order slightly above current price for better fill
        limit_price = round(current_price * 1.005, 2)  # 0.5% above current

        self.log_message(
            f"Placing BUY limit order: {shares} shares @ {limit_price:.2f}"
        )

        order = self.create_order(
            side="buy",
            asset=asset,
            quantity=shares,
            limit_price=limit_price,
            order_type=Order.OrderType.LIMIT,
        )

        submitted_order = self.submit_order(order)
        self.order_placed_today = True
        self.entry_price = limit_price

        self.log_message(f"Order submitted: ID={submitted_order.identifier}")


def run_live_trading():
    """
    Run live trading with QMT Bridge.

    IMPORTANT: This will place real orders! Use with caution.
    """

    # Configuration - Load from environment variables for security
    QMT_HOST = os.environ.get("QMT_BRIDGE_HOST", "192.168.1.100")
    QMT_PORT = int(os.environ.get("QMT_BRIDGE_PORT", "8000"))
    QMT_API_KEY = os.environ.get("QMT_BRIDGE_API_KEY", "")
    QMT_ACCOUNT_ID = os.environ.get("QMT_BRIDGE_ACCOUNT_ID", "")

    if not QMT_API_KEY:
        print("WARNING: QMT_BRIDGE_API_KEY not set. Trading may fail.")
        print("Set environment variable: export QMT_BRIDGE_API_KEY='your-key'")

    # Create data source
    data_source = QMTBridgeData(
        host=QMT_HOST,
        port=QMT_PORT,
        api_key=QMT_API_KEY,
        dividend_type="none",  # No adjustment for live trading
    )

    # Create broker
    broker = QMTBridgeBroker(
        host=QMT_HOST,
        port=QMT_PORT,
        api_key=QMT_API_KEY,
        account_id=QMT_ACCOUNT_ID,
        data_source=data_source,
        connect_stream=True,  # Enable WebSocket for real-time updates
    )

    # Create strategy
    strategy = LiveTradingStrategy(broker=broker)

    # Print warning
    print("=" * 60)
    print("WARNING: LIVE TRADING MODE")
    print("=" * 60)
    print("This will place REAL orders with REAL money!")
    print("Make sure you are connected to the correct account.")
    print("=" * 60)

    # Run live trading
    # The strategy will run continuously until stopped
    strategy.run()


class PaperTradingStrategy(Strategy):
    """
    Paper Trading Strategy - Safe for testing.

    This strategy simulates trades without actually placing orders.
    Useful for testing strategy logic before going live.
    """

    SYMBOL = "000001.SZ"

    def initialize(self):
        self.sleeptime = "5M"  # Check every 5 minutes
        self.simulated_position = 0
        self.simulated_cash = 100000  # Start with 100k CNY
        self.trades = []

    def on_trading_iteration(self):
        asset = Asset(self.SYMBOL, asset_type="stock")
        current_price = self.get_last_price(asset)

        if current_price is None:
            return

        # Simple logic: buy if price < 12, sell if price > 13
        if current_price < 12.0 and self.simulated_position == 0:
            shares = int(self.simulated_cash / current_price / 100) * 100
            if shares >= 100:
                self.simulated_cash -= shares * current_price
                self.simulated_position = shares
                self.trades.append({
                    "time": self.get_datetime(),
                    "action": "BUY",
                    "shares": shares,
                    "price": current_price,
                })
                self.log_message(f"[PAPER] BUY {shares} @ {current_price:.2f}")

        elif current_price > 13.0 and self.simulated_position > 0:
            proceeds = self.simulated_position * current_price
            self.simulated_cash += proceeds
            self.trades.append({
                "time": self.get_datetime(),
                "action": "SELL",
                "shares": self.simulated_position,
                "price": current_price,
            })
            self.log_message(f"[PAPER] SELL {self.simulated_position} @ {current_price:.2f}")
            self.simulated_position = 0

        # Print portfolio status
        portfolio_value = self.simulated_cash + (self.simulated_position * current_price)
        self.log_message(
            f"[PAPER] Cash: {self.simulated_cash:.2f}, "
            f"Position: {self.simulated_position}, "
            f"Value: {portfolio_value:.2f}"
        )


def run_paper_trading():
    """Run paper trading (no real orders)."""

    QMT_HOST = os.environ.get("QMT_BRIDGE_HOST", "192.168.1.100")
    QMT_PORT = int(os.environ.get("QMT_BRIDGE_PORT", "8000"))

    data_source = QMTBridgeData(
        host=QMT_HOST,
        port=QMT_PORT,
    )

    broker = QMTBridgeBroker(
        host=QMT_HOST,
        port=QMT_PORT,
        data_source=data_source,
        connect_stream=False,  # No need for WebSocket in paper mode
    )

    strategy = PaperTradingStrategy(broker=broker)

    print("=" * 60)
    print("PAPER TRADING MODE")
    print("No real orders will be placed.")
    print("=" * 60)

    strategy.run()


# ==================== Usage Examples ====================

def example_market_order():
    """Example: Place a simple market order."""
    from lumibot.brokers import QMTBridgeBroker
    from lumibot.data_sources import QMTBridgeData
    from lumibot.entities import Asset, Order

    # Setup
    data_source = QMTBridgeData(host="192.168.1.100")
    broker = QMTBridgeBroker(
        host="192.168.1.100",
        api_key="your-api-key",
        account_id="your-account-id",
        data_source=data_source,
    )

    # Create and submit market order
    asset = Asset("000001.SZ")
    order = Order(
        strategy="example",
        asset=asset,
        quantity=100,
        side="buy",
        order_type=Order.OrderType.MARKET,
    )

    submitted = broker.submit_order(order)
    print(f"Order submitted: {submitted.identifier}")


def example_limit_order():
    """Example: Place a limit order."""
    from lumibot.brokers import QMTBridgeBroker
    from lumibot.data_sources import QMTBridgeData
    from lumibot.entities import Asset, Order

    data_source = QMTBridgeData(host="192.168.1.100")
    broker = QMTBridgeBroker(
        host="192.168.1.100",
        api_key="your-api-key",
        account_id="your-account-id",
        data_source=data_source,
    )

    # Create and submit limit order
    asset = Asset("600519.SH")  # Kweichow Moutai
    order = Order(
        strategy="example",
        asset=asset,
        quantity=100,
        side="buy",
        order_type=Order.OrderType.LIMIT,
        limit_price=1800.00,  # Limit price in CNY
    )

    submitted = broker.submit_order(order)
    print(f"Limit order submitted: {submitted.identifier}")


def example_get_positions():
    """Example: Get current positions."""
    from lumibot.brokers import QMTBridgeBroker
    from lumibot.data_sources import QMTBridgeData

    data_source = QMTBridgeData(host="192.168.1.100")
    broker = QMTBridgeBroker(
        host="192.168.1.100",
        api_key="your-api-key",
        account_id="your-account-id",
        data_source=data_source,
    )

    # Get all positions
    positions = broker._pull_positions(strategy=None)
    for pos in positions:
        print(f"{pos.asset.symbol}: {pos.quantity} shares @ {pos.avg_price}")


def example_get_historical_data():
    """Example: Get historical price data."""
    from lumibot.data_sources import QMTBridgeData
    from lumibot.entities import Asset

    data_source = QMTBridgeData(
        host="192.168.1.100",
        dividend_type="front",  # Forward-adjusted
    )

    asset = Asset("000001.SZ")

    # Get last 30 days of daily bars
    bars = data_source.get_historical_prices(
        asset=asset,
        length=30,
        timestep="day",
    )

    if bars:
        print(bars.df.head())
        print(f"\nLast close: {bars.df['close'].iloc[-1]}")


if __name__ == "__main__":
    # Choose which example to run:

    # 1. Paper trading (safe for testing)
    run_paper_trading()

    # 2. Live trading (WARNING: real money!)
    # run_live_trading()

    # 3. Just test data connection
    # example_get_historical_data()
