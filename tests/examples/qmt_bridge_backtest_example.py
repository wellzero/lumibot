"""
QMT Bridge Backtesting Example

This example demonstrates how to use QMT Bridge for backtesting Chinese A-shares strategies.

Features demonstrated:
1. Setting up QMTBridgeData with dividend adjustment
2. Creating a simple moving average crossover strategy
3. Running backtests with Chinese stock data
4. Analyzing backtest results

Run with:
    python tests/examples/qmt_bridge_backtest_example.py
"""

from datetime import datetime, timedelta

from lumibot.backtesting import BacktestingBroker
from lumibot.data_sources import QMTBridgeData
from lumibot.entities import Asset
from lumibot.strategies import Strategy


class SimpleMovingAverageCrossover(Strategy):
    """
    Simple Moving Average Crossover Strategy for Chinese A-shares.

    Buy when short MA crosses above long MA.
    Sell when short MA crosses below long MA.
    """

    # Strategy parameters
    SYMBOL = "000001.SZ"  # Ping An Bank (平安银行)
    SHORT_MA_PERIOD = 5
    LONG_MA_PERIOD = 20
    POSITION_SIZE = 1000  # Shares (A-shares trade in lots of 100)

    def initialize(self):
        """Initialize strategy parameters."""
        self.sleeptime = "1D"  # Daily bars
        self.last_price = None

    def on_trading_iteration(self):
        """Main trading logic - runs on each iteration."""
        asset = Asset(self.SYMBOL, asset_type="stock")

        # Get historical prices for MA calculation
        bars = self.get_historical_prices(
            asset,
            length=self.LONG_MA_PERIOD + 5,  # Extra bars for calculation
            timestep="day"
        )

        if bars is None or len(bars.df) < self.LONG_MA_PERIOD:
            self.log_message("Not enough data to calculate MAs")
            return

        # Get current position
        position = self.get_position(asset)
        current_price = self.get_last_price(asset)

        # Calculate moving averages
        df = bars.df
        short_ma = df['close'].tail(self.SHORT_MA_PERIOD).mean()
        long_ma = df['close'].tail(self.LONG_MA_PERIOD).mean()

        self.log_message(
            f"Price: {current_price:.2f}, "
            f"Short MA({self.SHORT_MA_PERIOD}): {short_ma:.2f}, "
            f"Long MA({self.LONG_MA_PERIOD}): {long_ma:.2f}"
        )

        # Trading logic
        if short_ma > long_ma and position.quantity == 0:
            # Buy signal
            cash = self.cash
            if current_price and current_price > 0:
                qty = int(cash / current_price / 100) * 100  # Round to nearest 100
                if qty >= 100:
                    order = self.create_order("buy", asset, qty)
                    self.submit_order(order)
                    self.log_message(f"BUY signal: Buying {qty} shares at {current_price:.2f}")

        elif short_ma < long_ma and position.quantity > 0:
            # Sell signal
            qty = position.quantity
            order = self.create_order("sell", asset, qty)
            self.submit_order(order)
            self.log_message(f"SELL signal: Selling {qty} shares at {current_price:.2f}")

        self.last_price = current_price


def run_backtest():
    """Run the backtest with QMT Bridge data."""

    # Configure QMT Bridge data source
    # IMPORTANT: Replace with your actual QMT Bridge server details
    data_source = QMTBridgeData(
        host="192.168.1.100",  # Your QMT Bridge server IP
        port=8000,
        api_key="",  # Not required for historical data
        dividend_type="front",  # Forward adjustment (recommended for backtesting)
        fill_data=True,  # Fill missing data for suspended days
    )

    # Set backtest time range
    backtest_start = datetime.now() - timedelta(days=365)  # 1 year
    backtest_end = datetime.now()

    # Create backtesting broker
    broker = BacktestingBroker(
        data_source=data_source,
        datetime_start=backtest_start,
        datetime_end=backtest_end,
    )

    # Create strategy
    strategy = SimpleMovingAverageCrossover(broker=broker)

    # Run backtest
    print("=" * 60)
    print("QMT Bridge Backtest Example")
    print("=" * 60)
    print(f"Symbol: {strategy.SYMBOL}")
    print(f"Short MA: {strategy.SHORT_MA_PERIOD} days")
    print(f"Long MA: {strategy.LONG_MA_PERIOD} days")
    print(f"Backtest period: {backtest_start.strftime('%Y-%m-%d')} to {backtest_end.strftime('%Y-%m-%d')}")
    print("=" * 60)

    # Execute backtest
    result = strategy.run()

    # Print results
    print("\n" + "=" * 60)
    print("Backtest Results")
    print("=" * 60)
    if result:
        print(f"Total Return: {result.get('total_return', 'N/A')}")
        print(f"CAGR: {result.get('cagr', 'N/A')}")
        print(f"Max Drawdown: {result.get('max_drawdown', 'N/A')}")
        print(f"Sharpe Ratio: {result.get('sharpe', 'N/A')}")
        print(f"Total Trades: {result.get('total_trades', 'N/A')}")

    return result


class RSIStrategy(Strategy):
    """
    RSI Mean Reversion Strategy for Chinese A-shares.

    Buy when RSI < 30 (oversold).
    Sell when RSI > 70 (overbought).
    """

    SYMBOL = "600519.SH"  # Kweichow Moutai (贵州茅台)
    RSI_PERIOD = 14
    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 70

    def initialize(self):
        self.sleeptime = "1D"

    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator."""
        if len(prices) < period + 1:
            return None

        deltas = prices.diff()
        gains = deltas.where(deltas > 0, 0)
        losses = (-deltas).where(deltas < 0, 0)

        avg_gain = gains.rolling(period).mean()
        avg_loss = losses.rolling(period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi.iloc[-1]

    def on_trading_iteration(self):
        asset = Asset(self.SYMBOL, asset_type="stock")

        # Get historical prices
        bars = self.get_historical_prices(
            asset,
            length=self.RSI_PERIOD + 10,
            timestep="day"
        )

        if bars is None or len(bars.df) < self.RSI_PERIOD + 5:
            self.log_message("Not enough data for RSI calculation")
            return

        # Calculate RSI
        rsi = self.calculate_rsi(bars.df['close'], self.RSI_PERIOD)

        if rsi is None:
            return

        current_price = self.get_last_price(asset)
        position = self.get_position(asset)

        self.log_message(f"RSI: {rsi:.2f}, Price: {current_price:.2f}")

        # Trading logic
        if rsi < self.RSI_OVERSOLD and position.quantity == 0:
            # Oversold - Buy
            cash = self.cash
            if current_price and current_price > 0:
                qty = int(cash / current_price / 100) * 100
                if qty >= 100:
                    order = self.create_order("buy", asset, qty)
                    self.submit_order(order)
                    self.log_message(f"BUY (RSI oversold): {qty} shares")

        elif rsi > self.RSI_OVERBOUGHT and position.quantity > 0:
            # Overbought - Sell
            qty = position.quantity
            order = self.create_order("sell", asset, qty)
            self.submit_order(order)
            self.log_message(f"SELL (RSI overbought): {qty} shares")


def run_rsi_backtest():
    """Run RSI strategy backtest."""

    data_source = QMTBridgeData(
        host="192.168.1.100",
        port=8000,
        dividend_type="front",
    )

    broker = BacktestingBroker(
        data_source=data_source,
        datetime_start=datetime.now() - timedelta(days=180),
        datetime_end=datetime.now(),
    )

    strategy = RSIStrategy(broker=broker)
    return strategy.run()


if __name__ == "__main__":
    # Run the MA crossover backtest
    run_backtest()

    # Optionally run RSI strategy
    # run_rsi_backtest()
