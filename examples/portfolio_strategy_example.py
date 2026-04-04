"""
Portfolio Strategy Example for LumiBot

This example demonstrates how to use the portfolio strategy system
with order netting across multiple sub-strategies.

Scenario:
- SubStrategyA: Momentum strategy that buys on upward momentum
- SubStrategyB: Mean reversion strategy that sells on overbought conditions
- Both strategies may generate conflicting signals that get netted
"""

from datetime import datetime, timedelta

from lumibot.backtesting import PandasDataBacktesting
from lumibot.entities import Asset
from lumibot.strategies import Strategy
from lumibot.strategies.portfolio_strategy import PortfolioStrategy, SubStrategy


# ============== Sub-Strategies ==============

class MomentumStrategy(SubStrategy):
    """
    Simple momentum strategy.
    Buys when price is above 20-day SMA.
    """

    def initialize(self, sma_period: int = 20, position_size: int = 100):
        self.sma_period = sma_period
        self.position_size = position_size
        self.symbols = self.portfolio.main_strategy.symbols  # Access main strategy's symbols

    def on_trading_iteration(self):
        for symbol in self.symbols:
            try:
                prices = self.get_historical_prices(
                    symbol,
                    timestep="day",
                    length=self.sma_period + 5
                )

                if prices is None or len(prices.df) < self.sma_period:
                    continue

                df = prices.df
                close = df['close']
                sma = close.rolling(self.sma_period).mean()
                current_price = close.iloc[-1]
                current_sma = sma.iloc[-1]

                # Get current position
                current_pos = self.get_position(symbol)

                # Generate signal
                if current_price > current_sma and current_pos <= 0:
                    # Price above SMA and no position → Buy signal
                    self.buy(symbol, self.position_size)
                    print(f"[{self.strategy_id}] {symbol}: BUY signal (price={current_price:.2f} > SMA={current_sma:.2f})")

                elif current_price < current_sma and current_pos > 0:
                    # Price below SMA and have position → Sell signal
                    self.sell(symbol, self.position_size)
                    print(f"[{self.strategy_id}] {symbol}: SELL signal (price={current_price:.2f} < SMA={current_sma:.2f})")

            except Exception as e:
                print(f"[{self.strategy_id}] Error processing {symbol}: {e}")


class MeanReversionStrategy(SubStrategy):
    """
    Mean reversion strategy.
    Sells when RSI > 70 (overbought).
    Buys when RSI < 30 (oversold).
    """

    def initialize(self, rsi_period: int = 14, position_size: int = 100):
        self.rsi_period = rsi_period
        self.position_size = position_size
        self.symbols = self.portfolio.main_strategy.symbols

    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def on_trading_iteration(self):
        for symbol in self.symbols:
            try:
                prices = self.get_historical_prices(
                    symbol,
                    timestep="day",
                    length=self.rsi_period + 10
                )

                if prices is None or len(prices.df) < self.rsi_period + 5:
                    continue

                df = prices.df
                close = df['close']
                rsi = self.calculate_rsi(close, self.rsi_period)
                current_rsi = rsi.iloc[-1]

                # Get current position
                current_pos = self.get_position(symbol)

                # Generate signal
                if current_rsi > 70 and current_pos >= 0:
                    # Overbought → Sell signal
                    self.sell(symbol, self.position_size)
                    print(f"[{self.strategy_id}] {symbol}: SELL signal (RSI={current_rsi:.1f} > 70)")

                elif current_rsi < 30 and current_pos <= 0:
                    # Oversold → Buy signal
                    self.buy(symbol, self.position_size)
                    print(f"[{self.strategy_id}] {symbol}: BUY signal (RSI={current_rsi:.1f} < 30)")

            except Exception as e:
                print(f"[{self.strategy_id}] Error processing {symbol}: {e}")


# ============== Main Portfolio Strategy ==============

class MyPortfolioStrategy(Strategy):
    """
    Main strategy that uses the portfolio system.
    """

    def initialize(self, symbols: list):
        self.sleeptime = "1D"
        self.symbols = symbols

        # Create portfolio manager
        self.portfolio = PortfolioStrategy(self, min_net_quantity=1.0)

        # Add sub-strategies
        self.portfolio.add_sub_strategy(
            MomentumStrategy,
            "momentum",
            sma_period=20,
            position_size=100
        )

        self.portfolio.add_sub_strategy(
            MeanReversionStrategy,
            "mean_reversion",
            rsi_period=14,
            position_size=100
        )

    def on_trading_iteration(self):
        print(f"\n{'='*60}")
        print(f"Trading Iteration: {self.datetime}")
        print(f"{'='*60}")

        # Run all sub-strategies and collect signals
        signals = self.portfolio.run_iteration()

        print(f"\nGenerated {len(signals)} signals:")
        for sig in signals:
            print(f"  - {sig.strategy_id}: {sig.symbol} qty={sig.quantity}")

        # Get netted orders
        netted_orders = self.portfolio.get_netted_orders()

        print(f"\nNetted orders ({len(netted_orders)}):")
        for order in netted_orders:
            print(f"  - {order.symbol}: net_qty={order.net_quantity} "
                  f"(from {len(order.component_signals)} signals)")

        # Execute netted orders
        if netted_orders:
            executed = self.portfolio.execute_orders(netted_orders)
            print(f"\nExecuted {len(executed)} orders")

        # Print summary
        summary = self.portfolio.get_netting_summary()
        print(f"\nPortfolio positions:")
        for symbol, pos in summary["positions"].items():
            print(f"  - {symbol}: {pos['quantity']} shares")

    def on_fill(self, order, price, quantity):
        """Called when an order is filled."""
        # Update portfolio positions
        self.portfolio.update_positions_after_fill(order, price, quantity)
        print(f"Order filled: {order.asset.symbol} qty={quantity} @ {price}")


# ============== Backtest Configuration ==============

def prepare_backtest_data():
    """
    Prepare data for backtesting.
    In practice, you would load real price data here.
    """
    import pandas as pd
    import numpy as np

    symbols = ["000001.SZ", "600519.SH"]
    dates = pd.date_range(start="2023-01-01", end="2024-01-01", freq="B")

    pandas_data = {}

    for symbol in symbols:
        # Generate random price data
        np.random.seed(hash(symbol) % 2**32)
        returns = np.random.normal(0.0005, 0.02, len(dates))
        prices = 10 * (1 + returns).cumprod()

        df = pd.DataFrame({
            'open': prices * (1 + np.random.uniform(-0.01, 0.01, len(dates))),
            'high': prices * (1 + np.random.uniform(0, 0.02, len(dates))),
            'low': prices * (1 + np.random.uniform(-0.02, 0, len(dates))),
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)

        asset = Asset(symbol=symbol, asset_type=Asset.AssetType.STOCK)
        pandas_data[asset] = df

    return pandas_data


# ============== Run Backtest ==============

if __name__ == "__main__":
    # Prepare data
    pandas_data = prepare_backtest_data()

    # Run backtest
    result = MyPortfolioStrategy.backtest(
        datasource_class=PandasDataBacktesting,
        pandas_data=pandas_data,
        symbols=["000001.SZ", "600519.SH"],
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2024, 1, 1),
    )

    # Print results
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(result)
