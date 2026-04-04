"""
Unit tests for Portfolio Strategy system - Order Interception Approach.
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch
from collections import defaultdict

from lumibot.strategies.portfolio_strategy import (
    OrderInterceptor,
    AccumulatedOrder,
    NettedOrder,
    run_strategy_with_interception,
)


class TestAccumulatedOrder:
    """Tests for the AccumulatedOrder dataclass."""

    def test_create_buy_order(self):
        """Test creating a buy order."""
        order = AccumulatedOrder(
            strategy_id="strategy_a",
            symbol="000001.SZ",
            side="buy",
            quantity=100
        )
        assert order.strategy_id == "strategy_a"
        assert order.symbol == "000001.SZ"
        assert order.side == "buy"
        assert order.quantity == 100

    def test_create_sell_order(self):
        """Test creating a sell order."""
        order = AccumulatedOrder(
            strategy_id="strategy_a",
            symbol="000001.SZ",
            side="sell",
            quantity=50
        )
        assert order.side == "sell"
        assert order.quantity == 50


class TestNettedOrder:
    """Tests for the NettedOrder dataclass."""

    def test_should_execute_positive(self):
        """Test should_execute with positive quantity."""
        order = NettedOrder(symbol="000001.SZ", net_quantity=100)
        assert order.should_execute is True

    def test_should_execute_negative(self):
        """Test should_execute with negative quantity."""
        order = NettedOrder(symbol="000001.SZ", net_quantity=-100)
        assert order.should_execute is True

    def test_should_execute_zero(self):
        """Test should_execute with zero quantity."""
        order = NettedOrder(symbol="000001.SZ", net_quantity=0)
        assert order.should_execute is False

    def test_side_buy(self):
        """Test side property for buy."""
        order = NettedOrder(symbol="000001.SZ", net_quantity=100)
        assert order.side == "buy"

    def test_side_sell(self):
        """Test side property for sell."""
        order = NettedOrder(symbol="000001.SZ", net_quantity=-100)
        assert order.side == "sell"

    def test_abs_quantity(self):
        """Test absolute quantity property."""
        order = NettedOrder(symbol="000001.SZ", net_quantity=-150)
        assert order.abs_quantity == 150

    def test_component_orders_tracking(self):
        """Test that component orders are tracked."""
        components = [
            AccumulatedOrder(strategy_id="a", symbol="TEST", side="buy", quantity=100),
            AccumulatedOrder(strategy_id="b", symbol="TEST", side="sell", quantity=50),
        ]
        order = NettedOrder(symbol="TEST", net_quantity=50, component_orders=components)
        assert len(order.component_orders) == 2


class TestOrderInterceptor:
    """Tests for the OrderInterceptor class."""

    def test_interception_mode(self):
        """Test starting and stopping interception."""
        mock_strategy = MagicMock()
        mock_strategy.broker = MagicMock()

        interceptor = OrderInterceptor(mock_strategy)

        assert interceptor._is_intercepting is False

        interceptor.start_interception()
        assert interceptor._is_intercepting is True

        interceptor.stop_interception()
        assert interceptor._is_intercepting is False

    def test_get_netted_orders_buy_and_sell_cancel_out(self):
        """Test that buy and sell orders for same symbol cancel out."""
        mock_strategy = MagicMock()
        mock_strategy.broker = MagicMock()

        interceptor = OrderInterceptor(mock_strategy, min_net_quantity=1.0)
        interceptor.start_interception()

        # Add buy order
        buy_order = MagicMock()
        buy_order.asset.symbol = "000001.SZ"
        buy_order.side = "buy"
        buy_order.quantity = 100
        interceptor.intercept_order(buy_order, "strategy_a")

        # Add sell order
        sell_order = MagicMock()
        sell_order.asset.symbol = "000001.SZ"
        sell_order.side = "sell"
        sell_order.quantity = 100
        interceptor.intercept_order(sell_order, "strategy_b")

        # Get netted orders
        netted = interceptor.get_netted_orders()

        # Should be no orders since 100 - 100 = 0
        assert len(netted) == 0

    def test_get_netted_orders_partial_cancel(self):
        """Test partial cancellation of orders."""
        mock_strategy = MagicMock()
        mock_strategy.broker = MagicMock()

        interceptor = OrderInterceptor(mock_strategy, min_net_quantity=1.0)
        interceptor.start_interception()

        # Add buy order for 150
        buy_order = MagicMock()
        buy_order.asset.symbol = "000001.SZ"
        buy_order.side = "buy"
        buy_order.quantity = 150
        interceptor.intercept_order(buy_order, "strategy_a")

        # Add sell order for 100
        sell_order = MagicMock()
        sell_order.asset.symbol = "000001.SZ"
        sell_order.side = "sell"
        sell_order.quantity = 100
        interceptor.intercept_order(sell_order, "strategy_b")

        # Get netted orders
        netted = interceptor.get_netted_orders()

        # Should have 1 order for net 50
        assert len(netted) == 1
        assert netted[0].net_quantity == 50
        assert netted[0].side == "buy"

    def test_get_netted_orders_multiple_strategies_same_direction(self):
        """Test multiple strategies buying same symbol."""
        mock_strategy = MagicMock()
        mock_strategy.broker = MagicMock()

        interceptor = OrderInterceptor(mock_strategy)
        interceptor.start_interception()

        for i, qty in enumerate([100, 150, 50]):
            order = MagicMock()
            order.asset.symbol = "000001.SZ"
            order.side = "buy"
            order.quantity = qty
            interceptor.intercept_order(order, f"strategy_{i}")

        netted = interceptor.get_netted_orders()

        assert len(netted) == 1
        assert netted[0].net_quantity == 300
        assert len(netted[0].component_orders) == 3

    def test_get_netted_orders_different_symbols(self):
        """Test netting for different symbols."""
        mock_strategy = MagicMock()
        mock_strategy.broker = MagicMock()

        # Use low threshold so both symbols are included
        interceptor = OrderInterceptor(mock_strategy, min_net_quantity=1.0)
        interceptor.start_interception()

        # Symbol 1: buy 100, sell 50 = net 50
        order1 = MagicMock()
        order1.asset.symbol = "000001.SZ"
        order1.side = "buy"
        order1.quantity = 100
        interceptor.intercept_order(order1, "a")

        order2 = MagicMock()
        order2.asset.symbol = "000001.SZ"
        order2.side = "sell"
        order2.quantity = 50
        interceptor.intercept_order(order2, "b")

        # Symbol 2: sell 200, buy 100 = net -100
        order3 = MagicMock()
        order3.asset.symbol = "600519.SH"
        order3.side = "sell"
        order3.quantity = 200
        interceptor.intercept_order(order3, "c")

        order4 = MagicMock()
        order4.asset.symbol = "600519.SH"
        order4.side = "buy"
        order4.quantity = 100
        interceptor.intercept_order(order4, "d")

        netted = interceptor.get_netted_orders()

        # Should have 2 orders (one per symbol)
        assert len(netted) == 2

        sz_order = next(o for o in netted if o.symbol == "000001.SZ")
        sh_order = next(o for o in netted if o.symbol == "600519.SH")

        assert sz_order.net_quantity == 50  # 100 - 50
        assert sh_order.net_quantity == -100  # -200 + 100

    def test_min_net_quantity_threshold(self):
        """Test minimum quantity threshold."""
        mock_strategy = MagicMock()
        mock_strategy.broker = MagicMock()

        interceptor = OrderInterceptor(mock_strategy, min_net_quantity=100)
        interceptor.start_interception()

        # Net quantity is 50, below threshold
        order1 = MagicMock()
        order1.asset.symbol = "000001.SZ"
        order1.side = "buy"
        order1.quantity = 100
        interceptor.intercept_order(order1, "a")

        order2 = MagicMock()
        order2.asset.symbol = "000001.SZ"
        order2.side = "sell"
        order2.quantity = 50
        interceptor.intercept_order(order2, "b")

        netted = interceptor.get_netted_orders()

        # Should be filtered out (50 < 100)
        assert len(netted) == 0

    def test_get_accumulation_summary(self):
        """Test getting accumulation summary."""
        mock_strategy = MagicMock()
        mock_strategy.broker = MagicMock()

        interceptor = OrderInterceptor(mock_strategy)
        interceptor.start_interception()

        order1 = MagicMock()
        order1.asset.symbol = "000001.SZ"
        order1.side = "buy"
        order1.quantity = 100
        interceptor.intercept_order(order1, "strategy_a")

        order2 = MagicMock()
        order2.asset.symbol = "000001.SZ"
        order2.side = "sell"
        order2.quantity = 50
        interceptor.intercept_order(order2, "strategy_b")

        summary = interceptor.get_accumulation_summary()

        assert "000001.SZ" in summary
        assert summary["000001.SZ"]["order_count"] == 2
        assert summary["000001.SZ"]["net_quantity"] == 50
        assert "strategy_a" in summary["000001.SZ"]["strategies"]
        assert "strategy_b" in summary["000001.SZ"]["strategies"]


class TestRunStrategyWithInterception:
    """Tests for the run_strategy_with_interception function."""

    def test_function_exists(self):
        """Test that the function exists and is callable."""
        assert callable(run_strategy_with_interception)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
