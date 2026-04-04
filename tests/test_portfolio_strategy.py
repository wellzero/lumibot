"""
Unit tests for Portfolio Strategy system.
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from lumibot.strategies.portfolio_strategy import (
    PortfolioStrategy,
    SubStrategy,
    OrderNettingEngine,
    PortfolioPositionManager,
    TradingSignal,
    NettedOrder,
    SignalType,
)


class TestOrderNettingEngine:
    """Tests for the OrderNettingEngine class."""

    def test_add_single_buy_signal(self):
        """Test adding a single buy signal."""
        engine = OrderNettingEngine()
        signal = TradingSignal(
            strategy_id="strategy_a",
            symbol="000001.SZ",
            quantity=100
        )
        engine.add_signal(signal)
        orders = engine.get_netted_orders()

        assert len(orders) == 1
        assert orders[0].symbol == "000001.SZ"
        assert orders[0].net_quantity == 100
        assert orders[0].is_buy

    def test_add_single_sell_signal(self):
        """Test adding a single sell signal."""
        engine = OrderNettingEngine()
        signal = TradingSignal(
            strategy_id="strategy_a",
            symbol="000001.SZ",
            quantity=-100
        )
        engine.add_signal(signal)
        orders = engine.get_netted_orders()

        assert len(orders) == 1
        assert orders[0].symbol == "000001.SZ"
        assert orders[0].net_quantity == -100
        assert orders[0].is_sell

    def test_netting_buy_and_sell_cancel_out(self):
        """Test that buy and sell signals for same symbol cancel out."""
        engine = OrderNettingEngine(min_net_quantity=1.0)

        # Strategy A wants to buy 100
        engine.add_signal(TradingSignal(
            strategy_id="strategy_a",
            symbol="000001.SZ",
            quantity=100
        ))

        # Strategy B wants to sell 100
        engine.add_signal(TradingSignal(
            strategy_id="strategy_b",
            symbol="000001.SZ",
            quantity=-100
        ))

        orders = engine.get_netted_orders()

        # Should be no orders since 100 - 100 = 0
        assert len(orders) == 0

    def test_netting_partial_cancel(self):
        """Test partial cancellation of signals."""
        engine = OrderNettingEngine(min_net_quantity=1.0)

        # Strategy A wants to buy 150
        engine.add_signal(TradingSignal(
            strategy_id="strategy_a",
            symbol="000001.SZ",
            quantity=150
        ))

        # Strategy B wants to sell 100
        engine.add_signal(TradingSignal(
            strategy_id="strategy_b",
            symbol="000001.SZ",
            quantity=-100
        ))

        orders = engine.get_netted_orders()

        # Should have 1 order for net 50
        assert len(orders) == 1
        assert orders[0].net_quantity == 50
        assert orders[0].is_buy

    def test_netting_multiple_strategies_same_direction(self):
        """Test multiple strategies buying same symbol."""
        engine = OrderNettingEngine()

        engine.add_signal(TradingSignal(strategy_id="a", symbol="000001.SZ", quantity=100))
        engine.add_signal(TradingSignal(strategy_id="b", symbol="000001.SZ", quantity=150))
        engine.add_signal(TradingSignal(strategy_id="c", symbol="000001.SZ", quantity=50))

        orders = engine.get_netted_orders()

        assert len(orders) == 1
        assert orders[0].net_quantity == 300
        assert len(orders[0].component_signals) == 3

    def test_netting_different_symbols(self):
        """Test netting for multiple symbols."""
        engine = OrderNettingEngine()

        engine.add_signal(TradingSignal(strategy_id="a", symbol="000001.SZ", quantity=100))
        engine.add_signal(TradingSignal(strategy_id="b", symbol="000001.SZ", quantity=-50))
        engine.add_signal(TradingSignal(strategy_id="a", symbol="600519.SH", quantity=-200))
        engine.add_signal(TradingSignal(strategy_id="c", symbol="600519.SH", quantity=100))

        orders = engine.get_netted_orders()

        # Should have 2 orders (one per symbol)
        assert len(orders) == 2

        sz_order = next(o for o in orders if o.symbol == "000001.SZ")
        sh_order = next(o for o in orders if o.symbol == "600519.SH")

        assert sz_order.net_quantity == 50  # 100 - 50
        assert sh_order.net_quantity == -100  # -200 + 100

    def test_min_net_quantity_threshold(self):
        """Test minimum quantity threshold."""
        engine = OrderNettingEngine(min_net_quantity=100)

        # Net quantity is 50, below threshold
        engine.add_signal(TradingSignal(strategy_id="a", symbol="000001.SZ", quantity=100))
        engine.add_signal(TradingSignal(strategy_id="b", symbol="000001.SZ", quantity=-50))

        orders = engine.get_netted_orders()

        # Should be filtered out
        assert len(orders) == 0

    def test_clear_signals(self):
        """Test clearing signals."""
        engine = OrderNettingEngine()
        engine.add_signal(TradingSignal(strategy_id="a", symbol="000001.SZ", quantity=100))

        engine.clear_signals()
        orders = engine.get_netted_orders()

        assert len(orders) == 0

    def test_get_netting_summary(self):
        """Test getting netting summary."""
        engine = OrderNettingEngine()

        engine.add_signal(TradingSignal(strategy_id="a", symbol="000001.SZ", quantity=100))
        engine.add_signal(TradingSignal(strategy_id="b", symbol="000001.SZ", quantity=-50))

        summary = engine.get_netting_summary()

        assert "000001.SZ" in summary
        assert summary["000001.SZ"]["signal_count"] == 2
        assert summary["000001.SZ"]["net_quantity"] == 50
        assert "a" in summary["000001.SZ"]["strategies"]
        assert "b" in summary["000001.SZ"]["strategies"]


class TestPortfolioPositionManager:
    """Tests for the PortfolioPositionManager class."""

    def test_update_position_buy(self):
        """Test updating position after a buy."""
        manager = PortfolioPositionManager()

        manager.update_position("strategy_a", "000001.SZ", 100, 10.0)

        pos = manager.get_portfolio_position("000001.SZ")
        assert pos["quantity"] == 100
        assert pos["avg_cost"] == 10.0

    def test_update_position_multiple_buys(self):
        """Test average cost calculation with multiple buys."""
        manager = PortfolioPositionManager()

        manager.update_position("strategy_a", "000001.SZ", 100, 10.0)
        manager.update_position("strategy_a", "000001.SZ", 100, 12.0)

        pos = manager.get_portfolio_position("000001.SZ")
        assert pos["quantity"] == 200
        assert pos["avg_cost"] == 11.0  # (100*10 + 100*12) / 200

    def test_update_position_sell(self):
        """Test updating position after a sell."""
        manager = PortfolioPositionManager()

        manager.update_position("strategy_a", "000001.SZ", 100, 10.0)
        manager.update_position("strategy_a", "000001.SZ", -50, 12.0)

        pos = manager.get_portfolio_position("000001.SZ")
        assert pos["quantity"] == 50

    def test_sub_strategy_positions(self):
        """Test tracking positions per sub-strategy."""
        manager = PortfolioPositionManager()

        manager.update_position("strategy_a", "000001.SZ", 100, 10.0)
        manager.update_position("strategy_b", "000001.SZ", 50, 10.0)

        pos_a = manager.get_sub_strategy_position("strategy_a", "000001.SZ")
        pos_b = manager.get_sub_strategy_position("strategy_b", "000001.SZ")
        total = manager.get_portfolio_position("000001.SZ")

        assert pos_a == 100
        assert pos_b == 50
        assert total["quantity"] == 150

    def test_get_all_positions(self):
        """Test getting all positions."""
        manager = PortfolioPositionManager()

        manager.update_position("a", "000001.SZ", 100, 10.0)
        manager.update_position("b", "600519.SH", 200, 20.0)

        all_pos = manager.get_all_positions()

        assert len(all_pos) == 2
        assert "000001.SZ" in all_pos
        assert "600519.SH" in all_pos


class TestSubStrategy:
    """Tests for the SubStrategy class."""

    def test_signal_generation(self):
        """Test generating signals from sub-strategy."""
        mock_portfolio = MagicMock()

        class ConcreteSubStrategy(SubStrategy):
            def on_trading_iteration(self):
                pass

        sub = ConcreteSubStrategy(mock_portfolio, "test_strategy")

        sub.signal("000001.SZ", 100, price=10.0)

        signals = sub.get_signals()
        assert len(signals) == 1
        assert signals[0].symbol == "000001.SZ"
        assert signals[0].quantity == 100
        assert signals[0].price == 10.0

    def test_buy_sell_helpers(self):
        """Test buy and sell helper methods."""
        mock_portfolio = MagicMock()

        class ConcreteSubStrategy(SubStrategy):
            def on_trading_iteration(self):
                pass

        sub = ConcreteSubStrategy(mock_portfolio, "test_strategy")

        sub.buy("000001.SZ", 100)
        sub.sell("600519.SH", 50)

        signals = sub.get_signals()
        assert len(signals) == 2
        assert signals[0].quantity == 100
        assert signals[1].quantity == -50

    def test_clear_signals(self):
        """Test clearing signals."""
        mock_portfolio = MagicMock()

        class ConcreteSubStrategy(SubStrategy):
            def on_trading_iteration(self):
                pass

        sub = ConcreteSubStrategy(mock_portfolio, "test_strategy")

        sub.signal("000001.SZ", 100)
        sub.clear_signals()

        assert len(sub.get_signals()) == 0


class TestPortfolioStrategy:
    """Tests for the PortfolioStrategy class."""

    def test_add_sub_strategy(self):
        """Test adding sub-strategies."""
        mock_main = MagicMock()
        portfolio = PortfolioStrategy(mock_main)

        class TestSub(SubStrategy):
            def on_trading_iteration(self):
                pass

        sub = portfolio.add_sub_strategy(TestSub, "test_sub")

        assert "test_sub" in portfolio.get_all_sub_strategies()
        assert portfolio.get_sub_strategy("test_sub") == sub

    def test_remove_sub_strategy(self):
        """Test removing sub-strategies."""
        mock_main = MagicMock()
        portfolio = PortfolioStrategy(mock_main)

        class TestSub(SubStrategy):
            def on_trading_iteration(self):
                pass

        portfolio.add_sub_strategy(TestSub, "test_sub")
        removed = portfolio.remove_sub_strategy("test_sub")

        assert removed is not None
        assert "test_sub" not in portfolio.get_all_sub_strategies()

    def test_duplicate_strategy_id(self):
        """Test that duplicate IDs raise an error."""
        mock_main = MagicMock()
        portfolio = PortfolioStrategy(mock_main)

        class TestSub(SubStrategy):
            def on_trading_iteration(self):
                pass

        portfolio.add_sub_strategy(TestSub, "test_sub")

        with pytest.raises(ValueError):
            portfolio.add_sub_strategy(TestSub, "test_sub")

    def test_run_iteration_collects_signals(self):
        """Test that run_iteration collects signals from all sub-strategies."""
        mock_main = MagicMock()
        portfolio = PortfolioStrategy(mock_main)

        class BuyStrategy(SubStrategy):
            def on_trading_iteration(self):
                self.buy("000001.SZ", 100)

        class SellStrategy(SubStrategy):
            def on_trading_iteration(self):
                self.sell("000001.SZ", 50)

        portfolio.add_sub_strategy(BuyStrategy, "buyer")
        portfolio.add_sub_strategy(SellStrategy, "seller")

        signals = portfolio.run_iteration()

        assert len(signals) == 2
        assert any(s.quantity == 100 for s in signals)
        assert any(s.quantity == -50 for s in signals)

    def test_netted_orders_from_signals(self):
        """Test getting netted orders from collected signals."""
        mock_main = MagicMock()
        portfolio = PortfolioStrategy(mock_main)

        class BuyStrategy(SubStrategy):
            def on_trading_iteration(self):
                self.buy("000001.SZ", 100)

        class SellStrategy(SubStrategy):
            def on_trading_iteration(self):
                self.sell("000001.SZ", 100)

        portfolio.add_sub_strategy(BuyStrategy, "buyer")
        portfolio.add_sub_strategy(SellStrategy, "seller")

        portfolio.run_iteration()
        orders = portfolio.get_netted_orders()

        # Net should be 0, so no orders
        assert len(orders) == 0


class TestTradingSignal:
    """Tests for the TradingSignal dataclass."""

    def test_is_buy_is_sell_properties(self):
        """Test is_buy and is_sell properties."""
        buy_signal = TradingSignal(strategy_id="a", symbol="TEST", quantity=100)
        sell_signal = TradingSignal(strategy_id="a", symbol="TEST", quantity=-100)
        hold_signal = TradingSignal(strategy_id="a", symbol="TEST", quantity=0)

        assert buy_signal.is_buy
        assert not buy_signal.is_sell

        assert sell_signal.is_sell
        assert not sell_signal.is_buy

        assert not hold_signal.is_buy
        assert not hold_signal.is_sell

    def test_abs_quantity(self):
        """Test absolute quantity property."""
        signal = TradingSignal(strategy_id="a", symbol="TEST", quantity=-150)
        assert signal.abs_quantity == 150


class TestNettedOrder:
    """Tests for the NettedOrder dataclass."""

    def test_should_execute(self):
        """Test should_execute property."""
        order_with_qty = NettedOrder(symbol="TEST", net_quantity=100)
        order_zero = NettedOrder(symbol="TEST", net_quantity=0)

        assert order_with_qty.should_execute
        assert not order_zero.should_execute

    def test_component_signals_tracking(self):
        """Test that component signals are tracked."""
        signals = [
            TradingSignal(strategy_id="a", symbol="TEST", quantity=100),
            TradingSignal(strategy_id="b", symbol="TEST", quantity=50),
        ]
        order = NettedOrder(symbol="TEST", net_quantity=150, component_signals=signals)

        assert len(order.component_signals) == 2
        assert order.net_quantity == 150


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
