"""
Portfolio Strategy System for LumiBot

This module implements a portfolio-level strategy management system similar to VNPy's
PortfolioStrategy, providing order netting, position aggregation, and multi-strategy
coordination.

Key Features:
- Wrap existing strategies without modification
- Intercept order creation and convert to signals
- Order netting (combines buy/sell for same symbol)
- Position aggregation at portfolio level
- Cross-strategy coordination

Example:
    >>> from my_strategies import StrategyA, StrategyB
    >>>
    >>> class MyPortfolio(PortfolioStrategy):
    ...     def initialize(self):
    ...         self.add_strategy(StrategyA, "strategy_a", **params_a)
    ...         self.add_strategy(StrategyB, "strategy_b", **params_b)
    >>>
    >>> # Orders are netted: StrategyA buys 100 + StrategyB sells 100 = 0 trades
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from lumibot.entities import Asset, Order, Position
    from lumibot.strategies.strategy import Strategy

logger = logging.getLogger(__name__)


@dataclass
class TradingSignal:
    """Represents a trading signal from a sub-strategy."""
    strategy_id: str
    symbol: str
    quantity: float  # Positive for buy, negative for sell
    price: Optional[float] = None  # Limit price, None for market
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    order_type: str = "market"  # "market" or "limit"

    @property
    def is_buy(self) -> bool:
        return self.quantity > 0

    @property
    def is_sell(self) -> bool:
        return self.quantity < 0

    @property
    def abs_quantity(self) -> float:
        return abs(self.quantity)


@dataclass
class NettedOrder:
    """Represents a netted order after aggregating signals."""
    symbol: str
    net_quantity: float
    component_signals: List[TradingSignal] = field(default_factory=list)
    price: Optional[float] = None

    @property
    def should_execute(self) -> bool:
        """Returns True if the netted order has non-zero quantity."""
        return self.net_quantity != 0

    @property
    def is_buy(self) -> bool:
        return self.net_quantity > 0

    @property
    def is_sell(self) -> bool:
        return self.net_quantity < 0

    @property
    def abs_quantity(self) -> float:
        return abs(self.net_quantity)


class OrderNettingEngine:
    """
    Engine for netting orders across multiple strategies.

    Aggregates trading signals and combines orders for the same symbol,
    reducing unnecessary trades when signals offset each other.
    """

    def __init__(self, min_net_quantity: float = 1.0):
        """
        Initialize the order netting engine.

        Args:
            min_net_quantity: Minimum absolute quantity to execute after netting.
                            Orders with |net_quantity| < min_net_quantity are ignored.
        """
        self.min_net_quantity = min_net_quantity
        self._signals: Dict[str, List[TradingSignal]] = defaultdict(list)

    def add_signal(self, signal: TradingSignal) -> None:
        """
        Add a trading signal to the netting engine.

        Args:
            signal: The trading signal to add
        """
        self._signals[signal.symbol].append(signal)
        logger.debug(f"[OrderNetting] Added signal: {signal.strategy_id} -> "
                    f"{signal.symbol} qty={signal.quantity}")

    def add_signals(self, signals: List[TradingSignal]) -> None:
        """
        Add multiple trading signals.

        Args:
            signals: List of trading signals
        """
        for signal in signals:
            self.add_signal(signal)

    def clear_signals(self) -> None:
        """Clear all pending signals."""
        self._signals.clear()

    def get_netted_orders(self) -> List[NettedOrder]:
        """
        Calculate netted orders from all signals.

        Returns:
            List of NettedOrder objects representing the aggregated orders
        """
        netted_orders = []

        for symbol, signals in self._signals.items():
            if not signals:
                continue

            # Sum all quantities for the same symbol
            net_quantity = sum(s.quantity for s in signals)

            # Get average limit price if specified
            prices = [s.price for s in signals if s.price is not None]
            avg_price = sum(prices) / len(prices) if prices else None

            # Create netted order
            netted = NettedOrder(
                symbol=symbol,
                net_quantity=net_quantity,
                component_signals=signals,
                price=avg_price
            )

            # Only include if meets minimum quantity threshold
            if abs(net_quantity) >= self.min_net_quantity:
                netted_orders.append(netted)
                logger.info(f"[OrderNetting] Netted order: {symbol} -> "
                           f"net_qty={net_quantity} (from {len(signals)} signals)")
            else:
                logger.info(f"[OrderNetting] Order netted to zero for {symbol}: "
                           f"net_qty={net_quantity} (below threshold {self.min_net_quantity})")

        return netted_orders

    def get_netting_summary(self) -> Dict[str, Dict]:
        """
        Get a summary of the netting process.

        Returns:
            Dictionary with netting details per symbol
        """
        summary = {}
        for symbol, signals in self._signals.items():
            net_qty = sum(s.quantity for s in signals)
            summary[symbol] = {
                "signal_count": len(signals),
                "net_quantity": net_qty,
                "strategies": list(set(s.strategy_id for s in signals)),
                "signals": [
                    {"strategy": s.strategy_id, "qty": s.quantity, "price": s.price}
                    for s in signals
                ]
            }
        return summary


class PortfolioPositionManager:
    """
    Manages positions at the portfolio level across all strategies.
    """

    def __init__(self):
        self._positions: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"quantity": 0.0, "avg_cost": 0.0, "strategy_positions": {}}
        )
        self._strategy_positions: Dict[str, Dict[str, float]] = defaultdict(dict)

    def update_position(self, strategy_id: str, symbol: str, quantity: float, price: float) -> None:
        """
        Update position after a trade execution.

        Args:
            strategy_id: ID of the strategy
            symbol: Trading symbol
            quantity: Quantity traded (positive for buy, negative for sell)
            price: Execution price
        """
        # Update strategy-level position
        old_qty = self._strategy_positions[strategy_id].get(symbol, 0.0)
        new_qty = old_qty + quantity
        self._strategy_positions[strategy_id][symbol] = new_qty

        # Update portfolio-level position
        old_portfolio_qty = self._positions[symbol]["quantity"]
        new_portfolio_qty = old_portfolio_qty + quantity
        self._positions[symbol]["quantity"] = new_portfolio_qty

        # Update average cost
        if quantity > 0:  # Buying
            if new_portfolio_qty > 0:
                old_cost = self._positions[symbol]["avg_cost"] * old_portfolio_qty
                new_cost = price * quantity
                self._positions[symbol]["avg_cost"] = (old_cost + new_cost) / new_portfolio_qty

        logger.debug(f"[PositionManager] Updated: {strategy_id} {symbol} "
                    f"qty={old_qty}→{new_qty} (portfolio: {old_portfolio_qty}→{new_portfolio_qty})")

    def get_portfolio_position(self, symbol: str) -> Dict[str, Any]:
        """
        Get portfolio-level position for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Position dictionary with quantity, avg_cost, and strategy breakdown
        """
        return dict(self._positions[symbol])

    def get_strategy_position(self, strategy_id: str, symbol: str) -> float:
        """
        Get a strategy's position in a symbol.

        Args:
            strategy_id: ID of the strategy
            symbol: Trading symbol

        Returns:
            Position quantity
        """
        return self._strategy_positions[strategy_id].get(symbol, 0.0)

    def get_all_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get all portfolio positions."""
        return {k: dict(v) for k, v in self._positions.items() if v["quantity"] != 0}

    def get_strategy_positions(self, strategy_id: str) -> Dict[str, float]:
        """Get all positions for a specific strategy."""
        return dict(self._strategy_positions[strategy_id])


class StrategyWrapper:
    """
    Wraps an existing Strategy class to intercept orders and convert to signals.

    This allows using existing strategies without modification while
    enabling order netting at the portfolio level.
    """

    def __init__(self,
                 strategy_class: Type['Strategy'],
                 strategy_id: str,
                 portfolio: 'PortfolioStrategy',
                 **strategy_params):
        """
        Initialize the strategy wrapper.

        Args:
            strategy_class: The Strategy class to wrap
            strategy_id: Unique identifier for this strategy
            portfolio: The parent portfolio strategy
            **strategy_params: Parameters to pass to the strategy's initialize
        """
        self.strategy_class = strategy_class
        self.strategy_id = strategy_id
        self.portfolio = portfolio
        self.strategy_params = strategy_params
        self._signals: List[TradingSignal] = []
        self._instance: Optional['Strategy'] = None
        self._original_submit_order = None
        self._original_create_order = None

    def create_instance(self, main_strategy: 'Strategy') -> 'Strategy':
        """
        Create a strategy instance with order interception.

        Args:
            main_strategy: The main portfolio strategy

        Returns:
            The wrapped strategy instance
        """
        # Create instance
        self._instance = self.strategy_class.__new__(self.strategy_class)

        # Store original methods
        self._original_submit_order = self._instance.submit_order
        self._original_create_order = self._instance.create_order

        # Monkey-patch order methods to intercept orders
        def patched_submit_order(order):
            """Convert order to signal instead of submitting."""
            symbol = order.asset.symbol if hasattr(order, 'asset') else order.symbol
            quantity = order.quantity if order.side == "buy" else -order.quantity
            price = getattr(order, 'limit_price', None)

            signal = TradingSignal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                quantity=quantity,
                price=price,
                order_type="limit" if price else "market"
            )
            self._signals.append(signal)
            logger.debug(f"[{self.strategy_id}] Intercepted order: {symbol} qty={quantity}")
            return order  # Return order but don't submit

        def patched_create_order(asset, quantity, side, limit_price=None, **kwargs):
            """Create order but don't submit - will be netted."""
            from lumibot.entities import Order
            order = Order(
                strategy=self._instance,
                asset=asset,
                quantity=quantity,
                side=side,
                limit_price=limit_price,
                **kwargs
            )
            return order

        # Apply patches
        self._instance.submit_order = patched_submit_order
        self._instance.create_order = patched_create_order

        # Share data from main strategy
        self._instance._broker = main_strategy._broker
        self._instance._data_source = main_strategy._data_source
        self._instance.datetime = main_strategy.datetime

        # Initialize with params
        if hasattr(self._instance, 'initialize'):
            self._instance.initialize(**self.strategy_params)

        return self._instance

    def get_signals(self) -> List[TradingSignal]:
        """Get all signals generated by this strategy."""
        return self._signals

    def clear_signals(self) -> None:
        """Clear all signals."""
        self._signals.clear()

    def run_iteration(self) -> None:
        """Run one trading iteration of the wrapped strategy."""
        if self._instance and hasattr(self._instance, 'on_trading_iteration'):
            try:
                # Update datetime from main strategy
                self._instance.datetime = self.portfolio.main_strategy.datetime
                self._instance.on_trading_iteration()
            except Exception as e:
                logger.error(f"[{self.strategy_id}] Error in on_trading_iteration: {e}")


class PortfolioStrategy:
    """
    Portfolio Strategy that manages multiple strategies with order netting.

    This class coordinates multiple strategies, intercepts their orders,
    aggregates them as signals, and executes netted orders.

    Example:
        >>> portfolio = PortfolioStrategy(main_strategy=my_strategy)
        >>> portfolio.add_strategy(StrategyA, "strategy_a", param1=10)
        >>> portfolio.add_strategy(StrategyB, "strategy_b", param2=20)
        >>>
        >>> # In main strategy's on_trading_iteration:
        >>> portfolio.run_iteration()  # Runs all strategies, collects signals
        >>> orders = portfolio.get_netted_orders()  # Returns netted orders
        >>> portfolio.execute_orders(orders)  # Executes through main strategy
    """

    def __init__(self, main_strategy: 'Strategy', min_net_quantity: float = 1.0):
        """
        Initialize the portfolio strategy.

        Args:
            main_strategy: The main LumiBot strategy instance
            min_net_quantity: Minimum quantity threshold for netted orders
        """
        self.main_strategy = main_strategy
        self.netting_engine = OrderNettingEngine(min_net_quantity)
        self.position_manager = PortfolioPositionManager()
        self._strategies: Dict[str, StrategyWrapper] = {}
        self._iteration_count = 0

    def add_strategy(self,
                     strategy_class: Type['Strategy'],
                     strategy_id: str,
                     **kwargs) -> StrategyWrapper:
        """
        Add a strategy to the portfolio.

        Args:
            strategy_class: Strategy class (must inherit from Strategy)
            strategy_id: Unique identifier for this strategy
            **kwargs: Parameters to pass to the strategy's initialize method

        Returns:
            The created strategy wrapper
        """
        if strategy_id in self._strategies:
            raise ValueError(f"Strategy with ID '{strategy_id}' already exists")

        wrapper = StrategyWrapper(
            strategy_class=strategy_class,
            strategy_id=strategy_id,
            portfolio=self,
            **kwargs
        )

        # Create instance with order interception
        wrapper.create_instance(self.main_strategy)

        self._strategies[strategy_id] = wrapper
        logger.info(f"[PortfolioStrategy] Added strategy: {strategy_id}")
        return wrapper

    def remove_strategy(self, strategy_id: str) -> Optional[StrategyWrapper]:
        """
        Remove a strategy from the portfolio.

        Args:
            strategy_id: ID of the strategy to remove

        Returns:
            The removed strategy wrapper, or None if not found
        """
        return self._strategies.pop(strategy_id, None)

    def get_strategy(self, strategy_id: str) -> Optional[StrategyWrapper]:
        """Get a strategy wrapper by ID."""
        return self._strategies.get(strategy_id)

    def get_all_strategies(self) -> Dict[str, StrategyWrapper]:
        """Get all strategy wrappers."""
        return dict(self._strategies)

    def run_iteration(self) -> List[TradingSignal]:
        """
        Run one trading iteration across all strategies.

        This method:
        1. Clears previous signals
        2. Calls on_trading_iteration() for each strategy
        3. Collects all intercepted signals
        4. Adds them to the netting engine

        Returns:
            List of all signals generated
        """
        self._iteration_count += 1
        all_signals = []

        # Clear previous state
        self.netting_engine.clear_signals()
        for wrapper in self._strategies.values():
            wrapper.clear_signals()

        # Run each strategy
        for strategy_id, wrapper in self._strategies.items():
            try:
                logger.debug(f"[PortfolioStrategy] Running strategy: {strategy_id}")
                wrapper.run_iteration()

                # Collect signals
                signals = wrapper.get_signals()
                all_signals.extend(signals)
                self.netting_engine.add_signals(signals)

            except Exception as e:
                logger.error(f"[PortfolioStrategy] Error in {strategy_id}: {e}")

        logger.info(f"[PortfolioStrategy] Iteration {self._iteration_count}: "
                   f"{len(all_signals)} signals from {len(self._strategies)} strategies")
        return all_signals

    def get_netted_orders(self) -> List[NettedOrder]:
        """
        Get netted orders after aggregating all signals.

        Returns:
            List of netted orders ready for execution
        """
        return self.netting_engine.get_netted_orders()

    def execute_orders(self, orders: Optional[List[NettedOrder]] = None) -> List['Order']:
        """
        Execute netted orders through the main strategy.

        Args:
            orders: List of netted orders to execute. If None, uses current netted orders.

        Returns:
            List of Order objects created
        """
        if orders is None:
            orders = self.get_netted_orders()

        executed_orders = []

        for netted in orders:
            if not netted.should_execute:
                continue

            try:
                # Create order through main strategy
                from lumibot.entities import Asset, Order

                asset = Asset(symbol=netted.symbol, asset_type=Asset.AssetType.STOCK)

                # Use original (unpatched) methods
                side = "buy" if netted.is_buy else "sell"
                quantity = int(netted.abs_quantity)

                order = self.main_strategy.create_order(
                    asset=asset,
                    quantity=quantity,
                    side=side,
                    limit_price=netted.price
                )

                # Submit through main strategy
                self.main_strategy.submit_order(order)
                executed_orders.append(order)

                logger.info(f"[PortfolioStrategy] Executed: {netted.symbol} "
                           f"qty={netted.net_quantity}")

            except Exception as e:
                logger.error(f"[PortfolioStrategy] Failed to execute order for "
                           f"{netted.symbol}: {e}")

        return executed_orders

    def update_positions_after_fill(self, order: 'Order', fill_price: float,
                                    fill_quantity: float) -> None:
        """
        Update positions after an order is filled.

        This method should be called from the main strategy's on_fill method.

        Args:
            order: The filled order
            fill_price: Execution price
            fill_quantity: Filled quantity (always positive)
        """
        symbol = order.asset.symbol
        quantity = fill_quantity if order.side == "buy" else -fill_quantity

        # Distribute position update proportionally to contributing strategies
        netted = self.netting_engine.get_netted_orders()
        for netted_order in netted:
            if netted_order.symbol == symbol:
                total_signal_qty = sum(s.quantity for s in netted_order.component_signals)
                if total_signal_qty != 0:
                    for signal in netted_order.component_signals:
                        # Proportional allocation
                        ratio = signal.quantity / total_signal_qty
                        allocated_qty = quantity * ratio
                        self.position_manager.update_position(
                            signal.strategy_id, symbol, allocated_qty, fill_price
                        )
                break

    def get_netting_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current netting state.

        Returns:
            Dictionary with netting details and statistics
        """
        return {
            "iteration": self._iteration_count,
            "strategies": list(self._strategies.keys()),
            "netting_details": self.netting_engine.get_netting_summary(),
            "positions": self.position_manager.get_all_positions()
        }


# Convenience exports
__all__ = [
    'PortfolioStrategy',
    'StrategyWrapper',
    'OrderNettingEngine',
    'PortfolioPositionManager',
    'TradingSignal',
    'NettedOrder',
]
