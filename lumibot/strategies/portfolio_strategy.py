"""
Portfolio Strategy System for LumiBot

This module implements a portfolio-level strategy management system similar to VNPy's
PortfolioStrategy, providing order netting, position aggregation, and multi-strategy
coordination.

Key Features:
- Multiple sub-strategies under one portfolio
- Order netting (combines buy/sell for same symbol)
- Position aggregation at portfolio level
- Cross-strategy event communication
- Unified execution through single broker connection

Example:
    >>> class SubStrategyA(SubStrategy):
    ...     def on_trading_iteration(self):
    ...         self.signal("000001.SZ", 100)  # Buy 100 shares
    >>>
    >>> class SubStrategyB(SubStrategy):
    ...     def on_trading_iteration(self):
    ...         self.signal("000001.SZ", -100)  # Sell 100 shares
    >>>
    >>> class MyPortfolio(PortfolioStrategy):
    ...     def initialize(self):
    ...         self.add_sub_strategy(SubStrategyA, "strategy_a")
    ...         self.add_sub_strategy(SubStrategyB, "strategy_b")
    >>>     # Orders are netted: 100 + (-100) = 0 → No trade executed
"""

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from lumibot.entities import Asset, Order, Position
    from lumibot.strategies.strategy import Strategy

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Signal type for sub-strategy orders."""
    BUY = 1
    SELL = -1
    HOLD = 0


@dataclass
class TradingSignal:
    """Represents a trading signal from a sub-strategy."""
    strategy_id: str
    symbol: str
    quantity: float  # Positive for buy, negative for sell
    price: Optional[float] = None  # Limit price, None for market
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

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
    Engine for netting orders across multiple sub-strategies.

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
                "strategies": [s.strategy_id for s in signals],
                "signals": [
                    {"strategy": s.strategy_id, "qty": s.quantity, "price": s.price}
                    for s in signals
                ]
            }
        return summary


class PortfolioPositionManager:
    """
    Manages positions at the portfolio level across all sub-strategies.
    """

    def __init__(self):
        self._positions: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"quantity": 0.0, "avg_cost": 0.0, "sub_strategy_positions": {}}
        )
        self._sub_strategy_positions: Dict[str, Dict[str, float]] = defaultdict(dict)

    def update_position(self, strategy_id: str, symbol: str, quantity: float, price: float) -> None:
        """
        Update position after a trade execution.

        Args:
            strategy_id: ID of the sub-strategy
            symbol: Trading symbol
            quantity: Quantity traded (positive for buy, negative for sell)
            price: Execution price
        """
        # Update sub-strategy position
        old_qty = self._sub_strategy_positions[strategy_id].get(symbol, 0.0)
        new_qty = old_qty + quantity
        self._sub_strategy_positions[strategy_id][symbol] = new_qty

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
            Position dictionary with quantity, avg_cost, and sub-strategy breakdown
        """
        return dict(self._positions[symbol])

    def get_sub_strategy_position(self, strategy_id: str, symbol: str) -> float:
        """
        Get a sub-strategy's position in a symbol.

        Args:
            strategy_id: ID of the sub-strategy
            symbol: Trading symbol

        Returns:
            Position quantity
        """
        return self._sub_strategy_positions[strategy_id].get(symbol, 0.0)

    def get_all_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get all portfolio positions."""
        return {k: dict(v) for k, v in self._positions.items() if v["quantity"] != 0}

    def get_sub_strategy_positions(self, strategy_id: str) -> Dict[str, float]:
        """Get all positions for a specific sub-strategy."""
        return dict(self._sub_strategy_positions[strategy_id])


class SubStrategy(ABC):
    """
    Base class for sub-strategies within a portfolio.

    Sub-strategies generate signals that are aggregated by the portfolio
    and netted before execution.
    """

    def __init__(self, portfolio: 'PortfolioStrategy', strategy_id: str):
        """
        Initialize the sub-strategy.

        Args:
            portfolio: The parent portfolio strategy
            strategy_id: Unique identifier for this sub-strategy
        """
        self.portfolio = portfolio
        self.strategy_id = strategy_id
        self._signals: List[TradingSignal] = []

    def initialize(self, **kwargs) -> None:
        """
        Initialize the sub-strategy. Override this method to set up parameters.

        This method is called once when the sub-strategy is added to the portfolio.
        """
        pass

    @abstractmethod
    def on_trading_iteration(self) -> None:
        """
        Main trading logic. Override this method to implement your strategy.

        Use signal() or signal_asset() to generate trading signals.
        """
        pass

    def signal(self, symbol: str, quantity: float, price: Optional[float] = None,
               **metadata) -> None:
        """
        Generate a trading signal.

        Args:
            symbol: Trading symbol
            quantity: Quantity (positive for buy, negative for sell)
            price: Limit price (None for market order)
            **metadata: Additional metadata for the signal
        """
        signal = TradingSignal(
            strategy_id=self.strategy_id,
            symbol=symbol,
            quantity=quantity,
            price=price,
            metadata=metadata
        )
        self._signals.append(signal)

    def signal_asset(self, asset: 'Asset', quantity: float,
                     price: Optional[float] = None, **metadata) -> None:
        """
        Generate a trading signal for an Asset object.

        Args:
            asset: The asset to trade
            quantity: Quantity (positive for buy, negative for sell)
            price: Limit price (None for market order)
            **metadata: Additional metadata for the signal
        """
        self.signal(symbol=asset.symbol, quantity=quantity, price=price, **metadata)

    def buy(self, symbol: str, quantity: float, price: Optional[float] = None,
            **metadata) -> None:
        """Generate a buy signal."""
        self.signal(symbol, abs(quantity), price, **metadata)

    def sell(self, symbol: str, quantity: float, price: Optional[float] = None,
             **metadata) -> None:
        """Generate a sell signal."""
        self.signal(symbol, -abs(quantity), price, **metadata)

    def get_signals(self) -> List[TradingSignal]:
        """Get all signals generated in this iteration."""
        return self._signals

    def clear_signals(self) -> None:
        """Clear all signals."""
        self._signals.clear()

    def get_position(self, symbol: str) -> float:
        """
        Get this sub-strategy's position in a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Position quantity
        """
        return self.portfolio.position_manager.get_sub_strategy_position(
            self.strategy_id, symbol
        )

    def get_portfolio_position(self, symbol: str) -> float:
        """
        Get the portfolio-level position for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Total position quantity across all sub-strategies
        """
        pos = self.portfolio.position_manager.get_portfolio_position(symbol)
        return pos.get("quantity", 0.0)

    # Delegate data access to portfolio
    def get_last_price(self, symbol: str) -> Optional[float]:
        """Get last price for a symbol."""
        return self.portfolio.get_last_price(symbol)

    def get_historical_prices(self, symbol: str, **kwargs):
        """Get historical prices for a symbol."""
        return self.portfolio.get_historical_prices(symbol, **kwargs)


class PortfolioStrategy:
    """
    Portfolio Strategy that manages multiple sub-strategies with order netting.

    This class coordinates multiple sub-strategies, aggregates their signals,
    and executes netted orders through the main strategy.

    Example:
        >>> portfolio = PortfolioStrategy(main_strategy=my_strategy)
        >>> portfolio.add_sub_strategy(MySubStrategyA, "strategy_a", param1=10)
        >>> portfolio.add_sub_strategy(MySubStrategyB, "strategy_b", param2=20)
        >>>
        >>> # In main strategy's on_trading_iteration:
        >>> portfolio.run_iteration()
        >>> orders = portfolio.get_netted_orders()
        >>> portfolio.execute_orders(orders)
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
        self._sub_strategies: Dict[str, SubStrategy] = {}
        self._iteration_count = 0

    def add_sub_strategy(self, strategy_class: type, strategy_id: str,
                         **kwargs) -> SubStrategy:
        """
        Add a sub-strategy to the portfolio.

        Args:
            strategy_class: Sub-strategy class (must inherit from SubStrategy)
            strategy_id: Unique identifier for this sub-strategy
            **kwargs: Parameters to pass to the sub-strategy's initialize method

        Returns:
            The created sub-strategy instance
        """
        if strategy_id in self._sub_strategies:
            raise ValueError(f"Sub-strategy with ID '{strategy_id}' already exists")

        sub_strategy = strategy_class(self, strategy_id)
        sub_strategy.initialize(**kwargs)
        self._sub_strategies[strategy_id] = sub_strategy

        logger.info(f"[PortfolioStrategy] Added sub-strategy: {strategy_id}")
        return sub_strategy

    def remove_sub_strategy(self, strategy_id: str) -> Optional[SubStrategy]:
        """
        Remove a sub-strategy from the portfolio.

        Args:
            strategy_id: ID of the sub-strategy to remove

        Returns:
            The removed sub-strategy, or None if not found
        """
        return self._sub_strategies.pop(strategy_id, None)

    def get_sub_strategy(self, strategy_id: str) -> Optional[SubStrategy]:
        """Get a sub-strategy by ID."""
        return self._sub_strategies.get(strategy_id)

    def get_all_sub_strategies(self) -> Dict[str, SubStrategy]:
        """Get all sub-strategies."""
        return dict(self._sub_strategies)

    def run_iteration(self) -> List[TradingSignal]:
        """
        Run one trading iteration across all sub-strategies.

        This method:
        1. Clears previous signals
        2. Calls on_trading_iteration() for each sub-strategy
        3. Collects all signals
        4. Adds them to the netting engine

        Returns:
            List of all signals generated
        """
        self._iteration_count += 1
        all_signals = []

        # Clear previous state
        self.netting_engine.clear_signals()
        for sub in self._sub_strategies.values():
            sub.clear_signals()

        # Run each sub-strategy
        for strategy_id, sub in self._sub_strategies.items():
            try:
                logger.debug(f"[PortfolioStrategy] Running sub-strategy: {strategy_id}")
                sub.on_trading_iteration()

                # Collect signals
                signals = sub.get_signals()
                all_signals.extend(signals)
                self.netting_engine.add_signals(signals)

            except Exception as e:
                logger.error(f"[PortfolioStrategy] Error in {strategy_id}: {e}")

        logger.info(f"[PortfolioStrategy] Iteration {self._iteration_count}: "
                   f"{len(all_signals)} signals from {len(self._sub_strategies)} strategies")
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
                from lumibot.entities import Asset

                asset = Asset(symbol=netted.symbol, asset_type=Asset.AssetType.STOCK)

                if netted.is_buy:
                    order = self.main_strategy.create_order(
                        asset=asset,
                        quantity=int(netted.abs_quantity),
                        side="buy",
                        limit_price=netted.price
                    )
                else:
                    order = self.main_strategy.create_order(
                        asset=asset,
                        quantity=int(netted.abs_quantity),
                        side="sell",
                        limit_price=netted.price
                    )

                if order:
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
            "sub_strategies": list(self._sub_strategies.keys()),
            "netting_details": self.netting_engine.get_netting_summary(),
            "positions": self.position_manager.get_all_positions()
        }

    # Delegate data access to main strategy
    def get_last_price(self, symbol: str) -> Optional[float]:
        """Get last price for a symbol."""
        from lumibot.entities import Asset
        asset = Asset(symbol=symbol, asset_type=Asset.AssetType.STOCK)
        return self.main_strategy.get_last_price(asset)

    def get_historical_prices(self, symbol: str, **kwargs):
        """Get historical prices for a symbol."""
        from lumibot.entities import Asset
        asset = Asset(symbol=symbol, asset_type=Asset.AssetType.STOCK)
        return self.main_strategy.get_historical_prices(asset, **kwargs)


# Convenience exports
__all__ = [
    'PortfolioStrategy',
    'SubStrategy',
    'OrderNettingEngine',
    'PortfolioPositionManager',
    'TradingSignal',
    'NettedOrder',
    'SignalType',
]
