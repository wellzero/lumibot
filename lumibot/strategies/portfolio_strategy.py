"""
Portfolio Strategy System for LumiBot

This module implements a portfolio-level strategy management system similar to VNPy's
PortfolioStrategy, providing order netting, position aggregation, and multi-strategy
coordination.

Key Features:
- Combine signals from multiple strategies
- Order netting (combines buy/sell for same symbol)
- Position aggregation at portfolio level
- Cross-strategy coordination

Example:
    >>> # In your main strategy's on_trading_iteration:
    >>> signals = []
    >>> signals.extend(gap_fade_strategy.generate_signals(self))
    >>> signals.extend(momentum_strategy.generate_signals(self))
    >>>
    >>> netted = self.portfolio.net_signals(signals)
    >>> self.portfolio.execute_netted(netted)
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
    """Represents a trading signal from a strategy."""
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
        """Add a trading signal to the netting engine."""
        self._signals[signal.symbol].append(signal)
        logger.debug(f"[OrderNetting] Added signal: {signal.strategy_id} -> "
                    f"{signal.symbol} qty={signal.quantity}")

    def add_signals(self, signals: List[TradingSignal]) -> None:
        """Add multiple trading signals."""
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
                           f"net_qty={net_quantity}")

        return netted_orders

    def get_netting_summary(self) -> Dict[str, Dict]:
        """Get a summary of the netting process."""
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
    """Manages positions at the portfolio level across all strategies."""

    def __init__(self):
        self._positions: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"quantity": 0.0, "avg_cost": 0.0, "strategy_positions": {}}
        )
        self._strategy_positions: Dict[str, Dict[str, float]] = defaultdict(dict)

    def update_position(self, strategy_id: str, symbol: str, quantity: float, price: float) -> None:
        """Update position after a trade execution."""
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
                    f"qty={old_qty}→{new_qty}")

    def get_portfolio_position(self, symbol: str) -> Dict[str, Any]:
        """Get portfolio-level position for a symbol."""
        return dict(self._positions[symbol])

    def get_strategy_position(self, strategy_id: str, symbol: str) -> float:
        """Get a strategy's position in a symbol."""
        return self._strategy_positions[strategy_id].get(symbol, 0.0)

    def get_all_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get all portfolio positions."""
        return {k: dict(v) for k, v in self._positions.items() if v["quantity"] != 0}

    def get_strategy_positions(self, strategy_id: str) -> Dict[str, float]:
        """Get all positions for a specific strategy."""
        return dict(self._strategy_positions[strategy_id])


class PortfolioStrategyHelper:
    """
    Helper class for portfolio strategy with order netting.

    This class provides utility methods for combining signals from multiple
    strategies and netting them before execution.

    Usage:
        # In your main strategy's initialize:
        self.portfolio = PortfolioStrategyHelper(min_net_quantity=100)

        # In on_trading_iteration:
        # 1. Collect signals from your signal generators
        signals = []
        signals.extend(self.generate_gap_fade_signals())
        signals.extend(self.generate_momentum_signals())

        # 2. Add to netting engine
        self.portfolio.add_signals(signals)

        # 3. Get netted orders
        netted = self.portfolio.get_netted_orders()

        # 4. Execute
        self.portfolio.execute_netted_orders(netted, self)
    """

    def __init__(self, min_net_quantity: float = 1.0):
        """
        Initialize the portfolio strategy helper.

        Args:
            min_net_quantity: Minimum quantity threshold for netted orders
        """
        self.netting_engine = OrderNettingEngine(min_net_quantity)
        self.position_manager = PortfolioPositionManager()
        self._iteration_count = 0

    def add_signal(self, signal: TradingSignal) -> None:
        """Add a signal to the netting engine."""
        self.netting_engine.add_signal(signal)

    def add_signals(self, signals: List[TradingSignal]) -> None:
        """Add multiple signals to the netting engine."""
        self.netting_engine.add_signals(signals)

    def clear_signals(self) -> None:
        """Clear all pending signals."""
        self.netting_engine.clear_signals()

    def get_netted_orders(self) -> List[NettedOrder]:
        """Get netted orders from collected signals."""
        return self.netting_engine.get_netted_orders()

    def execute_netted_orders(self, orders: List[NettedOrder], strategy: 'Strategy',
                              lot_size: int = 100, max_positions: int = 10,
                              position_dict: Optional[Dict] = None) -> List['Order']:
        """
        Execute netted orders through the strategy.

        Args:
            orders: List of netted orders to execute
            strategy: The strategy to execute through
            lot_size: Lot size for rounding (default 100 for A-shares)
            max_positions: Maximum number of positions
            position_dict: Optional dict to track positions

        Returns:
            List of submitted orders
        """
        from lumibot.entities import Asset

        executed_orders = []
        current_positions = len(position_dict) if position_dict else 0

        for netted in orders:
            if not netted.should_execute:
                continue

            symbol = netted.symbol
            quantity = int(netted.abs_quantity)

            # Round to lot size
            quantity = (quantity // lot_size) * lot_size
            if quantity <= 0:
                continue

            try:
                asset = Asset(symbol=symbol, asset_type=Asset.AssetType.STOCK)
                last_price = strategy.get_last_price(asset)

                if last_price is None or last_price <= 0:
                    continue

                # Check position limit for buys
                if netted.is_buy and current_positions >= max_positions:
                    logger.info(f"Skip {symbol}: max positions reached")
                    continue

                # Create and submit order
                side = "buy" if netted.is_buy else "sell"
                order = strategy.create_order(asset, quantity, side)
                strategy.submit_order(order)
                executed_orders.append(order)

                # Update position tracking
                if position_dict is not None:
                    if netted.is_buy:
                        position_dict[symbol] = {
                            'quantity': quantity,
                            'entry_price': last_price,
                            'entry_date': strategy.get_datetime() if hasattr(strategy, 'get_datetime') else None,
                        }
                        current_positions += 1
                    else:
                        position_dict.pop(symbol, None)
                        current_positions -= 1

                # Update portfolio position manager
                fill_qty = quantity if netted.is_buy else -quantity
                self.position_manager.update_position(
                    "portfolio", symbol, fill_qty, last_price
                )

                logger.info(f"Executed: {side.upper()} {symbol} x {quantity} @ {last_price:.2f}")

            except Exception as e:
                logger.error(f"Error executing {symbol}: {e}")

        return executed_orders

    def get_netting_summary(self) -> Dict[str, Any]:
        """Get a summary of the current netting state."""
        return {
            "netting_details": self.netting_engine.get_netting_summary(),
            "positions": self.position_manager.get_all_positions()
        }

    def new_iteration(self) -> None:
        """Start a new iteration (clears previous signals)."""
        self._iteration_count += 1
        self.clear_signals()


# Convenience exports
__all__ = [
    'PortfolioStrategyHelper',
    'OrderNettingEngine',
    'PortfolioPositionManager',
    'TradingSignal',
    'NettedOrder',
]
