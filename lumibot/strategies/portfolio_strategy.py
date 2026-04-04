"""
Portfolio Strategy System for LumiBot - Order Interception Approach

This module implements portfolio-level order netting by intercepting orders
at the broker level, similar to VNPy's PortfolioStrategy architecture.

Key Features:
- Intercept submit_order() calls from sub-strategies
- Accumulate orders per symbol
- Net opposing orders (buy vs sell) before execution
- Single execution point with flushed netted orders

Usage:
    # In your main strategy's initialize:
    from lumibot.strategies.portfolio_strategy import OrderInterceptor, run_strategy_with_interception

    self.interceptor = OrderInterceptor(self, min_net_quantity=100)

    # In on_trading_iteration:
    self.interceptor.start_interception()

    # Run sub-strategies (orders are intercepted)
    run_strategy_with_interception(sub_strategy, "strategy_id", self.interceptor)

    # Net and execute orders
    executed = self.interceptor.execute_netted_orders(strategy=self, lot_size=100)

How Order Interception Works:
    1. Sub-strategies call submit_order() normally
    2. OrderInterceptor intercepts and accumulates orders (no real submission)
    3. After all sub-strategies run, execute_netted_orders() is called
    4. Netted orders are calculated and submitted to real broker
    5. Only ONE order per symbol is actually executed
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Type, TYPE_CHECKING

from lumibot.strategies.strategy import Strategy

if TYPE_CHECKING:
    from lumibot.entities import Asset, Order, Position
    from lumibot.brokers import Broker

logger = logging.getLogger(__name__)


@dataclass
class AccumulatedOrder:
    """Represents an accumulated order before netting."""
    strategy_id: str
    symbol: str
    side: str  # "buy" or "sell"
    quantity: float
    price: Optional[float] = None
    order_type: str = "market"
    timestamp: datetime = field(default_factory=datetime.now)
    original_order: Optional['Order'] = None


@dataclass
class NettedOrder:
    """Represents a netted order ready for execution."""
    symbol: str
    net_quantity: float  # Positive for buy, negative for sell
    component_orders: List[AccumulatedOrder] = field(default_factory=list)
    price: Optional[float] = None

    @property
    def should_execute(self) -> bool:
        """Returns True if the netted order has non-zero quantity."""
        return self.net_quantity != 0

    @property
    def side(self) -> str:
        """Returns 'buy' for positive, 'sell' for negative."""
        return "buy" if self.net_quantity > 0 else "sell"

    @property
    def abs_quantity(self) -> float:
        """Returns absolute quantity."""
        return abs(self.net_quantity)


class OrderInterceptor:
    """
    Intercepts orders from sub-strategies and nets them before execution.

    This is the core component for order interception approach:
    1. Start interception mode
    2. Sub-strategies call submit_order() -> orders are accumulated
    3. Stop interception and calculate netted orders
    4. Execute only netted orders to real broker

    Example:
        interceptor = OrderInterceptor(main_strategy, min_net_quantity=100)

        # Start intercepting
        interceptor.start_interception()

        # Run sub-strategies (orders are caught, not executed)
        run_strategy_with_interception(strategy_a, "strategy_a", interceptor)
        run_strategy_with_interception(strategy_b, "strategy_b", interceptor)

        # Net and execute
        executed = interceptor.execute_netted_orders(
            strategy=main_strategy,
            lot_size=100,
            max_positions=10,
            position_dict=positions
        )

    Order Netting Example:
        Strategy A: BUY 600519.SH x 100
        Strategy B: SELL 600519.SH x 50
        Net result: BUY 600519.SH x 50 (ONE order executed)
    """

    def __init__(self, main_strategy: 'Strategy', min_net_quantity: float = 100):
        """
        Initialize the order interceptor.

        Args:
            main_strategy: The main portfolio strategy
            min_net_quantity: Minimum quantity threshold for netted orders
        """
        self.main_strategy = main_strategy
        self.min_net_quantity = min_net_quantity
        self._accumulated_orders: Dict[str, List[AccumulatedOrder]] = {}
        self._is_intercepting = False
        self._original_submit_methods: Dict[str, Any] = {}

    def start_interception(self):
        """
        Start intercepting orders.

        All subsequent submit_order() calls will be accumulated
        instead of being sent to the real broker.
        """
        self._accumulated_orders.clear()
        self._original_submit_methods.clear()
        self._is_intercepting = True
        logger.debug("[OrderInterceptor] Started intercepting orders")

    def stop_interception(self):
        """
        Stop intercepting orders.

        Subsequent submit_order() calls will pass through to real broker.
        """
        self._is_intercepting = False
        logger.debug("[OrderInterceptor] Stopped intercepting orders")

    def intercept_order(self, order: 'Order', strategy_id: str) -> Optional['Order']:
        """
        Intercept an order from a sub-strategy.

        During interception mode, orders are stored instead of submitted.
        Outside interception mode, orders pass through to real broker.

        Args:
            order: The order to intercept
            strategy_id: ID of the strategy submitting the order

        Returns:
            During interception: None (order is stored)
            Outside interception: The submitted order from real broker
        """
        if not self._is_intercepting:
            # Pass through to real broker
            return self.main_strategy.broker.submit_order(order)

        # Extract symbol from order
        symbol = order.asset.symbol if hasattr(order.asset, 'symbol') else str(order.asset)

        # Create accumulated order record
        accumulated = AccumulatedOrder(
            strategy_id=strategy_id,
            symbol=symbol,
            side=order.side,
            quantity=order.quantity,
            price=getattr(order, 'limit_price', None),
            order_type=getattr(order, 'order_type', 'market'),
            original_order=order
        )

        # Store by symbol
        if symbol not in self._accumulated_orders:
            self._accumulated_orders[symbol] = []
        self._accumulated_orders[symbol].append(accumulated)

        logger.debug(f"[OrderInterceptor] Intercepted: {strategy_id} -> "
                    f"{order.side} {symbol} x {order.quantity}")

        return None  # No order submitted yet

    def get_netted_orders(self) -> List[NettedOrder]:
        """
        Calculate netted orders from accumulated orders.

        Returns:
            List of NettedOrder objects with net quantities per symbol
        """
        netted_orders = []

        for symbol, orders in self._accumulated_orders.items():
            if not orders:
                continue

            # Calculate net quantity (buy = +, sell = -)
            net_quantity = 0.0
            prices = []

            for o in orders:
                if o.side == "buy":
                    net_quantity += o.quantity
                else:  # sell
                    net_quantity -= o.quantity

                if o.price is not None:
                    prices.append(o.price)

            # Get average price if specified
            avg_price = sum(prices) / len(prices) if prices else None

            # Create netted order
            netted = NettedOrder(
                symbol=symbol,
                net_quantity=net_quantity,
                component_orders=orders,
                price=avg_price
            )

            # Only include if meets minimum quantity threshold
            if abs(net_quantity) >= self.min_net_quantity:
                netted_orders.append(netted)
                logger.info(f"[OrderInterceptor] Netted: {symbol} -> "
                           f"{'buy' if net_quantity > 0 else 'sell'} {abs(net_quantity)} "
                           f"(from {len(orders)} orders)")
            else:
                logger.info(f"[OrderInterceptor] Netted to zero: {symbol} "
                           f"(net_qty={net_quantity}, below threshold)")

        return netted_orders

    def get_accumulation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of accumulated orders before netting.

        Returns:
            Dict with order details per symbol
        """
        summary = {}
        for symbol, orders in self._accumulated_orders.items():
            net_qty = sum(
                o.quantity if o.side == "buy" else -o.quantity
                for o in orders
            )
            summary[symbol] = {
                "order_count": len(orders),
                "net_quantity": net_qty,
                "strategies": list(set(o.strategy_id for o in orders)),
                "orders": [
                    {"strategy": o.strategy_id, "side": o.side, "qty": o.quantity}
                    for o in orders
                ]
            }
        return summary

    def execute_netted_orders(self, strategy: Optional['Strategy'] = None,
                               lot_size: int = 100,
                               max_positions: int = 10,
                               position_dict: Optional[Dict] = None) -> List['Order']:
        """
        Execute netted orders to the real broker.

        This is the key method that:
        1. Calculates net quantities per symbol
        2. Creates orders for net quantities
        3. Submits to real broker

        Args:
            strategy: The main strategy to create orders through (default: self.main_strategy)
            lot_size: Lot size for rounding (default 100 for A-shares)
            max_positions: Maximum number of positions
            position_dict: Optional dict to track positions

        Returns:
            List of actually submitted orders
        """
        strategy = strategy or self.main_strategy
        self._is_intercepting = False  # Stop interception

        netted_orders = self.get_netted_orders()
        executed_orders = []
        current_positions = len(position_dict) if position_dict else 0

        for netted in netted_orders:
            if not netted.should_execute:
                continue

            symbol = netted.symbol
            quantity = int(netted.abs_quantity)

            # Round to lot size
            quantity = (quantity // lot_size) * lot_size
            if quantity <= 0:
                continue

            try:
                from lumibot.entities import Asset

                asset = Asset(symbol=symbol, asset_type=Asset.AssetType.STOCK)
                last_price = strategy.get_last_price(asset)

                if last_price is None or last_price <= 0:
                    logger.warning(f"[OrderInterceptor] Skip {symbol}: no valid price")
                    continue

                # Check position limit for buys
                if netted.net_quantity > 0 and current_positions >= max_positions:
                    logger.info(f"[OrderInterceptor] Skip {symbol}: max positions reached")
                    continue

                # Create and submit order to REAL broker
                order = strategy.create_order(asset, quantity, netted.side)
                submitted = strategy.broker.submit_order(order)
                executed_orders.append(submitted)

                # Update position tracking
                if position_dict is not None:
                    if netted.net_quantity > 0:
                        position_dict[symbol] = {
                            'quantity': quantity,
                            'entry_price': last_price,
                            'entry_date': strategy.get_datetime() if hasattr(strategy, 'get_datetime') else None,
                        }
                        current_positions += 1
                    else:
                        position_dict.pop(symbol, None)
                        current_positions -= 1

                logger.info(f"[OrderInterceptor] Executed: {netted.side.upper()} "
                           f"{symbol} x {quantity} @ {last_price:.2f}")

            except Exception as e:
                logger.error(f"[OrderInterceptor] Error executing {symbol}: {e}")

        # Clear accumulated orders
        self._accumulated_orders.clear()

        return executed_orders


def run_strategy_with_interception(strategy_instance: 'Strategy',
                                    strategy_id: str,
                                    interceptor: OrderInterceptor):
    """
    Run a strategy instance with order interception.

    Temporarily replaces the strategy's broker submit_order method
    to route orders through the interceptor.

    Args:
        strategy_instance: The strategy instance to run
        strategy_id: Unique identifier for this strategy
        interceptor: The OrderInterceptor to use

    Usage:
        interceptor = OrderInterceptor(main_strategy)

        # During on_trading_iteration:
        interceptor.start_interception()

        # Run sub-strategies (orders are intercepted)
        run_strategy_with_interception(gap_fade_strategy, "gap_fade", interceptor)
        run_strategy_with_interception(momentum_strategy, "momentum", interceptor)

        # Execute netted orders
        executed = interceptor.execute_netted_orders(lot_size=100)
    """
    # Save original broker's submit_order
    original_broker = strategy_instance._broker
    original_submit = original_broker.submit_order

    # Create intercepted submit function
    def intercepted_submit(order):
        return interceptor.intercept_order(order, strategy_id)

    # Temporarily replace broker's submit_order
    strategy_instance._broker.submit_order = intercepted_submit

    try:
        # Run the strategy's trading iteration
        # This will call submit_order() which gets intercepted
        on_trading = getattr(strategy_instance, 'on_trading_iteration', None)
        if on_trading:
            on_trading()

    finally:
        # Restore original broker submit_order
        strategy_instance._broker = original_broker
        strategy_instance._broker.submit_order = original_submit


class CombinedPortfolioStrategy(Strategy):
    """
    Generic portfolio combiner that runs multiple strategies with order netting.

    Strategies and their parameters are passed in via `strategies_config`:
    [
        (StrategyClass1, "strategy_id_1", {"param1": value1, ...}),
        (StrategyClass2, "strategy_id_2", {"param2": value2, ...}),
        ...
    ]

    Order Interception:
    - Each strategy runs independently with its own logic
    - All submit_order() calls are intercepted
    - Orders are netted by symbol
    - Only ONE net order per symbol is executed

    Benefits:
    - Reusable with any combination of strategies
    - Exit conditions handled by each strategy
    - Reduces unnecessary trades via netting

    Usage:
        from lumibot.strategies.portfolio_strategy import CombinedPortfolioStrategy

        # Define strategies to combine
        strategies_config = [
            (PolicyGapRetailReversal, "gap_fade", {
                "symbols": ["000001.SZ", "600519.SH"],
                "gap_min": 0.02,
            }),
            (InstitutionalFlowDivergence, "momentum", {
                "symbols": None,
                "full_data": data_dict,
                "params": params,
            }),
        ]

        # Backtest
        results = CombinedPortfolioStrategy.backtest(
            PandasDataBacktesting,
            start_date,
            end_date,
            parameters={
                "strategies_config": strategies_config,
                "lot_size": 100,
                "max_positions": 12,
            }
        )

        # Live trading
        strategy = CombinedPortfolioStrategy(
            broker=broker,
            parameters={
                "strategies_config": strategies_config,
                "lot_size": 100,
                "max_positions": 12,
            }
        )
    """

    def initialize(self,
                   strategies_config: List[Tuple[Type['Strategy'], str, Dict]],
                   lot_size: int = 100,
                   max_positions: int = 12):
        """
        Initialize the combined portfolio.

        Args:
            strategies_config: List of (StrategyClass, strategy_id, parameters) tuples
            lot_size: Lot size for order rounding (default 100 for A-shares)
            max_positions: Maximum number of positions
        """
        from typing import Tuple

        self.sleeptime = "1D"
        self.lot_size = lot_size
        self.max_positions = max_positions
        self.strategies_config = strategies_config

        # Position tracking
        self.positions_dict = {}

        # Create order interceptor
        self.interceptor = OrderInterceptor(
            main_strategy=self,
            min_net_quantity=lot_size
        )

        # Create strategy instances
        self._sub_strategies: Dict[str, 'Strategy'] = {}

        for strategy_class, strategy_id, params in strategies_config:
            # Create instance without __init__
            instance = strategy_class.__new__(strategy_class)

            # Set essential attributes from main strategy
            instance._broker = self._broker
            instance._data_source = self._data_source
            instance.datetime = self.datetime if hasattr(self, 'datetime') else None

            # Call strategy's initialize with params
            init_method = getattr(instance, 'initialize', None)
            if init_method:
                # Filter params to only those accepted by initialize
                import inspect
                sig = inspect.signature(init_method)
                valid_params = {
                    k: v for k, v in params.items()
                    if k in sig.parameters or any(
                        p.kind == inspect.Parameter.VAR_KEYWORD
                        for p in sig.parameters.values()
                    )
                }
                init_method(**valid_params)

            self._sub_strategies[strategy_id] = instance

        self.log_message(f"\n{'='*70}")
        self.log_message("Combined Portfolio Strategy (Order Interception Mode)")
        self.log_message(f"{'='*70}")
        self.log_message(f"Sub-strategies ({len(self._sub_strategies)}):")
        for sid, inst in self._sub_strategies.items():
            self.log_message(f"  - {sid}: {inst.__class__.__name__}")
        self.log_message(f"Order Netting: ENABLED (min_net_quantity={lot_size})")
        self.log_message(f"Max Positions: {max_positions}")
        self.log_message(f"{'='*70}")

    def on_trading_iteration(self):
        """
        Main trading iteration.

        1. Start order interception
        2. Run all sub-strategies (orders are intercepted)
           - Each sub-strategy handles its own exit conditions
           - Exit orders are also intercepted and netted
        3. Net orders and execute
        """
        current_date = self.get_datetime()
        logger.info(f"\n{'='*70}")
        logger.info(f"Trading Iteration: {current_date.strftime('%Y-%m-%d')}")
        logger.info(f"{'='*70}")

        # Update strategy datetime references
        for sid, instance in self._sub_strategies.items():
            instance.datetime = current_date

        # Start intercepting orders
        self.interceptor.start_interception()

        try:
            # Run each sub-strategy
            for strategy_id, instance in self._sub_strategies.items():
                logger.info(f"--- Running Strategy: {strategy_id} ---")
                run_strategy_with_interception(
                    instance,
                    strategy_id,
                    self.interceptor
                )

            # Net and execute orders
            logger.info("--- Netting & Executing Orders ---")
            executed = self.interceptor.execute_netted_orders(
                strategy=self,
                lot_size=self.lot_size,
                max_positions=self.max_positions,
                position_dict=self.positions_dict
            )

            if executed:
                logger.info(f"Executed {len(executed)} netted orders")
            else:
                logger.info("No orders executed (all netted to zero or below threshold)")

        finally:
            self.interceptor.stop_interception()

        # Print positions
        self._print_positions()

        self.await_market_to_close()

    def _print_positions(self):
        """Print current positions."""
        if not self.positions_dict:
            logger.info("Current Positions: None")
            return

        logger.info(f"Current Positions ({len(self.positions_dict)}):")
        total_value = 0
        for symbol, pos in self.positions_dict.items():
            try:
                from lumibot.entities import Asset
                asset = Asset(symbol=symbol, asset_type=Asset.AssetType.STOCK)
                current_price = self.get_last_price(asset)
                if current_price:
                    value = pos['quantity'] * current_price
                    pnl = (current_price - pos['entry_price']) / pos['entry_price'] * 100
                    total_value += value
                    logger.info(f"  {symbol}: {pos['quantity']} @ {pos['entry_price']:.2f} "
                          f"(current: {current_price:.2f}, PnL: {pnl:+.1f}%)")
            except Exception:
                pass

        logger.info(f"  Total Position Value: {total_value:,.0f}")

    def on_abrupt_closing(self):
        """Handle abrupt closing."""
        logger.info(f"{'='*70}")
        logger.info("Abrupt Closing - Selling All Positions")
        logger.info(f"{'='*70}")
        self.sell_all()
        self.positions_dict.clear()


# Convenience exports
__all__ = [
    'OrderInterceptor',
    'run_strategy_with_interception',
    'CombinedPortfolioStrategy',
    'AccumulatedOrder',
    'NettedOrder',
]
