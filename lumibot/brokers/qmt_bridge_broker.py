"""QMT Bridge Broker for LumiBot.

This module provides a broker integration for Chinese A-shares trading via QMT Bridge.
Supports order placement, cancellation, position management, and account tracking.

Order Types:
    - Market orders (price_type=5, latest price)
    - Limit orders (price_type=11)
    - Best 5-level immediate fill or cancel (price_type=42)

Example:
    >>> from lumibot.brokers import QMTBridgeBroker
    >>> from lumibot.data_sources import QMTBridgeData
    >>> 
    >>> data_source = QMTBridgeData(host="192.168.1.100")
    >>> broker = QMTBridgeBroker(
    ...     host="192.168.1.100",
    ...     port=8000,
    ...     api_key="your-api-key",
    ...     data_source=data_source,
    ...     account_id="your-account-id"
    ... )
    >>> 
    >>> # In a strategy
    >>> order = self.create_order("buy", asset, 100)
    >>> self.submit_order(order)
"""

import threading
import time
from datetime import datetime
from decimal import Decimal
from typing import Union

from lumibot.brokers.broker import Broker
from lumibot.entities import Asset, Order, Position
from lumibot.tools.lumibot_logger import get_logger
from lumibot.data_sources.qmt_bridge_data import qmt_bridge_normalize_symbol

logger = get_logger(__name__)

# QMT order type constants
ORDER_TYPE_BUY = 23
ORDER_TYPE_SELL = 24

# Price type constants
PRICE_TYPE_LATEST = 5  # Market order (latest price)
PRICE_TYPE_LIMIT = 11  # Limit order
PRICE_TYPE_BEST_5_FOK = 42  # Best 5-level immediate fill or cancel


class QMTBridgeBroker(Broker):
    """Broker for Chinese A-shares trading via QMT Bridge.

    This broker provides full trading capabilities for stocks on Shanghai
    and Shenzhen stock exchanges through the QMT Bridge API.

    Parameters
    ----------
    host : str
        QMT Bridge server host address.
    port : int, optional
        QMT Bridge server port, default 8000.
    api_key : str
        API key for authentication.
    account_id : str, optional
        Trading account ID. If not provided, uses default account.
    data_source : DataSource
        Data source for market data.
    connect_stream : bool, optional
        Whether to connect WebSocket stream, default True.
    **kwargs
        Additional keyword arguments passed to Broker.

    Attributes
    ----------
    IS_BACKTESTING_BROKER : bool
        Always False for live broker.

    Example
    -------
    >>> broker = QMTBridgeBroker(
    ...     host="192.168.1.100",
    ...     api_key="your-key",
    ...     account_id="12345678"
    ... )
    """

    IS_BACKTESTING_BROKER = False

    def __init__(
        self,
        config,
        data_source=None,
        connect_stream: bool = True,
        **kwargs
    ):
        """Initialize QMT Bridge broker.

        Parameters
        ----------
        config: dict
            host : str
                QMT Bridge server host address.
            port : int, optional
                QMT Bridge server port, default 8000.
            api_key : str
                API key for authentication.
            account_id : str, optional
                Trading account ID.
        data_source : DataSource
            Data source for market data.
        connect_stream : bool, optional
            Whether to connect WebSocket stream.
        **kwargs
            Additional keyword arguments.
        """
        self.host = config["host"]
        self.port = config.get("port", 8000)
        self._api_key = config.get("api_key", "")
        self.account_id = config.get("account_id", "")
        self._client = None

        # Set market for Chinese exchanges (Shanghai Stock Exchange)
        config = kwargs.get("config", {}) or {}
        config["MARKET"] = config.get("MARKET", "SSE")

        super().__init__(
            name="qmt_bridge",
            connect_stream=connect_stream,
            data_source=data_source,
            config=config,
            **kwargs
        )

    def _get_client(self):
        """Get or create the QMT Bridge client.

        Returns
        -------
        QMTClient
            The QMT Bridge client instance.
        """
        if self._client is None:
            try:
                from qmt_bridge import QMTClient
                self._client = QMTClient(
                    host=self.host,
                    port=self.port,
                    api_key=self._api_key
                )
            except ImportError as e:
                raise ImportError(
                    "qmt_bridge package is required for QMT Bridge broker. "
                    "Please install it with: pip install qmt-bridge"
                ) from e
        return self._client

    def _normalize_symbol(self, asset: Asset) -> str:
        """Normalize asset symbol to QMT format.

        Parameters
        ----------
        asset : Asset
            The asset to normalize.

        Returns
        -------
        str
            Symbol in QMT format (e.g., "600519.SH", "000001.SZ").

        Supported Input Formats:
            - "600519" or "SH600519" → "600519.SH"
            - "000001" or "SZ000001" → "000001.SZ"
            - "600519.SH" → "600519.SH" (already normalized)
        """
        symbol = asset.symbol.upper()

        # Already in exchange.suffix format
        if "." in symbol:
            return symbol

        return qmt_bridge_normalize_symbol(symbol)

    # ==================== Order Methods ====================

    def _submit_order(self, order: Order) -> Order:
        """Submit an order to QMT.

        Parameters
        ----------
        order : Order
            The order to submit.

        Returns
        -------
        Order
            The submitted order with updated status.
        """
        client = self._get_client()
        symbol = self._normalize_symbol(order.asset)

        # Determine order type (buy/sell)
        if order.side.lower() in ("buy", "bot"):
            qmt_order_type = ORDER_TYPE_BUY
        elif order.side.lower() in ("sell", "sld"):
            qmt_order_type = ORDER_TYPE_SELL
        else:
            order.set_error(f"Invalid order side: {order.side}")
            return order

        # Determine price type and price
        if order.order_type == Order.OrderType.MARKET:
            price_type = PRICE_TYPE_LATEST
            price = 0.0  # Market orders don't specify price
        elif order.order_type == Order.OrderType.LIMIT:
            price_type = PRICE_TYPE_LIMIT
            price = float(order.limit_price) if order.limit_price else 0.0
        else:
            # Default to limit order
            price_type = PRICE_TYPE_LIMIT
            price = float(order.limit_price) if order.limit_price else 0.0

        # try:
        #     result = client.place_order(
        #         stock_code=symbol,
        #         order_type=qmt_order_type,
        #         order_volume=int(order.quantity),
        #         price_type=price_type,
        #         price=price,
        #         strategy_name=order.strategy or "",
        #         order_remark=f"LumiBot-{order.identifier}",
        #         account_id=self.account_id,
        #     )

        #     # Check result
        #     if result and "order_id" in result:
        #         order.identifier = str(result["order_id"])
        #         order.set_transmitted()
        #         self.logger.info(
        #             f"Order submitted: {order.side} {order.quantity} {symbol} "
        #             f"@ {price} (ID: {order.identifier})"
        #         )
        #     else:
        #         error_msg = result.get("error", "Unknown error") if result else "No response"
        #         order.set_error(f"Order submission failed: {error_msg}")
        #         self.logger.error(f"Order submission failed: {error_msg}")

        # except Exception as e:
        #     order.set_error(f"Exception during order submission: {e}")
        #     self.logger.error(f"Exception during order submission: {e}")

        return order

    def cancel_order(self, order: Order) -> None:
        """Cancel an order.

        Parameters
        ----------
        order : Order
            The order to cancel.
        """
        if not order.identifier:
            self.logger.warning("Cannot cancel order without identifier")
            return

        client = self._get_client()

        try:
            result = client.cancel_order(
                order_id=int(order.identifier),
                account_id=self.account_id,
            )

            if result and result.get("success"):
                self.logger.info(f"Order {order.identifier} canceled successfully")
            else:
                error_msg = result.get("error", "Unknown error") if result else "No response"
                self.logger.error(f"Failed to cancel order {order.identifier}: {error_msg}")

        except Exception as e:
            self.logger.error(f"Exception during order cancellation: {e}")

    def _modify_order(
        self,
        order: Order,
        limit_price: Union[float, None] = None,
        stop_price: Union[float, None] = None
    ):
        """Modify an order.

        Note: QMT does not support order modification directly.
        Orders must be canceled and resubmitted.

        Parameters
        ----------
        order : Order
            The order to modify.
        limit_price : float, optional
            New limit price.
        stop_price : float, optional
            New stop price (not supported).
        """
        # QMT doesn't support order modification
        # Cancel and resubmit
        self.logger.warning(
            "QMT does not support order modification. "
            "Canceling and resubmitting order."
        )

        self.cancel_order(order)

        # Wait briefly for cancellation
        time.sleep(0.1)

        # Update order with new price
        if limit_price:
            order.limit_price = limit_price

        # Reset order status
        order.status = Order.OrderStatus.NEW

        # Resubmit
        self._submit_order(order)

    # ==================== Position Methods ====================

    def _pull_positions(self, strategy) -> list[Position]:
        """Pull all positions from the broker.

        Parameters
        ----------
        strategy : Strategy
            The strategy requesting positions.

        Returns
        -------
        list[Position]
            List of Position objects.
        """
        client = self._get_client()
        positions = []

        try:
            result = client.query_positions(account_id=self.account_id)

            if not result or "data" not in result:
                return positions

            for pos_data in result.get("data", []):
                position = self._parse_broker_position(pos_data, strategy)
                if position and position.quantity != 0:
                    positions.append(position)

        except Exception as e:
            self.logger.error(f"Error fetching positions: {e}")

        return positions

    def _pull_position(self, strategy, asset: Asset) -> Union[Position, None]:
        """Pull a single position from the broker.

        Parameters
        ----------
        strategy : Strategy
            The strategy requesting the position.
        asset : Asset
            The asset to get the position for.

        Returns
        -------
        Position or None
            The position, or None if not found.
        """
        client = self._get_client()
        symbol = self._normalize_symbol(asset)

        try:
            result = client.query_single_position(
                stock_code=symbol,
                account_id=self.account_id,
            )

            if result and "data" in result:
                pos_data = result["data"]
                if pos_data:
                    return self._parse_broker_position(pos_data, strategy)

        except Exception as e:
            self.logger.error(f"Error fetching position for {symbol}: {e}")

        return None

    def _parse_broker_position(self, pos_data: dict, strategy) -> Union[Position, None]:
        """Parse a broker position response to Position object.

        Parameters
        ----------
        pos_data : dict
            Position data from broker.
        strategy : Strategy
            The strategy.

        Returns
        -------
        Position or None
            Parsed Position object.
        """
        if not pos_data:
            return None

        try:
            # Extract symbol
            stock_code = pos_data.get("stock_code", "")
            if "." in stock_code:
                symbol = stock_code.split(".")[0]
            else:
                symbol = stock_code

            # Create asset
            asset = Asset(symbol=symbol, asset_type=Asset.AssetType.STOCK)

            # Extract quantity
            quantity = pos_data.get("volume", 0) or pos_data.get("position", 0)

            # Create position
            position = Position(
                strategy=strategy.name if hasattr(strategy, 'name') else str(strategy),
                asset=asset,
                quantity=Decimal(str(quantity)),
            )

            # Set additional attributes
            position.avg_price = Decimal(str(pos_data.get("cost_price", 0) or 0))
            position.market_value = Decimal(str(pos_data.get("market_value", 0) or 0))

            return position

        except Exception as e:
            self.logger.error(f"Error parsing position: {e}")
            return None

    # ==================== Account Methods ====================

    def _get_balances_at_broker(self, quote_asset, strategy) -> tuple:
        """Get account balances from the broker.

        Parameters
        ----------
        quote_asset : Asset
            The quote asset (typically CNY for Chinese markets).
        strategy : Strategy
            The strategy.

        Returns
        -------
        tuple
            (cash, positions_value, total_value)
        """
        client = self._get_client()

        try:
            result = client.query_asset(account_id=self.account_id)

            if not result or "data" not in result:
                return (0.0, 0.0, 0.0)

            data = result["data"]

            # Extract balance information
            cash = float(data.get("cash", 0) or data.get("available_cash", 0) or 0)
            positions_value = float(data.get("market_value", 0) or 0)
            total_value = float(data.get("total_asset", 0) or (cash + positions_value))

            return (cash, positions_value, total_value)

        except Exception as e:
            self.logger.error(f"Error fetching balances: {e}")
            return (0.0, 0.0, 0.0)

    def get_historical_account_value(self) -> dict:
        """Get historical account value.

        Note: QMT does not provide historical account value data.
        Returns empty dict for compatibility.

        Returns
        -------
        dict
            Empty dictionary.
        """
        # QMT doesn't provide historical account value
        return {}

    # ==================== Order Query Methods ====================

    def _parse_broker_order(
        self,
        response: dict,
        strategy_name: str,
        strategy_object=None
    ) -> Union[Order, None]:
        """Parse a broker order response to Order object.

        Parameters
        ----------
        response : dict
            Order data from broker.
        strategy_name : str
            Name of the strategy.
        strategy_object : Strategy, optional
            The strategy object.

        Returns
        -------
        Order or None
            Parsed Order object.
        """
        if not response:
            return None

        try:
            # Extract symbol
            stock_code = response.get("stock_code", "")
            if "." in stock_code:
                symbol = stock_code.split(".")[0]
            else:
                symbol = stock_code

            # Create asset
            asset = Asset(symbol=symbol, asset_type=Asset.AssetType.STOCK)

            # Determine side
            order_type = response.get("order_type", 0)
            if order_type == ORDER_TYPE_BUY:
                side = "buy"
            elif order_type == ORDER_TYPE_SELL:
                side = "sell"
            else:
                side = "unknown"

            # Create order
            order = Order(
                strategy=strategy_name,
                asset=asset,
                quantity=response.get("order_volume", 0),
                side=side,
                limit_price=response.get("price", None),
            )

            # Set identifier
            order.identifier = str(response.get("order_id", ""))

            # Set status
            order_status = response.get("order_status", 0)
            filled_volume = response.get("filled_volume", 0)
            total_volume = response.get("order_volume", 0)

            if order_status == 0:
                order.status = Order.OrderStatus.NEW
            elif order_status == 1:
                order.status = Order.OrderStatus.PARTIALLY_FILLED
            elif order_status == 2:
                order.status = Order.OrderStatus.FILLED
            elif order_status == 3:
                order.status = Order.OrderStatus.CANCELED
            else:
                order.status = Order.OrderStatus.NEW

            # Set fill information
            order.avg_fill_price = response.get("trade_price", None)
            order.filled_quantity = filled_volume

            return order

        except Exception as e:
            self.logger.error(f"Error parsing broker order: {e}")
            return None

    def _pull_broker_order(self, identifier: str) -> Union[Order, None]:
        """Pull a single order from the broker.

        Parameters
        ----------
        identifier : str
            Order identifier.

        Returns
        -------
        Order or None
            The order, or None if not found.
        """
        client = self._get_client()

        try:
            result = client.query_single_order(
                order_id=int(identifier),
                account_id=self.account_id,
            )

            if result and "data" in result:
                return self._parse_broker_order(result["data"], "")

        except Exception as e:
            self.logger.error(f"Error fetching order {identifier}: {e}")

        return None

    def _pull_broker_all_orders(self) -> list[dict]:
        """Pull all open orders from the broker.

        Returns
        -------
        list[dict]
            List of order dictionaries.
        """
        client = self._get_client()
        orders = []

        try:
            result = client.query_orders(
                account_id=self.account_id,
                cancelable_only=False,
            )

            if result and "data" in result:
                orders = result["data"]

        except Exception as e:
            self.logger.error(f"Error fetching all orders: {e}")

        return orders

    # ==================== Stream Methods ====================

    def _get_stream_object(self):
        """Get the stream connection object.

        Note: QMT Bridge uses WebSocket for real-time updates.
        Returns the client for WebSocket operations.

        Returns
        -------
        QMTClient or None
            The client instance, or None if WebSocket not available.
        """
        # Return client for WebSocket operations
        # Actual WebSocket handling is done via subscribe methods
        return self._get_client()

    def _register_stream_events(self):
        """Register stream event handlers.

        Subscribes to trade events via WebSocket.
        """
        client = self._get_client()

        if hasattr(client, 'subscribe_trade_events'):
            try:
                # Subscribe to trade event callbacks
                client.subscribe_trade_events(
                    callback=self._on_trade_event_callback,
                )
                self._stream_established()
                self.logger.info("Trade event stream registered")
            except Exception as e:
                self.logger.error(f"Error registering stream events: {e}")

    def _run_stream(self):
        """Run the WebSocket stream.

        Note: QMT Bridge WebSocket runs in background.
        This method keeps the thread alive.
        """
        self.logger.info("QMT Bridge stream thread started")

        # Keep thread alive to receive callbacks
        while not self._stop_event.is_set():
            time.sleep(1)

        self.logger.info("QMT Bridge stream thread stopped")

    def _on_trade_event_callback(self, event_data: dict):
        """Handle trade event callback from WebSocket.

        Parameters
        ----------
        event_data : dict
            Trade event data from QMT Bridge.
        """
        try:
            order_id = str(event_data.get("order_id", ""))
            event_type = event_data.get("event_type", "")

            # Find the tracked order
            order = self.get_tracked_order(order_id)
            if not order:
                self.logger.warning(f"Received event for unknown order: {order_id}")
                return

            # Process event based on type
            if event_type == "filled":
                price = event_data.get("trade_price", 0)
                quantity = event_data.get("filled_volume", 0)
                self._process_trade_event(
                    order,
                    self.FILLED_ORDER,
                    price=price,
                    filled_quantity=quantity,
                    multiplier=1,
                )
            elif event_type == "partial_fill":
                price = event_data.get("trade_price", 0)
                quantity = event_data.get("filled_volume", 0)
                self._process_trade_event(
                    order,
                    self.PARTIALLY_FILLED_ORDER,
                    price=price,
                    filled_quantity=quantity,
                    multiplier=1,
                )
            elif event_type == "canceled":
                self._process_trade_event(order, self.CANCELED_ORDER)
            elif event_type == "rejected":
                self._process_trade_event(
                    order,
                    self.ERROR_ORDER,
                    error=event_data.get("error", "Order rejected"),
                )

        except Exception as e:
            self.logger.error(f"Error processing trade event: {e}")
