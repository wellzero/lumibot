"""Tests for QMT Bridge Broker and Data Source.

This module contains unit tests for the QMT Bridge integration with LumiBot.
Tests are designed to run without actual QMT Bridge connection by mocking
the API responses.

Run with:
    pytest tests/test_qmt_bridge.py -v
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pandas as pd
import pytz


# ==================== Fixtures ====================

@pytest.fixture
def mock_qmt_client():
    """Create a mock QMT client."""
    client = MagicMock()
    return client


@pytest.fixture
def qmt_bridge_data(mock_qmt_client):
    """Create a QMTBridgeData instance with mocked client."""
    # Patch the qmt_bridge import inside the data source module
    with patch('qmt_bridge.QMTClient', return_value=mock_qmt_client):
        from lumibot.data_sources.qmt_bridge_data import QMTBridgeData

        config = {
            "host": "192.168.1.100",
            "port": 8000,
            "api_key": "test-key",
        }
        data_source = QMTBridgeData(config)
        # Set the client directly to avoid lazy initialization
        data_source._client = mock_qmt_client
        return data_source


@pytest.fixture
def qmt_bridge_broker(mock_qmt_client):
    """Create a QMTBridgeBroker instance with mocked client."""
    # Patch the qmt_bridge import inside the broker module
    with patch('qmt_bridge.QMTClient', return_value=mock_qmt_client):
        from lumibot.brokers.qmt_bridge_broker import QMTBridgeBroker
        from lumibot.data_sources.qmt_bridge_data import QMTBridgeData

        data_source = MagicMock(spec=QMTBridgeData)
        data_source.tzinfo = pytz.timezone("Asia/Shanghai")
        data_source.get_datetime.return_value = datetime.now(pytz.timezone("Asia/Shanghai"))

        config = {
            "host": "192.168.1.100",
            "port": 8000,
            "api_key": "test-key",
            "account_id": "12345678",
        }
        broker = QMTBridgeBroker(
            config,
            data_source=data_source,
            connect_stream=False,
        )
        broker._client = mock_qmt_client
        return broker


@pytest.fixture
def sample_asset():
    """Create a sample Chinese A-share asset."""
    from lumibot.entities import Asset
    return Asset(symbol="000001.SZ", asset_type=Asset.AssetType.STOCK)


@pytest.fixture
def sample_position_data():
    """Sample position data from QMT."""
    return {
        "stock_code": "000001.SZ",
        "volume": 1000,
        "cost_price": 12.50,
        "market_value": 12500.00,
    }


@pytest.fixture
def sample_order_data():
    """Sample order data from QMT."""
    return {
        "order_id": 12345,
        "stock_code": "000001.SZ",
        "order_type": 23,  # Buy
        "order_volume": 100,
        "price": 12.50,
        "order_status": 0,  # New
        "filled_volume": 0,
        "trade_price": 0,
    }


@pytest.fixture
def sample_tick_data():
    """Sample tick data from QMT."""
    return {
        "lastPrice": 12.55,
        "bid1": 12.54,
        "ask1": 12.56,
        "bidQty1": 1000,
        "askQty1": 2000,
        "volume": 1000000,
        "high": 12.80,
        "low": 12.30,
        "open": 12.40,
    }


@pytest.fixture
def sample_historical_data():
    """Sample historical price data from QMT."""
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=10),
        periods=10,
        freq="1D",
        tz="Asia/Shanghai"
    )

    return pd.DataFrame({
        "time": [int(d.timestamp() * 1000) for d in dates],
        "open": [10.0 + i * 0.1 for i in range(10)],
        "high": [10.2 + i * 0.1 for i in range(10)],
        "low": [9.8 + i * 0.1 for i in range(10)],
        "close": [10.0 + i * 0.1 for i in range(10)],
        "volume": [1000000] * 10,
    })


# ==================== Data Source Tests ====================

class TestQMTBridgeData:
    """Tests for QMTBridgeData class."""

    def test_initialization(self, qmt_bridge_data):
        """Test data source initialization."""
        assert qmt_bridge_data.SOURCE == "QMT_BRIDGE"
        assert qmt_bridge_data.host == "192.168.1.100"
        assert qmt_bridge_data.port == 8000
        assert qmt_bridge_data.dividend_type == "front"
        assert qmt_bridge_data.fill_data is True

    def test_normalize_symbol_shanghai(self, qmt_bridge_data):
        """Test symbol normalization for Shanghai stocks."""
        from lumibot.entities import Asset

        # Shanghai main board (6xxx)
        asset = Asset("600000")
        assert qmt_bridge_data._normalize_symbol(asset) == "600000.SH"

        # STAR Market (68xxxx)
        asset = Asset("688001")
        assert qmt_bridge_data._normalize_symbol(asset) == "688001.SH"

    def test_normalize_symbol_shenzhen(self, qmt_bridge_data):
        """Test symbol normalization for Shenzhen stocks."""
        from lumibot.entities import Asset

        # Shenzhen main board (0xxx)
        asset = Asset("000001")
        assert qmt_bridge_data._normalize_symbol(asset) == "000001.SZ"

        # ChiNext (3xxxx)
        asset = Asset("300001")
        assert qmt_bridge_data._normalize_symbol(asset) == "300001.SZ"

    def test_normalize_symbol_with_suffix(self, qmt_bridge_data):
        """Test symbol normalization when suffix already exists."""
        from lumibot.entities import Asset

        asset = Asset("000001.SZ")
        assert qmt_bridge_data._normalize_symbol(asset) == "000001.SZ"

    def test_convert_timestep_to_qmt(self, qmt_bridge_data):
        """Test timestep conversion."""
        assert qmt_bridge_data._convert_timestep_to_qmt("minute") == "1m"
        assert qmt_bridge_data._convert_timestep_to_qmt("5m") == "5m"
        assert qmt_bridge_data._convert_timestep_to_qmt("day") == "1d"
        assert qmt_bridge_data._convert_timestep_to_qmt("hour") == "1h"
        assert qmt_bridge_data._convert_timestep_to_qmt("week") == "1w"

    def test_get_last_price(self, qmt_bridge_data, sample_asset, mock_qmt_client, sample_tick_data):
        """Test getting last price."""
        mock_qmt_client.get_full_tick.return_value = {
            "000001.SZ": sample_tick_data
        }

        price = qmt_bridge_data.get_last_price(sample_asset)

        assert price is not None
        assert price == 12.55
        mock_qmt_client.get_full_tick.assert_called_once_with(["000001.SZ"])

    def test_get_last_price_not_found(self, qmt_bridge_data, sample_asset, mock_qmt_client):
        """Test getting last price when no data available."""
        mock_qmt_client.get_full_tick.return_value = {}

        price = qmt_bridge_data.get_last_price(sample_asset)

        assert price is None

    def test_get_quote(self, qmt_bridge_data, sample_asset, mock_qmt_client, sample_tick_data):
        """Test getting quote."""
        mock_qmt_client.get_full_tick.return_value = {
            "000001.SZ": sample_tick_data
        }

        quote = qmt_bridge_data.get_quote(sample_asset)

        assert quote is not None
        assert quote.bid == 12.54
        assert quote.ask == 12.56
        assert quote.last == 12.55
        assert quote.bid_size == 1000
        assert quote.ask_size == 2000

    def test_get_historical_prices(self, qmt_bridge_data, sample_asset, mock_qmt_client, sample_historical_data):
        """Test getting historical prices."""
        mock_qmt_client.get_history_ex.return_value = {
            "000001.SZ": sample_historical_data
        }

        bars = qmt_bridge_data.get_historical_prices(
            asset=sample_asset,
            length=5,
            timestep="day",
        )

        assert bars is not None
        assert len(bars.df) == 5
        assert "open" in bars.df.columns
        assert "close" in bars.df.columns
        assert bars.asset == sample_asset

    def test_get_historical_prices_empty_data(self, qmt_bridge_data, sample_asset, mock_qmt_client):
        """Test getting historical prices when no data available."""
        mock_qmt_client.get_history_ex.return_value = {}

        bars = qmt_bridge_data.get_historical_prices(
            asset=sample_asset,
            length=5,
            timestep="day",
        )

        assert bars is None

    def test_get_chains(self, qmt_bridge_data, sample_asset):
        """Test getting option chains (should return empty structure)."""
        chains = qmt_bridge_data.get_chains(sample_asset)

        assert chains is not None
        assert "Multiplier" in chains
        assert "Chains" in chains
        assert chains["Multiplier"] == "100"


# ==================== Broker Tests ====================

class TestQMTBridgeBroker:
    """Tests for QMTBridgeBroker class."""

    def test_initialization(self, qmt_bridge_broker):
        """Test broker initialization."""
        assert qmt_bridge_broker.name == "qmt_bridge"
        assert qmt_bridge_broker.host == "192.168.1.100"
        assert qmt_bridge_broker.port == 8000
        assert qmt_bridge_broker.account_id == "12345678"
        assert qmt_bridge_broker.IS_BACKTESTING_BROKER is False

    def test_normalize_symbol(self, qmt_bridge_broker):
        """Test symbol normalization in broker."""
        from lumibot.entities import Asset

        asset = Asset("600000")
        assert qmt_bridge_broker._normalize_symbol(asset) == "600000.SH"

    def test_get_balances_at_broker(self, qmt_bridge_broker, mock_qmt_client):
        """Test getting account balances."""
        mock_qmt_client.query_asset.return_value = {
            "data": {
                "cash": 100000.00,
                "market_value": 50000.00,
                "total_asset": 150000.00,
            }
        }

        cash, positions_value, total_value = qmt_bridge_broker._get_balances_at_broker(
            quote_asset=None,
            strategy=None,
        )

        assert cash == 100000.00
        assert positions_value == 50000.00
        assert total_value == 150000.00

    def test_get_balances_error(self, qmt_bridge_broker, mock_qmt_client):
        """Test getting balances when error occurs."""
        mock_qmt_client.query_asset.side_effect = Exception("Connection error")

        cash, positions_value, total_value = qmt_bridge_broker._get_balances_at_broker(
            quote_asset=None,
            strategy=None,
        )

        assert cash == 0.0
        assert positions_value == 0.0
        assert total_value == 0.0

    def test_parse_broker_position(self, qmt_bridge_broker, sample_position_data):
        """Test parsing broker position."""
        position = qmt_bridge_broker._parse_broker_position(
            sample_position_data,
            strategy="test_strategy",
        )

        assert position is not None
        assert position.asset.symbol == "000001"
        assert position.quantity == Decimal("1000")

    def test_parse_broker_order(self, qmt_bridge_broker, sample_order_data):
        """Test parsing broker order."""
        order = qmt_bridge_broker._parse_broker_order(
            sample_order_data,
            strategy_name="test_strategy",
        )

        assert order is not None
        assert order.identifier == "12345"
        assert order.side == "buy"
        assert order.quantity == 100
        assert order.limit_price == 12.50

    def test_parse_broker_order_sell(self, qmt_bridge_broker):
        """Test parsing sell order."""
        order_data = {
            "order_id": 12346,
            "stock_code": "000001.SZ",
            "order_type": 24,  # Sell
            "order_volume": 100,
            "price": 12.50,
            "order_status": 0,
            "filled_volume": 0,
        }

        order = qmt_bridge_broker._parse_broker_order(
            order_data,
            strategy_name="test_strategy",
        )

        assert order is not None
        assert order.side == "sell"

    def test_pull_broker_all_orders(self, qmt_bridge_broker, mock_qmt_client, sample_order_data):
        """Test pulling all orders."""
        mock_qmt_client.query_orders.return_value = {
            "data": [sample_order_data]
        }

        orders = qmt_bridge_broker._pull_broker_all_orders()

        assert len(orders) == 1
        assert orders[0]["order_id"] == 12345

    def test_pull_positions(self, qmt_bridge_broker, mock_qmt_client, sample_position_data):
        """Test pulling positions."""
        mock_qmt_client.query_positions.return_value = {
            "data": [sample_position_data]
        }

        positions = qmt_bridge_broker._pull_positions(strategy=MagicMock())

        assert len(positions) == 1
        assert positions[0].quantity == Decimal("1000")


# ==================== Integration Tests ====================

class TestQMTBridgeIntegration:
    """Integration tests for QMT Bridge."""

    @pytest.mark.integration
    def test_live_data_source_initialization(self):
        """Test live data source initialization (requires actual QMT Bridge)."""
        pytest.skip("Integration test - requires actual QMT Bridge connection")

    @pytest.mark.integration
    def test_live_broker_order_flow(self):
        """Test live order flow (requires actual QMT Bridge)."""
        pytest.skip("Integration test - requires actual QMT Bridge connection")


# ==================== Edge Case Tests ====================

class TestQMTBridgeEdgeCases:
    """Tests for edge cases and error handling."""

    def test_get_last_price_with_zero_price(self, qmt_bridge_data, sample_asset, mock_qmt_client):
        """Test getting last price when price is zero."""
        mock_qmt_client.get_full_tick.return_value = {
            "000001.SZ": {
                "lastPrice": 0,
                "close": 12.50,
            }
        }

        price = qmt_bridge_data.get_last_price(sample_asset)

        assert price == 12.50  # Should fall back to close price

    def test_get_quote_with_missing_data(self, qmt_bridge_data, sample_asset, mock_qmt_client):
        """Test getting quote when some fields are missing."""
        mock_qmt_client.get_full_tick.return_value = {
            "000001.SZ": {
                "lastPrice": 12.55,
                # Missing bid/ask data
            }
        }

        quote = qmt_bridge_data.get_quote(sample_asset)

        assert quote is not None
        assert quote.last == 12.55
        assert quote.bid == 0
        assert quote.ask == 0

    def test_parse_position_with_zero_quantity(self, qmt_bridge_broker):
        """Test parsing position with zero quantity."""
        position_data = {
            "stock_code": "000001.SZ",
            "volume": 0,
            "cost_price": 12.50,
        }

        position = qmt_bridge_broker._parse_broker_position(
            position_data,
            strategy="test_strategy",
        )

        assert position is not None
        assert position.quantity == Decimal("0")

    def test_parse_order_with_invalid_side(self, qmt_bridge_broker):
        """Test parsing order with invalid side - should return None."""
        order_data = {
            "order_id": 12345,
            "stock_code": "000001.SZ",
            "order_type": 99,  # Invalid
            "order_volume": 100,
            "price": 12.50,
        }

        order = qmt_bridge_broker._parse_broker_order(
            order_data,
            strategy_name="test_strategy",
        )

        # Invalid side causes Order creation to fail, returns None
        assert order is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
