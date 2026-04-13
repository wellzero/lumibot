"""QMT Bridge Data Source for LumiBot.

This module provides a data source integration for Chinese A-shares trading via QMT Bridge.
QMT Bridge is a HTTP/WebSocket client that connects to QMT (Quantitative Trading Platform)
for trading and market data in Chinese stock markets.

Supported Markets:
    - Shanghai Stock Exchange (SHSE)
    - Shenzhen Stock Exchange (SZSE)

Example:
    >>> from lumibot.data_sources import QMTBridgeData
    >>> data_source = QMTBridgeData(
    ...     host="192.168.1.100",
    ...     port=8000,
    ...     api_key="your-api-key"
    ... )
    >>> price = data_source.get_last_price(Asset("000001.SZ"))
"""

import os
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Union

import pandas as pd
import pytz

from lumibot.constants import LUMIBOT_DEFAULT_TIMEZONE
from lumibot.data_sources.data_source import DataSource
from lumibot.entities import Asset, Bars, Data, Quote
from lumibot.tools.lumibot_logger import get_logger

logger = get_logger(__name__)


def normalize_qmt_dataframe_time(df: pd.DataFrame, timezone: str = "Asia/Shanghai") -> pd.DataFrame:
    """Normalize time column and localize timezone for QMT Bridge DataFrame.

    This function handles the time column conversion from QMT Bridge data:
    - Converts "time" column from milliseconds to datetime
    - Converts "index" column to datetime
    - Sets the datetime as the DataFrame index
    - Localizes to the specified timezone if not already timezone-aware

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from QMT Bridge with either "time" or "index" column.
    timezone : str, optional
        Timezone to localize to if index is not timezone-aware.
        Default is "Asia/Shanghai" for Chinese markets.

    Returns
    -------
    pd.DataFrame
        DataFrame with datetime index localized to the specified timezone.

    Raises
    ------
    ValueError
        If neither "time" nor "index" column is found in the DataFrame.
    """
    if df is None or df.empty:
        return df

    df = df.copy()

    if "time" in df.columns:
        # QMT returns time as timestamp in milliseconds (UTC)
        df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
        df.set_index("time", inplace=True)
    elif "index" in df.columns:
        df["time"] = pd.to_datetime(df["index"], utc=True)
        df.set_index("time", inplace=True)
    elif isinstance(df.index, pd.DatetimeIndex):
        # Index is already datetime, just ensure UTC
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
    else:
        raise ValueError("DataFrame must have 'time' or 'index' column, or have a DatetimeIndex")

    # Convert from UTC to target timezone if not already in target timezone
    tz = pytz.timezone(timezone)
    if df.index.tz is not None:
        df.index = df.index.tz_convert(tz)
    else:
        df.index = df.index.tz_localize(tz)

    return df

def qmt_bridge_normalize_symbol(symbol: str) -> str:
    # Handle SHxxxxxx → xxxxxx.SH format
    if symbol.startswith("SH") and len(symbol) > 2:
        code = symbol[2:]
        return f"{code}.SH"

    # Handle SZxxxxxx → xxxxxx.SZ format
    if symbol.startswith("SZ") and len(symbol) > 2:
        code = symbol[2:]
        return f"{code}.SZ"

    # Determine exchange based on stock code
    if symbol.startswith("6"):
        # Shanghai Stock Exchange stocks start with 6
        return f"{symbol}.SH"
    elif symbol.startswith(("0", "3")):
        # Shenzhen Stock Exchange: 0xxx (main board), 3xxx (ChiNext)
        return f"{symbol}.SZ"
    elif symbol.startswith("68"):
        # Shanghai STAR Market
        return f"{symbol}.SH"
    else:
        # Default to Shenzhen
        return f"{symbol}.SZ"


def qmt_bridge_denormalize_symbol(qmt_symbol: str) -> str:
    """Convert QMT format (xxxxxx.SH / xxxxxx.SZ) to lumibot format (SHxxxxxx / SZxxxxxx).

    Parameters
    ----------
    qmt_symbol : str
        Symbol in QMT format, e.g. ``"600519.SH"``, ``"000001.SZ"``.

    Returns
    -------
    str
        Symbol in lumibot format, e.g. ``"SH600519"``, ``"SZ000001"``.
        If no ``.`` is present the input is returned unchanged.
    """
    if "." not in qmt_symbol:
        return qmt_symbol
    code, exchange = qmt_symbol.rsplit(".", 1)
    return f"{exchange}{code}"

# QMT Bridge timestep mapping
# Maps LumiBot timestep names to QMT period strings
QMT_TIMESTEP_MAPPING = [
    {
        "timestep": "minute",
        "representations": ["1m", "minute", "1minute", "min", "1min"],
    },
    {
        "timestep": "5minute",
        "representations": ["5m", "5minute", "5min"],
    },
    {
        "timestep": "15minute",
        "representations": ["15m", "15minute", "15min"],
    },
    {
        "timestep": "30minute",
        "representations": ["30m", "30minute", "30min"],
    },
    {
        "timestep": "hour",
        "representations": ["1h", "hour", "1hour"],
    },
    {
        "timestep": "day",
        "representations": ["1d", "day", "1day", "d"],
    },
    {
        "timestep": "week",
        "representations": ["1w", "week", "1week", "w"],
    },
    {
        "timestep": "month",
        "representations": ["1mon", "month", "1month", "mon"],
    },
]


class QMTBridgeData(DataSource):
    """Data source for Chinese A-shares via QMT Bridge.

    This data source connects to a QMT Bridge server to fetch market data
    for stocks traded on Shanghai and Shenzhen stock exchanges.

    Parameters
    ----------
    host : str
        The QMT Bridge server host address (e.g., "192.168.1.100").
    port : int, optional
        The QMT Bridge server port, default 8000.
    api_key : str, optional
        API key for authentication (required for trading, optional for market data).
    tzinfo : pytz.timezone, optional
        Timezone for datetime operations, defaults to Asia/Shanghai.
    dividend_type : str, optional
        Dividend adjustment type: "none", "front" (forward adjustment),
        "back" (backward adjustment), "front_ratio", "back_ratio".
        Default is "front" for backtesting consistency.
    fill_data : bool, optional
        Whether to fill missing data (e.g., suspended trading days).
        Default is True.
    **kwargs
        Additional keyword arguments passed to DataSource.

    Attributes
    ----------
    SOURCE : str
        Data source identifier ("QMT_BRIDGE").
    MIN_TIMESTEP : str
        Minimum supported timestep ("minute").
    IS_BACKTESTING_DATA_SOURCE : bool
        Whether this is a backtesting data source (False for live data).

    Example
    -------
    >>> from lumibot.data_sources import QMTBridgeData
    >>> from lumibot.entities import Asset
    >>> ds = QMTBridgeData(host="192.168.1.100", port=8000)
    >>> asset = Asset("000001.SZ")  # Ping An Bank
    >>> price = ds.get_last_price(asset)
    >>> print(f"Current price: {price}")
    """

    SOURCE = "QMT_BRIDGE"
    IS_BACKTESTING_DATA_SOURCE = False
    MIN_TIMESTEP = "minute"
    TIMESTEP_MAPPING = QMT_TIMESTEP_MAPPING
    DEFAULT_TIMEZONE = "Asia/Shanghai"

    def __init__(
        self,
        config,
        tzinfo=None,
        dividend_type: str = "front",
        fill_data: bool = True,
        **kwargs
    ):
        """Initialize QMT Bridge data source.

        Parameters
        ----------
        host : str
            QMT Bridge server host address.
        port : int, optional
            QMT Bridge server port, default 8000.
        api_key : str, optional
            API key for authentication.
        tzinfo : pytz.timezone, optional
            Timezone for datetime operations.
        dividend_type : str, optional
            Dividend adjustment type: "none", "front", "back", "front_ratio", "back_ratio".
        fill_data : bool, optional
            Whether to fill missing data for suspended trading days.
        **kwargs
            Additional keyword arguments.
        """
        # Set timezone to Asia/Shanghai for Chinese markets
        tzinfo = pytz.timezone("Asia/Shanghai")

        super().__init__(tzinfo=tzinfo, **kwargs)

        self.host = config["host"]
        self.port = config.get("port", 8000)
        self.dividend_type = dividend_type
        self.fill_data = fill_data

        # Initialize QMT client lazily (defer import to avoid dependency issues)
        self._client = None

    def _get_client(self):
        """Get or create the QMT Bridge client.

        Returns
        -------
        QMTClient
            The QMT Bridge client instance.

        Raises
        ------
        ImportError
            If qmt_bridge package is not installed.
        """
        if self._client is None:
            try:
                from qmt_bridge import QMTClient
                self._client = QMTClient(
                    host=self.host,
                    port=self.port,
                    api_key=self._api_key or ""
                )
            except ImportError as e:
                raise ImportError(
                    "qmt_bridge package is required for QMT Bridge data source. "
                    "Please install it with: pip install qmt-bridge"
                ) from e
        return self._client

    def get_datetime(self, adjust_for_delay=False):
        """Get the current datetime in the Shanghai timezone.

        This override ensures the datetime is timezone-aware for Chinese markets.

        Parameters
        ----------
        adjust_for_delay : bool, optional
            Whether to adjust for data delay, by default False.

        Returns
        -------
        datetime
            Current datetime in Asia/Shanghai timezone (tz-aware).
        """
        from datetime import datetime

        # Get current time and make it timezone-aware in Asia/Shanghai
        now = datetime.now(self.tzinfo)

        if adjust_for_delay and self._delay:
            now -= self._delay

        return now

    def _convert_timestep_to_qmt(self, timestep: str) -> str:
        """Convert LumiBot timestep to QMT period string.

        Parameters
        ----------
        timestep : str
            LumiBot timestep string (e.g., "minute", "day", "1h").

        Returns
        -------
        str
            QMT period string (e.g., "1m", "1d", "1h").
        """
        timestep_lower = timestep.lower()

        for mapping in self.TIMESTEP_MAPPING:
            if timestep_lower in [r.lower() for r in mapping["representations"]]:
                return mapping["representations"][0]

        # Default to daily if not found
        logger.warning(f"Unknown timestep '{timestep}', defaulting to '1d'")
        return "1d"

    def _normalize_symbol(self, asset: Asset) -> str:
        """Normalize asset symbol to QMT format.

        QMT expects symbols in the format "CODE.EXCHANGE" where:
        - CODE is the 6-digit stock code
        - EXCHANGE is SH (Shanghai) or SZ (Shenzhen)

        Parameters
        ----------
        asset : Asset
            The asset to normalize.

        Returns
        -------
        str
            Symbol in QMT format (e.g., "000001.SZ").
        """
        symbol = asset.symbol.upper()

        # If symbol already has exchange suffix, return as-is
        if "." in symbol:
            return symbol
        
        return qmt_bridge_normalize_symbol(symbol)

    def get_chains(self, asset: Asset, quote: Asset = None) -> dict:
        """Get option chain information for an asset.

        Note: Chinese A-shares do not have standardized equity options
        in the same way as US markets. This method returns an empty
        chain structure for compatibility.

        Parameters
        ----------
        asset : Asset
            The underlying asset.
        quote : Asset, optional
            The quote asset (not used for Chinese markets).

        Returns
        -------
        dict
            Empty option chain structure.
        """
        # Chinese A-shares don't have traditional equity options
        # Return empty structure for compatibility
        return {
            "Multiplier": "100",
            "Chains": {
                "CALL": {},
                "PUT": {},
            },
        }

    def get_historical_prices(
        self,
        asset,
        length: int,
        timestep: str = "",
        timeshift=None,
        quote=None,
        exchange=None,
        include_after_hours: bool = True,
        **kwargs
    ) -> Union[Bars, None]:
        """Get historical price bars for an asset.

        Parameters
        ----------
        asset : Asset
            The asset to get historical prices for.
        length : int
            Number of bars to retrieve.
        timestep : str, optional
            The bar timestep (e.g., "minute", "day"). Default is "day".
        timeshift : datetime, optional
            Time shift for backtesting (not used for live data).
        quote : Asset, optional
            Quote asset (not used for Chinese markets).
        exchange : str, optional
            Exchange filter (not used).
        include_after_hours : bool, optional
            Whether to include after-hours data. Default is True.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        Bars or None
            Bars object containing historical price data, or None if no data available.

        Raises
        ------
        ValueError
            If the asset type is not supported.
        """
        if asset.asset_type not in (Asset.AssetType.STOCK, "stock", "index"):
            logger.warning(f"Asset type '{asset.asset_type}' not supported by QMT Bridge")
            return None

        client = self._get_client()
        symbol = self._normalize_symbol(asset)
        qmt_period = self._convert_timestep_to_qmt(timestep or "day")

        try:
            # Calculate time range
            # timeshift can be:
            # - None: use current datetime
            # - int: number of days to shift back from now
            # - timedelta: duration to shift back from now
            # - datetime: direct end datetime
            if timeshift is None:
                end_dt = datetime.now(self.tzinfo)
            elif isinstance(timeshift, int):
                # Integer means days to shift back
                end_dt = datetime.now(self.tzinfo) - timedelta(days=timeshift)
            elif isinstance(timeshift, timedelta):
                # Timedelta to shift back
                end_dt = datetime.now(self.tzinfo) - timeshift
            elif isinstance(timeshift, datetime):
                # Direct datetime object
                end_dt = timeshift
                # Ensure timezone-aware
                if end_dt.tzinfo is None:
                    end_dt = end_dt.replace(tzinfo=self.tzinfo)
            else:
                # Unknown type, default to current time
                logger.warning(f"Unknown timeshift type: {type(timeshift)}, using current time")
                end_dt = datetime.now(self.tzinfo)

            # Get more bars than needed to ensure we have enough after filtering
            fetch_count = length * 2

            # Format dates for QMT (YYYYMMDDHHMMSS)
            end_time = end_dt.strftime("%Y%m%d%H%M%S")
            start_dt = end_dt - timedelta(days=length * 3)  # Buffer for weekends/holidays
            start_time = start_dt.strftime("%Y%m%d%H%M%S")

            # Fetch historical data using get_history_ex
            result = client.get_history_ex(
                stocks=[symbol],
                period=qmt_period,
                start_time=start_time,
                end_time=end_time,
                count=-1,  # Get all data in range
                dividend_type=self.dividend_type,
                fill_data=self.fill_data,
            )

            if not result or symbol not in result:
                logger.warning(f"No historical data returned for {symbol}")
                return None

            df = result[symbol]

            if df is None or df.empty:
                logger.warning(f"Empty DataFrame returned for {symbol}")
                return None

            # QMT Bridge returns DataFrame with columns: time, open, high, low, close, volume, etc.
            # The 'time' column is in milliseconds since epoch

            # Build normalized DataFrame with required columns
            normalized_data = {}

            # Handle time/index - use UTC for initial parsing
            if "time" in df.columns:
                # QMT returns time as timestamp in milliseconds
                normalized_data["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
            elif "index" in df.columns:
                normalized_data["time"] = pd.to_datetime(df["index"], utc=True)
            else:
                # Try to use the index if it's datetime-like
                normalized_data["time"] = pd.to_datetime(df.index, utc=True)

            # Get OHLCV columns (case-insensitive)
            for col in ["open", "high", "low", "close", "volume"]:
                col_mapping = {c.lower(): c for c in df.columns}
                if col in col_mapping:
                    normalized_data[col] = df[col_mapping[col]]
                else:
                    logger.warning(f"Missing column '{col}' in historical data for {symbol}")
                    normalized_data[col] = 0

            # Create DataFrame with normalized columns
            normalized_df = pd.DataFrame(normalized_data)

            # Set time as index and convert to target timezone
            normalized_df.set_index("time", inplace=True)
            normalized_df.index = normalized_df.index.tz_convert(self.tzinfo)

            # Sort by index (oldest first)
            normalized_df.sort_index(inplace=True)

            # Take only the requested number of bars (most recent)
            if len(normalized_df) > length:
                normalized_df = normalized_df.iloc[-length:]

            # Create Bars object
            bars = Bars(
                asset=asset,
                df=normalized_df,
                quote=quote,
                source=self.SOURCE,
            )

            return bars

        except Exception as e:
            logger.error(f"Error fetching historical prices for {symbol}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _get_pandas_freq(self, timestep: str) -> str:
        """Convert timestep to pandas frequency string.

        Parameters
        ----------
        timestep : str
            LumiBot timestep string.

        Returns
        -------
        str
            Pandas frequency string.
        """
        timestep_lower = timestep.lower()

        if timestep_lower in ["minute", "1m", "min", "1min"]:
            return "1min"
        elif timestep_lower in ["5minute", "5m", "5min"]:
            return "5min"
        elif timestep_lower in ["15minute", "15m", "15min"]:
            return "15min"
        elif timestep_lower in ["30minute", "30m", "30min"]:
            return "30min"
        elif timestep_lower in ["hour", "1h"]:
            return "1h"
        elif timestep_lower in ["day", "1d", "d"]:
            return "1D"
        elif timestep_lower in ["week", "1w", "w"]:
            return "1W"
        elif timestep_lower in ["month", "1mon", "mon"]:
            return "1M"
        else:
            return "1D"

    def get_last_price(
        self,
        asset,
        quote=None,
        exchange=None
    ) -> Union[float, Decimal, None]:
        """Get the last known price for an asset.

        Parameters
        ----------
        asset : Asset
            The asset to get the price for.
        quote : Asset, optional
            Quote asset (not used for Chinese markets).
        exchange : str, optional
            Exchange filter (not used).

        Returns
        -------
        float, Decimal, or None
            The last known price, or None if not available.
        """
        if asset.asset_type not in (Asset.AssetType.STOCK, "stock", "index"):
            return None

        client = self._get_client()
        symbol = self._normalize_symbol(asset)

        try:
            # Get real-time snapshot using get_full_tick
            tick_data = client.get_full_tick([symbol])

            if not tick_data or symbol not in tick_data:
                logger.warning(f"No tick data returned for {symbol}")
                return None

            data = tick_data[symbol]

            # Extract last price
            # QMT returns lastPrice field
            last_price = data.get("lastPrice")

            if last_price is None or last_price == 0:
                # Try alternative fields
                last_price = data.get("close", data.get("price"))

            if last_price is None:
                return None

            return float(last_price)

        except Exception as e:
            logger.error(f"Error fetching last price for {symbol}: {e}")
            return None

    def get_quote(self, asset: Asset, quote: Asset = None, exchange: str = None) -> Quote:
        """Get the latest quote for an asset.

        Parameters
        ----------
        asset : Asset
            The asset to get the quote for.
        quote : Asset, optional
            Quote asset (not used for Chinese markets).
        exchange : str, optional
            Exchange filter (not used).

        Returns
        -------
        Quote
            Quote object with bid, ask, last, and other fields.
        """
        if asset.asset_type not in (Asset.AssetType.STOCK, "stock", "index"):
            return Quote(asset=asset)

        client = self._get_client()
        symbol = self._normalize_symbol(asset)

        try:
            tick_data = client.get_full_tick([symbol])

            if not tick_data or symbol not in tick_data:
                return Quote(asset=asset)

            data = tick_data[symbol]

            # Build quote from tick data
            quote_obj = Quote(
                asset=asset,
                timestamp=datetime.now(self.tzinfo),
            )

            # Set bid/ask
            quote_obj.bid = float(data.get("bid1", 0) or 0)
            quote_obj.ask = float(data.get("ask1", 0) or 0)
            quote_obj.bid_size = int(data.get("bidQty1", 0) or 0)
            quote_obj.ask_size = int(data.get("askQty1", 0) or 0)

            # Set last price
            last_price = data.get("lastPrice")
            if last_price is not None:
                quote_obj.last = float(last_price)

            # Set volume
            quote_obj.volume = int(data.get("volume", 0) or 0)

            # Set high/low/open
            quote_obj.high = float(data.get("high", 0) or 0)
            quote_obj.low = float(data.get("low", 0) or 0)
            quote_obj.open = float(data.get("open", 0) or 0)

            return quote_obj

        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {e}")
            return Quote(asset=asset)

    def get_market_snapshot(self, assets: list) -> dict:
        """Get market snapshot for multiple assets.

        Parameters
        ----------
        assets : list
            List of Asset objects.

        Returns
        -------
        dict
            Dictionary mapping assets to their snapshot data.
        """
        if not assets:
            return {}

        client = self._get_client()

        # Normalize all symbols
        symbols = [self._normalize_symbol(asset) for asset in assets]

        try:
            snapshot = client.get_market_snapshot(symbols)

            result = {}
            for asset, symbol in zip(assets, symbols):
                if symbol in snapshot:
                    result[asset] = snapshot[symbol]

            return result

        except Exception as e:
            logger.error(f"Error fetching market snapshot: {e}")
            return {}


def get_qmt_symbols_historical_price(
    symbols: list,
    start_date: str,
    end_date: str,
    config: dict,
    dividend_type: str = "front",
    lookback_days: int = 60,
):
    """Fetch historical daily data from QMT Bridge for multiple symbols.

    Creates a :class:`Data` object for each symbol suitable for use with
    :class:`~lumibot.backtesting.PandasDataBacktesting`.

    Parameters
    ----------
    symbols : list[str]
        Stock symbols in QMT format (e.g. ``"000001.SZ"``, ``"600519.SH"``).
    start_date : str
        Backtest start date in ``'YYYY-MM-DD'`` format.
    end_date : str
        Backtest end date in ``'YYYY-MM-DD'`` format.
    config : dict
        QMT Bridge configuration dict with keys:
        - host: QMT Bridge server host
        - port: QMT Bridge server port
        - api_key: API key for authentication
    dividend_type : str, optional
        Dividend adjustment type: ``"none"``, ``"front"`` (forward),
        ``"back"`` (backward), ``"front_ratio"``, ``"back_ratio"``.
        Default is ``"front"``.
    lookback_days : int, optional
        Extra calendar days to look back before *start_date* for indicator
        warm-up.  Default is 60.

    Returns
    -------
    dict
        Mapping of :class:`~lumibot.entities.Asset` → :class:`~lumibot.entities.Data`.
    """
    fetch_start = (pd.to_datetime(start_date) - timedelta(days=lookback_days + 50)).strftime("%Y%m%d")
    fetch_end = pd.to_datetime(end_date).strftime("%Y%m%d")

    pandas_data = {}
    quote = Asset(symbol="USD", asset_type="forex")

    logger.info("Fetching data from QMT Bridge for %d symbols (%s ~ %s)", len(symbols), fetch_start, fetch_end)

    data_source = QMTBridgeData(config, dividend_type=dividend_type, fill_data=True)

    client = data_source._get_client()

    try:
        # Step 1: Download data to local storage in batches
        total_symbols = len(symbols)
        batch_api = [qmt_bridge_normalize_symbol(s) for s in symbols]
        batch_size = 10
        logger.info("Downloading data in batches of %d...", batch_size)

        for i in range(0, total_symbols, batch_size):
            batch = symbols[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_symbols + batch_size - 1) // batch_size
            logger.info("  Downloading batch %d/%d (%d symbols)...", batch_num, total_batches, len(batch))

            # Convert symbols to QMT format: SZ000001 → 000001.SZ, SH600519 → 600519.SH
            batch_api = [qmt_bridge_normalize_symbol(s) for s in batch]
            try:
                client.download_batch(
                    stocks=batch_api,
                    period="1d",
                    start_time=fetch_start,
                    end_time=fetch_end
                )
            except Exception as e:
                logger.debug("    Batch download skipped: %s", e)

        # Step 2: Load data from local storage
        logger.info("Loading historical data from local storage...")
        symbols_qmt = [qmt_bridge_normalize_symbol(s) for s in symbols]
        result = client.get_local_data(
            stocks=symbols_qmt,
            period="1d",
            start_time=fetch_start,
            end_time=fetch_end,
            dividend_type=dividend_type,
        )

        for symbol in symbols:
            try:
                asset = Asset(symbol=symbol, asset_type=Asset.AssetType.STOCK)

                qmt_symbol = qmt_bridge_normalize_symbol(symbol)
                if qmt_symbol in result and result[qmt_symbol] is not None and not result[qmt_symbol].empty:
                    df = result[qmt_symbol]

                    df = normalize_qmt_dataframe_time(df, timezone="Asia/Shanghai")

                    required_cols = ["open", "high", "low", "close", "volume"]
                    keep_cols = [c for c in required_cols if c in df.columns]
                    df = df[keep_cols]

                    if len(df) > 0:
                        pandas_data[asset] = Data(
                            asset=asset,
                            df=df,
                            timestep="day",
                            quote=quote,
                        )
                        logger.info("  Loaded %s: %d bars", symbol, len(df))
                    else:
                        logger.warning("  No data for %s", symbol)
                else:
                    logger.warning("  No data for %s", symbol)

            except Exception as e:
                logger.error("  Error processing %s: %s", symbol, e)

    except Exception as e:
        logger.error("Error fetching batch data: %s", e)
        import traceback
        traceback.print_exc()

    logger.info("Successfully loaded data for %d/%d symbols", len(pandas_data), len(symbols))
    return pandas_data
