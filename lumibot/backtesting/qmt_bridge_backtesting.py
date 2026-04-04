"""QMT Bridge Backtesting Data Source for LumiBot.

This module provides a backtesting data source that automatically fetches
historical data from QMT Bridge and feeds it into the backtesting framework.

It follows the same pattern as YahooDataBacktesting but fetches data from
the QMT Bridge API for Chinese A-shares.

IMPORTANT: This module provides a factory that returns PandasDataBacktesting
instances so that strategy_executor.py recognizes them as valid pandas daily
data sources (the check uses type().__name__ which must be "PandasData" or
"PandasDataBacktesting").

Example
-------
>>> from lumibot.backtesting import QMTBridgeDataBacktesting
>>> results = MyStrategy.backtest(
...     QMTBridgeDataBacktesting,
...     backtesting_start=datetime(2022, 1, 1),
...     backtesting_end=datetime(2024, 12, 31),
...     config={
...         "host": "192.168.1.100",
...         "port": 8000,
...         "api_key": "your-key",
...         "symbols": ["000001.SZ", "600519.SH"],
...     },
... )
"""

from lumibot.backtesting.pandas_backtesting import PandasDataBacktesting
from lumibot.data_sources.qmt_bridge_data import get_qmt_symbols_historical_price


class QMTBridgeDataBacktesting(PandasDataBacktesting):
    """Factory class that creates PandasDataBacktesting instances with QMT Bridge data.

    This class overrides __new__ to return a PandasDataBacktesting instance
    instead of itself. This allows Strategy.backtest() to use the optimized
    pandas daily data processing path in strategy_executor.py.

    The class appears as "QMTBridgeDataBacktesting" for import purposes but
    the actual instance type is PandasDataBacktesting, which passes the
    type().__name__ check in _is_pandas_daily_data_source().

    Parameters expected in ``config`` dict:
        host : str
            QMT Bridge server host.
        port : int
            QMT Bridge server port.
        api_key : str
            API key for authentication.
        symbols : list[str]
            Stock symbols in QMT format (e.g. ``"000001.SZ"``).
        dividend_type : str, optional
            Dividend adjustment type. Default ``"back"``.

    Example
    -------
    >>> backtest_results = Strategy.backtest(
    ...     QMTBridgeDataBacktesting,
    ...     backtesting_start,
    ...     backtesting_end,
    ...     config={
    ...         "host": "localhost",
    ...         "port": 8083,
    ...         "api_key": "my-key",
    ...         "symbols": ["000001.SZ", "600519.SH"],
    ...     },
    ... )
    """

    def __new__(
        cls,
        datetime_start,
        datetime_end,
        config=None,
        pandas_data=None,
        **kwargs,
    ):
        """Create and return a PandasDataBacktesting instance with QMT Bridge data."""
        config = config or {}

        if pandas_data is None:
            host = config.get("host", "localhost")
            port = config.get("port", 8083)
            api_key = config.get("api_key", "")
            symbols = config.get("symbols", [])
            dividend_type = config.get("dividend_type", "front")  # Default to "front" for forward adjustment

            start_str = datetime_start.strftime("%Y-%m-%d")
            end_str = datetime_end.strftime("%Y-%m-%d")

            pandas_data = get_qmt_symbols_historical_price(
                symbols=symbols,
                start_date=start_str,
                end_date=end_str,
                host=host,
                port=port,
                api_key=api_key,
                dividend_type=dividend_type,
            )

        # Create a PandasDataBacktesting instance (not QMTBridgeDataBacktesting)
        # so that strategy_executor.py's type().__name__ check passes.
        instance = PandasDataBacktesting(
            datetime_start=datetime_start,
            datetime_end=datetime_end,
            pandas_data=pandas_data,
            **kwargs,
        )

        # Initialize the data (sets _timestep from Data objects for daily data detection)
        instance.load_data()

        return instance

    def __init__(
        self,
        datetime_start,
        datetime_end,
        config=None,
        pandas_data=None,
        **kwargs,
    ):
        """Initialize is handled by __new__ for this factory class."""
        # __new__ creates and returns a PandasDataBacktesting instance,
        # so this __init__ is never actually called on the returned object.
        pass
