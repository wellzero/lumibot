"""QMT Bridge Backtesting Data Source for LumiBot.

This module provides a backtesting data source that automatically fetches
historical data from QMT Bridge and feeds it into the backtesting framework.

It follows the same pattern as YahooDataBacktesting but fetches data from
the QMT Bridge API for Chinese A-shares.

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

from lumibot.data_sources.pandas_data import PandasData
from lumibot.data_sources.qmt_bridge_data import get_qmt_symbols_historical_price


class QMTBridgeDataBacktesting(PandasData):
    """Backtesting data source for Chinese A-shares via QMT Bridge.

    Extends PandasData to auto-load historical data from a QMT Bridge server
    at initialization time. This allows ``Strategy.backtest()`` to use
    ``QMTBridgeDataBacktesting`` as the ``datasource_class`` directly.

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

    def __init__(
        self,
        datetime_start,
        datetime_end,
        config=None,
        pandas_data=None,
        **kwargs,
    ):
        config = config or {}

        if pandas_data is None:
            host = config.get("host", "localhost")
            port = config.get("port", 8083)
            api_key = config.get("api_key", "")
            symbols = config.get("symbols", [])
            dividend_type = config.get("dividend_type", "back")

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

        super().__init__(
            datetime_start=datetime_start,
            datetime_end=datetime_end,
            pandas_data=pandas_data,
            **kwargs,
        )
