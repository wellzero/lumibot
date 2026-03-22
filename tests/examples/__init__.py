"""
QMT Bridge Examples for LumiBot

This directory contains example strategies and usage patterns for QMT Bridge
integration with LumiBot for Chinese A-shares trading.

Files:
    - qmt_bridge_unified_example.py   : Unified backtest & live trading (RECOMMENDED)
    - qmt_bridge_backtest_example.py  : Backtesting examples
    - qmt_bridge_live_example.py      : Live trading examples

Quick Start:
    # Backtesting (recommended for testing)
    python tests/examples/qmt_bridge_unified_example.py --mode backtest

    # Paper trading (simulated live trading)
    python tests/examples/qmt_bridge_unified_example.py --mode paper

    # Live trading (WARNING: uses real money!)
    python tests/examples/qmt_bridge_unified_example.py --mode live

Configuration:
    Set environment variables for live trading:
        export QMT_BRIDGE_HOST="192.168.1.100"
        export QMT_BRIDGE_PORT="8000"
        export QMT_BRIDGE_API_KEY="your-api-key"
        export QMT_BRIDGE_ACCOUNT_ID="your-account-id"

Custom Parameters:
    python tests/examples/qmt_bridge_unified_example.py \\
        --mode backtest \\
        --symbol 600519.SH \\
        --symbol 000858.SZ \\
        --short-ma 10 \\
        --long-ma 30 \\
        --stop-loss 0.03 \\
        --take-profit 0.15
"""

from .qmt_bridge_unified_example import (
    UnifiedMovingAverageStrategy,
    run_backtest as run_unified_backtest,
    run_paper_trading as run_unified_paper,
    run_live_trading as run_unified_live,
    main as unified_main,
)

from .qmt_bridge_backtest_example import (
    SimpleMovingAverageCrossover,
    RSIStrategy,
    run_backtest,
    run_rsi_backtest,
)

from .qmt_bridge_live_example import (
    LiveTradingStrategy,
    PaperTradingStrategy,
    run_live_trading,
    run_paper_trading,
)

__all__ = [
    # Unified example (RECOMMENDED)
    "UnifiedMovingAverageStrategy",
    "run_unified_backtest",
    "run_unified_paper",
    "run_unified_live",
    "unified_main",
    # Backtesting
    "SimpleMovingAverageCrossover",
    "RSIStrategy",
    "run_backtest",
    "run_rsi_backtest",
    # Live trading
    "LiveTradingStrategy",
    "PaperTradingStrategy",
    "run_live_trading",
    "run_paper_trading",
]
