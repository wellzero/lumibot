#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Combined Portfolio Strategy - Example Usage

This demonstrates how to use CombinedPortfolioStrategy to combine
two existing strategies with order netting.

Strategies:
1. PolicyGapRetailReversal - Gap fade (contrarian)
2. InstitutionalFlowDivergence - Momentum breakout

Usage:
    # Backtest (default)
    python combined_portfolio_strategy.py

    # Live trading
    python combined_portfolio_strategy.py --live

    # Custom date range
    python combined_portfolio_strategy.py --start 2023-01-01 --end 2024-12-31
"""

import os
import sys
import argparse
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Tuple, Type, Dict, Any

# Add parent paths for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from lumibot.strategies import Strategy
from lumibot.backtesting import PandasDataBacktesting
from lumibot.strategies.portfolio_strategy import CombinedPortfolioStrategy

# Import existing strategies
from tests.backtest.cnshare.policy_gap_retail.cn_policy_gap_retail_reversal_qmt import (
    PolicyGapRetailReversal,
    DEFAULT_STOCKS as GAP_FADE_STOCKS,
)
from tests.backtest.cnshare.institutional_flow_divergence.csi300_institutional_flow_divergence import (
    InstitutionalFlowDivergence,
    StrategyParams as MomentumParams,
)

# ── Logger Setup ──────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

# ── Environment Setup ──────────────────────────────────────────────────────────

QMT_BRIDGE_ENV_PATH = "/home/quant_volumn/docker/data/qmt-bridge/.env"
load_dotenv(QMT_BRIDGE_ENV_PATH)


# ── Build Strategy Config ──────────────────────────────────────────────────────

def build_strategies_config(gap_fade_params: Dict = None,
                             momentum_params: Dict = None,
                             full_data: Dict = None) -> List:
    """
    Build strategy configuration combining Gap Fade + Momentum.

    Returns:
        List of (StrategyClass, strategy_id, parameters) tuples
    """
    # Default gap fade params
    if gap_fade_params is None:
        gap_fade_params = {
            "symbols": GAP_FADE_STOCKS,
            "lot_size": 100,
            "gap_min": 0.02,
            "gap_max": 0.10,
            "volume_multiple": 1.2,
            "max_positions": 8,
            "base_position_pct": 0.0833,
            "max_sector_pct": 0.40,
            "profit_target": 0.05,
            "stop_loss": 0.03,
            "max_hold_days": 5,
            "daily_loss_limit": 0.03,
            "rebalance_freq": "daily",
        }

    # Default momentum params
    if momentum_params is None:
        params = MomentumParams()
        momentum_params = {
            "symbols": None,
            "full_data": full_data or {},
            "params": params,
        }
    elif "full_data" not in momentum_params:
        momentum_params["full_data"] = full_data or {}

    return [
        (PolicyGapRetailReversal, "gap_fade", gap_fade_params),
        (InstitutionalFlowDivergence, "momentum", momentum_params),
    ]


# ── Live Trading Mode ──────────────────────────────────────────────────────────

def run_live_trading(strategies_config: List, lot_size: int = 100, max_positions: int = 12):
    """Run live trading with QMT Bridge."""
    from lumibot.data_sources import QMTBridgeData
    from lumibot.brokers import QMTBridgeBroker
    from lumibot.traders import Trader

    qmt_host = os.getenv("QMT_BRIDGE_HOST", "localhost")
    qmt_port = int(os.getenv("QMT_BRIDGE_PORT", "8083"))
    qmt_api_key = os.getenv("QMT_BRIDGE_API_KEY", "")
    qmt_account_id = os.getenv("QMT_BRIDGE_TRADING_ACCOUNT_ID", "")

    if not qmt_account_id:
        logger.error("QMT_BRIDGE_TRADING_ACCOUNT_ID is required for live trading")
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("Combined Portfolio Strategy - LIVE TRADING")
    logger.info("=" * 70)
    logger.info(f"QMT Bridge: {qmt_host}:{qmt_port}")
    logger.info(f"Account ID: {qmt_account_id}")
    logger.info("=" * 70)
    logger.info("Strategies:")
    for strategy_class, strategy_id, _ in strategies_config:
        logger.info(f"  - {strategy_id}: {strategy_class.__name__}")
    logger.info("Order Interception: ENABLED")
    logger.info("=" * 70)
    logger.info("Starting live trading... (Ctrl+C to stop)")

    # Create data source
    data_source = QMTBridgeData(
        host=qmt_host,
        port=qmt_port,
        api_key=qmt_api_key,
    )

    # Create broker
    broker = QMTBridgeBroker(
        host=qmt_host,
        port=qmt_port,
        api_key=qmt_api_key,
        account_id=qmt_account_id,
        data_source=data_source,
        connect_stream=True,
    )

    # Create strategy
    strategy = CombinedPortfolioStrategy(
        broker=broker,
        parameters={
            "strategies_config": strategies_config,
            "lot_size": lot_size,
            "max_positions": max_positions,
        },
    )

    # Run
    trader = Trader(backtest=False)
    trader.add_strategy(strategy)
    trader.run_all()


# ── Backtest Mode ──────────────────────────────────────────────────────────────

def run_backtest(strategies_config: List, lot_size: int = 100, max_positions: int = 12,
                  start_date: str = '2022-01-01', end_date: str = '2024-12-31'):
    """Run backtest with QMT Bridge data."""
    from lumibot.data_sources.qmt_bridge_data import get_qmt_symbols_historical_price

    qmt_host = os.getenv("QMT_BRIDGE_HOST", "localhost")
    qmt_port = int(os.getenv("QMT_BRIDGE_PORT", "8083"))
    qmt_api_key = os.getenv("QMT_BRIDGE_API_KEY", "")

    # Calculate data loading start with lookback
    lookback_period = 100
    data_loading_start = pd.to_datetime(start_date) - pd.Timedelta(days=lookback_period + 50)
    data_loading_start_str = data_loading_start.strftime('%Y-%m-%d')

    test_date = datetime.now().strftime('%Y-%m-%d')
    quant_data_dir = "/home/quant_volumn/quant_data"
    execution_folder_path = f"{quant_data_dir}/html/backtest/{test_date}/combined_portfolio"
    Path(execution_folder_path).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    base_filename = f"{execution_folder_path}/combined_portfolio_{timestamp}"

    logger.info("=" * 70)
    logger.info("Combined Portfolio Strategy - BACKTEST")
    logger.info("=" * 70)
    logger.info("Strategies:")
    for strategy_class, strategy_id, params in strategies_config:
        logger.info(f"  - {strategy_id}: {strategy_class.__name__}")
        if params.get('symbols'):
            logger.info(f"    Symbols: {len(params['symbols'])}")
    logger.info("Order Interception: ENABLED")
    logger.info(f"Backtest: {start_date} to {end_date}")
    logger.info("=" * 70)

    # Collect all symbols from all strategies
    all_symbols = set()
    for _, _, params in strategies_config:
        symbols = params.get('symbols')
        if symbols:
            all_symbols.update(symbols)

    # Load data
    logger.info("Loading data from QMT Bridge...")
    pandas_data = get_qmt_symbols_historical_price(
        symbols=list(all_symbols),
        start_date=data_loading_start_str,
        end_date=end_date,
        host=qmt_host,
        port=qmt_port,
        api_key=qmt_api_key,
        dividend_type='front'
    )

    if not pandas_data:
        logger.error("No data loaded from QMT Bridge")
        sys.exit(1)

    logger.info(f"Loaded data for {len(pandas_data)} symbols")

    # Update strategies_config with full_data for strategies that need it
    updated_config = []
    for strategy_class, strategy_id, params in strategies_config:
        params = params.copy()  # Don't modify original
        if 'full_data' in params and params['full_data'] is None:
            params['full_data'] = {symbol: df for symbol, df in pandas_data.items()}
        updated_config.append((strategy_class, strategy_id, params))

    # Run backtest
    results = CombinedPortfolioStrategy.backtest(
        PandasDataBacktesting,
        pd.to_datetime(start_date),
        pd.to_datetime(end_date),
        benchmark_asset="000001.SS",
        pandas_data=pandas_data,
        sleeptime="1D",
        logfile=f"{base_filename}_log.txt",
        stats_file=f"{base_filename}_stats.csv",
        parameters={
            "strategies_config": updated_config,
            "lot_size": lot_size,
            "max_positions": max_positions,
        },
    )

    logger.info(f"Backtest completed: {execution_folder_path}")

    if results:
        logger.info("=" * 70)
        logger.info("Backtest Results Summary")
        logger.info("=" * 70)
        logger.info(f"Total Return: {results.get('total_return', 'N/A')}")
        logger.info(f"CAGR: {results.get('cagr', 'N/A')}")
        logger.info(f"Max Drawdown: {results.get('max_drawdown', 'N/A')}")
        logger.info(f"Sharpe Ratio: {results.get('sharpe', 'N/A')}")
        logger.info(f"Total Trades: {results.get('total_trades', 'N/A')}")
        logger.info("=" * 70)

    return results


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    parser = argparse.ArgumentParser(
        description="Combined Portfolio Strategy with Order Interception",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run backtest (default)
  python combined_portfolio_strategy.py

  # Run live trading
  python combined_portfolio_strategy.py --live

  # Custom date range
  python combined_portfolio_strategy.py --start 2023-01-01 --end 2024-12-31
        """
    )

    # Mode selection
    parser.add_argument(
        "--live", action="store_true",
        help="Run live trading (default: backtest)"
    )

    # Date range
    parser.add_argument("--start", type=str, default="2022-01-01", help="Start date")
    parser.add_argument("--end", type=str, default="2024-12-31", help="End date")

    # Portfolio settings
    parser.add_argument("--lot-size", type=int, default=100, help="Lot size")
    parser.add_argument("--max-positions", type=int, default=12, help="Max positions")

    args = parser.parse_args()

    # Build strategy config
    strategies_config = build_strategies_config()

    # Run in appropriate mode
    if args.live:
        run_live_trading(
            strategies_config=strategies_config,
            lot_size=args.lot_size,
            max_positions=args.max_positions,
        )
    else:
        run_backtest(
            strategies_config=strategies_config,
            lot_size=args.lot_size,
            max_positions=args.max_positions,
            start_date=args.start,
            end_date=args.end,
        )
