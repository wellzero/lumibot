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
    # Backtest (default, set IS_BACKTESTING=true in .env)
    python combined_portfolio_strategy.py

    # Live trading (set IS_BACKTESTING=false in .env)
    python combined_portfolio_strategy.py
"""

import os
import sys
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from lumibot import LUMIBOT_SOURCE_PATH
from lumibot.strategies import Strategy
from lumibot.backtesting import PandasDataBacktesting
from lumibot.strategies.portfolio_strategy import CombinedPortfolioStrategy
from lumibot.credentials import IS_BACKTESTING

# Import existing strategies
from tests.backtest.cnshare.policy_gap_retail.cn_policy_gap_retail_reversal_qmt_bridge_data import (
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

QMT_BRIDGE_ENV_PATH = f"{Path(LUMIBOT_SOURCE_PATH).parent}/.env"
load_dotenv(QMT_BRIDGE_ENV_PATH)

STRATEGY_NAME = "combined_portfolio"
STRATEGY_VERSION = "1.0"


# ── Build Strategy Config ──────────────────────────────────────────────────────

def build_strategies_config(gap_fade_params: dict = None,
                             momentum_params: dict = None,
                             full_data: dict = None) -> list:
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


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    from lumibot.credentials import QMT_BRIDGE_CONFIG

    # Build strategy config
    strategies_config = build_strategies_config()

    # ── Backtest mode ───────────────────────────────────────────────────────
    if IS_BACKTESTING:
        backtesting_start_date = '2022-01-01'
        backtesting_end_date = '2024-12-31'

        # Calculate data loading start with lookback
        lookback_period = 100
        data_loading_start = pd.to_datetime(backtesting_start_date) - pd.Timedelta(days=lookback_period + 50)
        data_loading_start_str = data_loading_start.strftime('%Y-%m-%d')

        # Collect all symbols from all strategies
        all_symbols = set()
        for _, _, params in strategies_config:
            symbols = params.get('symbols')
            if symbols:
                all_symbols.update(symbols)

        # Load data
        logging.info("Loading historical data for %d symbols...", len(all_symbols))

        from quant_free.dataset.xq_daily_data import multi_sym_daily_load_for_lumibot
        pandas_data = multi_sym_daily_load_for_lumibot(
            market="cn", symbols=list(all_symbols),
            start_date=data_loading_start_str,
            end_date=backtesting_end_date,
            column_option="all", dir_option='xtq'
        )

        logging.info("Loaded data for %d symbols", len(pandas_data))

        # Update strategies_config with full_data for strategies that need it
        updated_config = []
        for strategy_class, strategy_id, params in strategies_config:
            params = params.copy()
            if 'full_data' in params and params['full_data'] is None:
                params['full_data'] = {symbol: df for symbol, df in pandas_data.items()}
            updated_config.append((strategy_class, strategy_id, params))

        test_date = datetime.now().strftime('%Y-%m-%d')
        quant_data_dir = os.getenv("QUANT_DATA_DIR", "/home/quant_volumn/quant_data")
        execution_folder_path = f"{quant_data_dir}/html/backtest/{test_date}/{STRATEGY_NAME}"
        Path(execution_folder_path).mkdir(parents=True, exist_ok=True)
        html_link = f"{os.getenv('RESULT_LINK', '')}/backtest/{test_date}/{STRATEGY_NAME}"

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
        base_filename = f"{execution_folder_path}/{STRATEGY_NAME}_{timestamp}"

        print("=" * 60)
        print("Combined Portfolio Strategy - BACKTEST")
        print("=" * 60)
        print(f"Symbols: {len(all_symbols)}")
        print(f"Backtest: {backtesting_start_date} to {backtesting_end_date}")
        print("=" * 60)
        print(f"Strategy: {STRATEGY_NAME} v{STRATEGY_VERSION}")
        print(f"Strategies: Gap Fade + Momentum Breakout")
        print("=" * 60)

        results = CombinedPortfolioStrategy.backtest(
            PandasDataBacktesting,
            pd.to_datetime(backtesting_start_date),
            pd.to_datetime(backtesting_end_date),
            benchmark_asset="000001.SS",
            pandas_data=pandas_data,
            budget=10000,
            sleeptime="1D",
            logfile=f"{base_filename}_log.txt",
            stats_file=f"{base_filename}_stats.csv",
            parameters={
                "strategies_config": updated_config,
                "lot_size": 100,
                "max_positions": 12,
            },
        )

        print(f"\nBacktest completed: {execution_folder_path}")
        print(f"HTML results available at: {html_link}")

        if results:
            print("\n" + "=" * 60)
            print("Backtest Results Summary")
            print("=" * 60)
            print(f"Total Return: {results.get('total_return', 'N/A')}")
            print(f"CAGR: {results.get('cagr', 'N/A')}")
            print(f"Max Drawdown: {results.get('max_drawdown', 'N/A')}")
            print(f"Sharpe Ratio: {results.get('sharpe', 'N/A')}")
            print(f"Total Trades: {results.get('total_trades', 'N/A')}")
            print("=" * 60)

    # ── Live trading mode ───────────────────────────────────────────────────
    else:
        print("=" * 60)
        print("Combined Portfolio Strategy - LIVE TRADING")
        print("=" * 60)
        print(f"Strategy: {STRATEGY_NAME} v{STRATEGY_VERSION}")
        print(f"Strategies: Gap Fade + Momentum Breakout")
        print("=" * 60)

        from lumibot.data_sources import QMTBridgeData
        from lumibot.brokers import QMTBridgeBroker
        from lumibot.traders import Trader

        data_source = QMTBridgeData(QMT_BRIDGE_CONFIG)
        broker = QMTBridgeBroker(QMT_BRIDGE_CONFIG, data_source=data_source, connect_stream=True)

        strategy = CombinedPortfolioStrategy(
            broker=broker,
            parameters={
                "strategies_config": strategies_config,
                "lot_size": 100,
                "max_positions": 12,
            },
        )

        trader = Trader(backtest=False)
        trader.add_strategy(strategy)
        print("\nStarting live trading... (Ctrl+C to stop)")
        trader.run_all()
