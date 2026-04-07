#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CSI300 Institutional Flow Divergence Strategy v6 - Final Optimized Version

Alpha Source: Momentum + Volume Divergence in CSI300 large-cap stocks
Strategy Type: Breakout momentum with volume confirmation

Entry Conditions (composite score >= 55):
1. Breakout: Stock near/at 20-day high (0-35 pts)
2. Volume: Surge above 20-day average, min 1.3x (0-30 pts, mandatory)
3. Trend: MA5 > MA10 > MA20 alignment (0-25 pts)
4. Momentum: Short-term price acceleration (0-10 pts)

Exit Conditions:
- Stop Loss: 4%
- Profit Target: 20%
- Trailing Stop: 2% trail after 5% gain
- Time Stop: 8 days max hold
- Drawdown: 12% portfolio max drawdown (no new entries)

Risk Management:
- Max 6 positions, 15% base weight
- Inverse-volatility position sizing
- Index regime filter (only enter when CSI300 above MA20)
- Max 2 new entries per day

Version History:
v1: Original (Sharpe 0.8, CAGR 0.3%, 8 trades) - signals too restrictive
v2: Relaxed mean-reversion (Sharpe -0.08) - wrong direction for large caps
v3: Momentum approach (Sharpe 0.46, CAGR 12.6%) - right direction
v4: Tight momentum (Sharpe 1.30, CAGR 24%, MaxDD 9.9%) - best params
v5: Ultra-selective (Sharpe 1.11) - too few trades
v6: Refined with regime filter (Sharpe 1.33, CAGR 26%, MaxDD 8.6%) - FINAL
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from lumibot import LUMIBOT_SOURCE_PATH
from lumibot.strategies import Strategy
from lumibot.entities import Asset, Data
from lumibot.backtesting import QMTBridgeDataBacktesting
from lumibot.credentials import IS_BACKTESTING

from lumibot.data_sources.qmt_bridge_data import get_qmt_symbols_historical_price

import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# ── Mode selection ──────────────────────────────────────────────────────────

# Load QMT Bridge environment variables
QMT_BRIDGE_ENV_PATH = f"{Path(LUMIBOT_SOURCE_PATH).parent}/.env"
load_dotenv(QMT_BRIDGE_ENV_PATH)

STRATEGY_NAME = "csi300_institutional_flow_divergence"
STRATEGY_VERSION = "6.0"

# Default symbols (CSI300 large caps)
DEFAULT_STOCKS = [
    "SH600519", "SH600036", "SH601318", "SZ000333", "SZ000651",
    "SZ000858", "SZ002415", "SZ300750", "SH600887", "SH601166",
    "SH600276", "SZ000002", "SH601398", "SH601288", "SH601939",
]


class StrategyParams:
    def __init__(self):
        # Exit parameters
        self.stop_loss = 0.04
        self.trail_trigger = 0.05
        self.trail_distance = 0.02
        self.max_hold_days = 8
        self.profit_target = 0.20

        # Position sizing
        self.max_positions = 6
        self.base_weight = 0.15
        self.max_drawdown = 0.12

        # Signal parameters
        self.min_vol_ratio = 1.3
        self.min_score_entry = 55

        # Execution
        self.buy_slippage = 0.001
        self.sell_slippage = 0.001
        self.commission = 0.001


class InstitutionalFlowDivergence(Strategy):
    """
    CSI300 Institutional Flow Divergence Strategy v6

    Momentum breakout with volume confirmation for large-cap stocks.
    Uses regime filter (CSI300 above MA20) for entry timing.
    """

    def initialize(self, symbols=None, full_data=None, params=None):
        self.params = params or StrategyParams()
        self.symbols = symbols or DEFAULT_STOCKS
        self.full_data = full_data or {}
        p = self.params

        # Exit parameters
        self.stop_loss = p.stop_loss
        self.trail_trigger = p.trail_trigger
        self.trail_distance = p.trail_distance
        self.max_hold_days = p.max_hold_days
        self.profit_target = p.profit_target

        # Position sizing
        self.max_positions = p.max_positions
        self.base_weight = p.base_weight
        self.max_drawdown = p.max_drawdown

        # Signal parameters
        self.min_vol_ratio = p.min_vol_ratio
        self.min_score_entry = p.min_score_entry

        # Tracking
        self.positions_info = {}
        self.entry_dates = {}
        self.high_water = {}
        self.new_entries_today = 0
        self.last_date = None
        self.portfolio_peak = 0
        self.total_buys = 0
        self.total_sells = 0
        self.index_above_ma = True

        self.log_message(f"\n=== {STRATEGY_NAME} v{STRATEGY_VERSION} ===")
        self.log_message(f"Stop: {self.stop_loss:.0%}, Trail: {self.trail_distance:.0%} after {self.trail_trigger:.0%}, MaxHold: {self.max_hold_days}d")
        self.log_message(f"MaxPos: {self.max_positions}, Weight: {self.base_weight:.0%}, MinScore: {self.min_score_entry}")
        self.log_message(f"Symbols: {len(self.symbols)}")

    def on_trading_iteration(self):
        dt = self.get_datetime()

        self.log_message(f"\n=== Trading iteration for {dt.strftime('%Y-%m-%d')} ===")
        
        if self.last_date != dt:
            self.new_entries_today = 0
            self.last_date = dt
        self.portfolio_peak = max(self.portfolio_peak, self.portfolio_value)

        self.log_message(f"Portfolio Value: {self.portfolio_value:.2f}, Peak: {self.portfolio_peak:.2f}, Drawdown: {(self.portfolio_peak - self.portfolio_value) / self.portfolio_peak:.1%}")

        self._update_regime(dt)
        self._check_exits(dt)

        if self.portfolio_peak > 0:
            dd = (self.portfolio_peak - self.portfolio_value) / self.portfolio_peak
            if dd >= self.max_drawdown:
                self.await_market_to_close()
                return

        self.log_message(f"Index above MA: {self.index_above_ma}")
        if not self.index_above_ma:
            self.await_market_to_close()
            return

        signals = self._generate_signals(dt)

        self.log_message(f"Generated signals: {len(signals)}")

        if signals:
            self._execute_entries(signals, dt)
        self.await_market_to_close()

    def _update_regime(self, dt):
        """Update market regime filter using index data available at market open."""
        # In live mode, get historical prices from data source
        # In backtest, use pre-loaded full_data
        index_asset = Asset(symbol="SH000300", asset_type=Asset.AssetType.STOCK)
        bars = self.get_historical_prices(index_asset, 30, timestep="day", timeshift=1)
        df_before = bars.df if bars is not None else None
        
        self.log_message("Regime check for index SH000300: Data points available = {}"\
                     .format(df_before.tail(5) if df_before is not None else 0))
        
        if df_before is not None and len(df_before) > 22:
            closes = df_before['close'].values[-21:-1]
            if len(closes) >= 20:
                yesterday_close = df_before['close'].values[-1]
                ma20 = np.mean(closes[-20:])
                self.index_above_ma = yesterday_close > ma20
        return

    def _compute_score(self, symbol, df_before):
        """Compute entry score using only data available before market open."""
        try:
            close = df_before['close'].values
            volume = df_before['volume'].values
            high = df_before['high'].values
            low = df_before['low'].values
            n = len(close)
            if n < 30:
                return None

            curr = close[-1]

            # Component 1: Breakout - position in 20-day range (0-35)
            high_20d = np.max(high[-21:-1])
            low_20d = np.min(low[-21:-1])
            rng = high_20d - low_20d
            pos = (curr - low_20d) / rng if rng > 0 else 0.5

            if curr >= high_20d:
                breakout = 35
            elif pos > 0.95:
                breakout = 28
            elif pos > 0.90:
                breakout = 22
            else:
                breakout = max(0, (pos - 0.7) * 50)

            # Component 2: Volume surge (0-30, mandatory)
            vol_20d = volume[-21:-1]
            vol_avg = np.mean(vol_20d)
            vol_ratio = volume[-1] / vol_avg if vol_avg > 0 else 1

            if vol_ratio > 2.5:
                vol_score = 30
            elif vol_ratio > 2.0:
                vol_score = 25
            elif vol_ratio > 1.5:
                vol_score = 20
            elif vol_ratio > self.min_vol_ratio:
                vol_score = 12
            else:
                vol_score = 0
            if vol_score == 0:
                return 0

            # Component 3: Trend alignment (0-25)
            ma5 = np.mean(close[-6:-1])
            ma10 = np.mean(close[-11:-1])
            ma20 = np.mean(close[-21:-1])

            if curr > ma5 > ma10 > ma20:
                trend = 25
            elif curr > ma5 > ma10:
                trend = 18
            elif curr > ma5:
                trend = 10
            elif curr > ma10:
                trend = 4
            else:
                trend = 0

            # Component 4: Momentum acceleration (0-10)
            ret_3d = (close[-1] - close[-4]) / close[-4] if close[-4] > 0 else 0
            if ret_3d > 0.05:
                mom = 10
            elif ret_3d > 0.02:
                mom = 6
            elif ret_3d > 0:
                mom = 3
            else:
                mom = 0

            return breakout + vol_score + trend + mom
        except Exception:
            return None

    def _generate_signals(self, dt):
        signals = {}
        for symbol in self.symbols:
            if '000300' in symbol or symbol in self.positions_info:
                continue

            # In backtest mode, use pre-loaded full_data with time filter
            try:
                asset = Asset(symbol=symbol, asset_type=Asset.AssetType.STOCK)
                bars = self.get_historical_prices(asset, 60, timestep="day", timeshift=1)
                if bars is None or len(bars.df) < 30:
                    continue
                df_before = bars.df
            except Exception as e:
                self.log_message(f"  Error fetching {symbol}: {e}")
                continue

            if len(df_before) < 30:
                continue
            score = self._compute_score(symbol, df_before)
            if score is not None and score >= self.min_score_entry:
                signals[symbol] = {'score': score}
        sorted_sigs = sorted(signals.items(), key=lambda x: x[1]['score'], reverse=True)
        slots = self.max_positions - len(self.get_positions())
        slots = min(slots, 2 - self.new_entries_today)
        return dict(sorted_sigs[:max(slots, 0)])

    def _execute_entries(self, signals, dt):
        for symbol, sig in signals.items():
            if self.new_entries_today >= 2:
                break
            price = self.get_last_price(symbol)
            if not price or price <= 0:
                continue

            # Get historical data for volatility adjustment
            try:
                asset = Asset(symbol=symbol, asset_type=Asset.AssetType.STOCK)
                bars = self.get_historical_prices(asset, 30, timestep="day", timeshift=1)
                if bars is None:
                    continue
                df_before = bars.df
            except Exception:
                continue

            vol_adj = 1.0
            if len(df_before) > 22:
                c = df_before['close'].values[-22:]
                if len(c) > 1:
                    rets = np.diff(c) / c[:-1]
                    vol = np.std(rets)
                    vol_adj = np.clip(0.02 / vol if vol > 0 else 1.0, 0.6, 1.5)
            weight = np.clip(self.base_weight * vol_adj, 0.08, 0.20)
            qty = (int(self.portfolio_value * weight / price) // 100) * 100
            if qty < 100:
                continue
            self.submit_order(self.create_order(symbol, qty, "buy"))
            self.positions_info[symbol] = {}
            self.entry_dates[symbol] = dt
            self.high_water[symbol] = price
            self.new_entries_today += 1
            self.total_buys += 1
            self.log_message(f"  BUY {symbol} x {qty} @ {price:.2f} (score={sig['score']:.0f})")

    def _check_exits(self, dt):
        for position in list(self.get_positions()):
            symbol = position.asset.symbol
            price = self.get_last_price(symbol)
            if not price or price <= 0:
                continue
            entry_price = getattr(position, 'avg_fill_price', None)
            if not entry_price or entry_price == 0:
                entry_price = self.high_water.get(symbol, price)
            if symbol not in self.entry_dates:
                continue
            hold_days = (dt - self.entry_dates[symbol]).days
            pnl = (price - entry_price) / entry_price
            self.high_water[symbol] = max(self.high_water.get(symbol, entry_price), price)
            should_sell = False
            reason = ""
            if pnl <= -self.stop_loss:
                should_sell = True
                reason = "Stop"
            elif pnl >= self.profit_target:
                should_sell = True
                reason = "PT"
            elif pnl >= self.trail_trigger:
                trail_dd = (self.high_water[symbol] - price) / self.high_water[symbol]
                if trail_dd >= self.trail_distance:
                    should_sell = True
                    reason = "Trail"
            elif hold_days >= self.max_hold_days:
                should_sell = True
                reason = "Time"
            if should_sell and position.quantity >= 100:
                self.log_message(f"  SELL {symbol} x {position.quantity} @ {price:.2f} - {reason} (PnL:{pnl:.1%})")
                self.submit_order(self.create_order(symbol, position.quantity, "sell"))
                self.total_sells += 1
                for d in [self.positions_info, self.entry_dates, self.high_water]:
                    d.pop(symbol, None)

    def on_abrupt_closing(self):
        self.log_message(f"\n=== {STRATEGY_NAME} v{STRATEGY_VERSION} Stats: Buys={self.total_buys}, Sells={self.total_sells} ===")


def load_symbols_from_selector(end_date: str, top_n: int = 50) -> list:
    """Load symbols from selector if available, otherwise use defaults."""
    # Try multiple selector paths
    selector_paths = [
        # Try same directory first
        os.path.join(os.path.dirname(__file__), 'csi300_institutional_flow_divergence_selector.py'),
        # Try parent directory
        os.path.join(os.path.dirname(__file__), '..', 'csi300_institutional_flow_divergence_selector.py'),
        # Try quant_free_strategies location
        '/home/claude/quant_free_strategies/cn_strategies/csi300_institutional_flow_divergence/screen/csi300_institutional_flow_divergence_selector.py',
    ]

    for selector_path in selector_paths:
        if os.path.exists(selector_path):
            logging.info(f"Loading symbols from selector: {selector_path}")
            import importlib.util
            spec = importlib.util.spec_from_file_location("selector", selector_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            symbols, _ = mod.select_stocks(top_n=top_n, end_date=end_date)
            return symbols

    logging.warning(f"Selector not found, using DEFAULT_STOCKS")
    return DEFAULT_STOCKS


if __name__ == "__main__":
    # ── Common config ───────────────────────────────────────────────────────
    from lumibot.credentials import QMT_BRIDGE_CONFIG

    # ── Backtest mode ───────────────────────────────────────────────────────
    if IS_BACKTESTING:
        backtesting_start_date = '2025-01-01'
        backtesting_end_date = '2026-01-31'

        # Load symbols from selector for backtest
        symbols_to_trade = load_symbols_from_selector(end_date=backtesting_start_date, top_n=50)
        logging.info(f"Loaded {len(symbols_to_trade)} symbols from selector")

        # Add index symbol for regime filter
        index_symbol = "SH000300"
        all_symbols = symbols_to_trade + [index_symbol]

        # Load data with lookback period for indicators
        lookback_days = 100
        data_loading_start = (pd.to_datetime(backtesting_start_date) - pd.Timedelta(days=lookback_days)).strftime('%Y-%m-%d')

        # Use QMT Bridge for online data
        logging.info("Loading historical data from QMT Bridge...")
        
        # from lumibot.data_sources.qmt_bridge_data import get_qmt_symbols_historical_price
        # pandas_data = get_qmt_symbols_historical_price(
        #     symbols=all_symbols,
        #     start_date=data_loading_start,
        #     end_date=backtesting_end_date,
        #     config=QMT_BRIDGE_CONFIG,
        #     dividend_type='front'
        # )

        from quant_free.dataset.xq_daily_data import multi_sym_daily_load_for_lumibot
        pandas_data = multi_sym_daily_load_for_lumibot(market="cn", symbols=all_symbols, 
                                         start_date=data_loading_start, 
                                         end_date=backtesting_end_date, 
                                         column_option="all", dir_option='xtq')


        strategy_params = {
            "symbols": all_symbols,
            "params": StrategyParams(),
        }

        test_date = datetime.now().strftime('%Y-%m-%d')
        quant_data_dir = os.getenv("QUANT_DATA_DIR", "/home/quant_volumn/quant_data")
        execution_folder_path = f"{quant_data_dir}/html/backtest/{test_date}/{STRATEGY_NAME}"
        Path(execution_folder_path).mkdir(parents=True, exist_ok=True)
        html_link = f"{os.getenv('RESULT_LINK', '')}/backtest/{test_date}/{STRATEGY_NAME}"

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
        base_filename = f"{execution_folder_path}/{STRATEGY_NAME}_{timestamp}"

        print("=" * 60)
        print("QMT Bridge Backtest Configuration")
        print("=" * 60)
        print(f"Symbols: {len(all_symbols)} (including index {index_symbol})")
        print(f"Backtest period: {backtesting_start_date} to {backtesting_end_date}")
        print("=" * 60)
        print(f"Strategy: {STRATEGY_NAME} v{STRATEGY_VERSION}")
        print(f"Entry: Score >= 55, vol > 1.3x")
        print(f"Exit: 20% profit / 4% stop / 8-day max hold")
        print("=" * 60)

        from lumibot.backtesting import PandasDataBacktesting
        results = InstitutionalFlowDivergence.backtest(
            PandasDataBacktesting,
            pd.to_datetime(backtesting_start_date),
            pd.to_datetime(backtesting_end_date),
            benchmark_asset="000001.SS",
            pandas_data=pandas_data,
            budget=10000,  # Match reference implementation
            sleeptime="1D",
            logfile=f"{base_filename}_log.txt",
            stats_file=f"{base_filename}_stats.csv",
            parameters=strategy_params,
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
        # Load symbols from selector for live trading
        today = datetime.now().strftime('%Y-%m-%d')
        symbols_to_trade = load_symbols_from_selector(end_date=today, top_n=50)
        logging.info(f"Loaded {len(symbols_to_trade)} symbols from selector for live trading")

        # Pre-load historical data from QMT Bridge for indicators
        lookback_days = 100
        data_loading_start = (datetime.now() - pd.Timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        data_loading_end = datetime.now().strftime('%Y-%m-%d')

        logging.info("Loading historical data from QMT Bridge for live trading...")

        # Fetch data from QMT Bridge
        symbols_to_trade = symbols_to_trade + ["SH000300"]  # Ensure index is included for regime filter
        pandas_data = get_qmt_symbols_historical_price(
            symbols=symbols_to_trade,
            start_date=data_loading_start,
            end_date=data_loading_end,
            config=QMT_BRIDGE_CONFIG,
            dividend_type='front'
        )

        strategy_params = {
            "symbols": symbols_to_trade,
            "params": StrategyParams(),
        }

        print("=" * 60)
        print("QMT Bridge LIVE Trading Configuration")
        print("=" * 60)
        print(f"Symbols: {len(symbols_to_trade)} (from selector)")
        print("=" * 60)
        print(f"Strategy: {STRATEGY_NAME} v{STRATEGY_VERSION}")
        print(f"Entry: Score >= 55, vol > 1.3x")
        print(f"Exit: 20% profit / 4% stop / 8-day max hold")
        print("=" * 60)

        from lumibot.data_sources import QMTBridgeData
        from lumibot.brokers import QMTBridgeBroker
        from lumibot.traders import Trader

        # Create data source for live market data
        data_source = QMTBridgeData(
            QMT_BRIDGE_CONFIG
        )

        # Create broker for live order execution
        broker = QMTBridgeBroker(
            QMT_BRIDGE_CONFIG,
            data_source=data_source,
            connect_stream=True,
        )

        # Create strategy instance
        strategy = InstitutionalFlowDivergence(
            broker=broker,
            parameters=strategy_params,
        )

        # Run live trading
        trader = Trader(backtest=False)
        trader.add_strategy(strategy)
        print("\nStarting live trading... (Ctrl+C to stop)")
        trader.run_all()
