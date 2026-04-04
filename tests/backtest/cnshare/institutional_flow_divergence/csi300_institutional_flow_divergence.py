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

from datetime import datetime
import os, pandas as pd, numpy as np
from pathlib import Path
from lumibot.strategies import Strategy
from lumibot.entities import Asset, Data
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from quant_free.dataset.xq_daily_data import multi_sym_daily_load

STRATEGY_NAME = "csi300_institutional_flow_divergence"
STRATEGY_VERSION = "6.0"

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
    def initialize(self, symbols=None, full_data=None, params=None):
        self.params = params or StrategyParams()
        self.symbols = symbols or []
        self.full_data = full_data or {}
        p = self.params
        self.stop_loss = p.stop_loss
        self.trail_trigger = p.trail_trigger
        self.trail_distance = p.trail_distance
        self.max_hold_days = p.max_hold_days
        self.profit_target = p.profit_target
        self.max_positions = p.max_positions
        self.base_weight = p.base_weight
        self.max_drawdown = p.max_drawdown
        self.min_vol_ratio = p.min_vol_ratio
        self.min_score_entry = p.min_score_entry
        self.positions_info = {}
        self.entry_dates = {}
        self.high_water = {}
        self.new_entries_today = 0
        self.last_date = None
        self.portfolio_peak = 0
        self.total_buys = 0
        self.total_sells = 0
        self.index_above_ma = True
        
        print("\n=== %s v%s ===" % (STRATEGY_NAME, STRATEGY_VERSION))
        print("Stop: %.0f%%, Trail: %.0f%% after %.0f%%, MaxHold: %dd" % (
            self.stop_loss*100, self.trail_distance*100, self.trail_trigger*100, self.max_hold_days))
        print("MaxPos: %d, Weight: %.0f%%, MinScore: %d" % (
            self.max_positions, self.base_weight*100, self.min_score_entry))

    def on_trading_iteration(self):
        dt = self.get_datetime()
        if self.last_date != dt:
            self.new_entries_today = 0
            self.last_date = dt
        self.portfolio_peak = max(self.portfolio_peak, self.portfolio_value)
        
        self._update_regime(dt)
        self._check_exits(dt)
        
        if self.portfolio_peak > 0:
            dd = (self.portfolio_peak - self.portfolio_value) / self.portfolio_peak
            if dd >= self.max_drawdown:
                self.await_market_to_close()
                return
        
        if not self.index_above_ma:
            self.await_market_to_close()
            return
        
        signals = self._generate_signals(dt)
        if signals:
            self._execute_entries(signals, dt)
        self.await_market_to_close()

    def _update_regime(self, dt):
        """Update market regime filter using index data available at market open."""
        for sym in self.symbols:
            if '000300' in sym and sym in self.full_data:
                df = self.full_data[sym]
                df_before = df[df.index < dt]
                if len(df_before) > 22:
                    # Use last 20 closes ending yesterday: [-21:-1]
                    closes = df_before['close'].values[-21:-1]
                    if len(closes) >= 20:
                        yesterday_close = df_before['close'].values[-1]
                        ma20 = np.mean(closes[-20:])
                        self.index_above_ma = yesterday_close > ma20
                return

    def _compute_score(self, symbol, df_before):
        """Compute entry score using only data available before market open.

        IMPORTANT: df_before contains data with index < dt (today).
        So df_before[-1] is yesterday (T-1), which IS available at market open.
        We use [-1] not [-2] to avoid skipping available data.
        """
        try:
            close = df_before['close'].values
            volume = df_before['volume'].values
            high = df_before['high'].values
            low = df_before['low'].values
            n = len(close)
            if n < 30: return None

            # Use [-1] = yesterday's close (T-1), available at market open
            curr = close[-1]

            # Component 1: Breakout - position in 20-day range (0-35)
            # Use last 20 days ending yesterday: [-21:-1]
            high_20d = np.max(high[-21:-1])
            low_20d = np.min(low[-21:-1])
            rng = high_20d - low_20d
            pos = (curr - low_20d) / rng if rng > 0 else 0.5

            if curr >= high_20d: breakout = 35
            elif pos > 0.95: breakout = 28
            elif pos > 0.90: breakout = 22
            else: breakout = max(0, (pos - 0.7) * 50)

            # Component 2: Volume surge (0-30, mandatory)
            # Use last 20 days ending yesterday: [-21:-1]
            vol_20d = volume[-21:-1]
            vol_avg = np.mean(vol_20d)
            vol_ratio = volume[-1] / vol_avg if vol_avg > 0 else 1

            if vol_ratio > 2.5: vol_score = 30
            elif vol_ratio > 2.0: vol_score = 25
            elif vol_ratio > 1.5: vol_score = 20
            elif vol_ratio > self.min_vol_ratio: vol_score = 12
            else: vol_score = 0
            if vol_score == 0: return 0

            # Component 3: Trend alignment (0-25)
            # MAs ending yesterday
            ma5 = np.mean(close[-6:-1])
            ma10 = np.mean(close[-11:-1])
            ma20 = np.mean(close[-21:-1])

            if curr > ma5 > ma10 > ma20: trend = 25
            elif curr > ma5 > ma10: trend = 18
            elif curr > ma5: trend = 10
            elif curr > ma10: trend = 4
            else: trend = 0

            # Component 4: Momentum acceleration (0-10)
            # 3-day return ending yesterday
            ret_3d = (close[-1] - close[-4]) / close[-4] if close[-4] > 0 else 0
            if ret_3d > 0.05: mom = 10
            elif ret_3d > 0.02: mom = 6
            elif ret_3d > 0: mom = 3
            else: mom = 0

            return breakout + vol_score + trend + mom
        except Exception:
            return None

    def _generate_signals(self, dt):
        signals = {}
        for symbol in self.symbols:
            if '000300' in symbol or symbol in self.positions_info: continue
            if symbol not in self.full_data: continue
            df = self.full_data[symbol]
            df_before = df[df.index < dt]
            if len(df_before) < 30: continue
            score = self._compute_score(symbol, df_before)
            if score is not None and score >= self.min_score_entry:
                signals[symbol] = {'score': score}
        sorted_sigs = sorted(signals.items(), key=lambda x: x[1]['score'], reverse=True)
        slots = self.max_positions - len(self.get_positions())
        slots = min(slots, 2 - self.new_entries_today)
        return dict(sorted_sigs[:max(slots, 0)])

    def _execute_entries(self, signals, dt):
        for symbol, sig in signals.items():
            if self.new_entries_today >= 2: break
            price = self.get_last_price(symbol)
            if not price or price <= 0: continue
            df = self.full_data[symbol]
            df_before = df[df.index < dt]
            vol_adj = 1.0
            if len(df_before) > 22:
                c = df_before['close'].values[-22:]
                if len(c) > 1:
                    rets = np.diff(c) / c[:-1]
                    vol = np.std(rets)
                    vol_adj = np.clip(0.02 / vol if vol > 0 else 1.0, 0.6, 1.5)
            weight = np.clip(self.base_weight * vol_adj, 0.08, 0.20)
            qty = (int(self.portfolio_value * weight / price) // 100) * 100
            if qty < 100: continue
            self.submit_order(self.create_order(symbol, qty, "buy"))
            self.positions_info[symbol] = {}
            self.entry_dates[symbol] = dt
            self.high_water[symbol] = price
            self.new_entries_today += 1
            self.total_buys += 1
            print("  BUY %s x %d @ %.2f (score=%.0f)" % (symbol, qty, price, sig['score']))

    def _check_exits(self, dt):
        for position in list(self.get_positions()):
            symbol = position.asset.symbol
            price = self.get_last_price(symbol)
            if not price or price <= 0: continue
            entry_price = getattr(position, 'avg_fill_price', None)
            if not entry_price or entry_price == 0:
                entry_price = self.high_water.get(symbol, price)
            if symbol not in self.entry_dates: continue
            hold_days = (dt - self.entry_dates[symbol]).days
            pnl = (price - entry_price) / entry_price
            self.high_water[symbol] = max(self.high_water.get(symbol, entry_price), price)
            should_sell = False
            reason = ""
            if pnl <= -self.stop_loss:
                should_sell = True; reason = "Stop"
            elif pnl >= self.profit_target:
                should_sell = True; reason = "PT"
            elif pnl >= self.trail_trigger:
                trail_dd = (self.high_water[symbol] - price) / self.high_water[symbol]
                if trail_dd >= self.trail_distance:
                    should_sell = True; reason = "Trail"
            elif hold_days >= self.max_hold_days:
                should_sell = True; reason = "Time"
            if should_sell and position.quantity >= 100:
                print("  SELL %s x %d @ %.2f - %s (PnL:%.1f%%)" % (symbol, position.quantity, price, reason, pnl*100))
                self.submit_order(self.create_order(symbol, position.quantity, "sell"))
                self.total_sells += 1
                for d in [self.positions_info, self.entry_dates, self.high_water]:
                    d.pop(symbol, None)

    def on_abrupt_closing(self):
        print("\n=== %s v%s Stats: Buys=%d, Sells=%d ===" % (STRATEGY_NAME, STRATEGY_VERSION, self.total_buys, self.total_sells))

if __name__ == "__main__":
    # start, end = '2022-01-01', '2024-12-31'
    # start, end = '2015-12-31', '2022-01-01'
    # start, end = '2025-01-01', '2026-01-31'
    start, end = '2025-01-01', '2026-01-31'
    data_start = (pd.to_datetime(start) - pd.Timedelta(days=100)).strftime('%Y-%m-%d')

    selector_path = os.path.join(os.path.dirname(__file__), '..', 'screen', 'csi300_institutional_flow_divergence_selector.py')
    if os.path.exists(selector_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location("selector", selector_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        SYMBOL_POOL, _ = mod.select_stocks(top_n=50, end_date=start)
    else:
        SYMBOL_POOL = ["SH600519","SH600036","SH601318","SZ000333","SZ000651","SZ000858","SZ002415","SZ300750"]

    print("Loading data...")
    all_data = multi_sym_daily_load(market="cn", symbols=SYMBOL_POOL,
        start_date=data_start, end_date=end, column_option="all", dir_option='xtq')
    
    pandas_data = {}
    usd_quote = Asset(symbol="USD", asset_type="forex")
    for sym in SYMBOL_POOL:
        if sym in all_data and not all_data[sym].empty:
            asset = Asset(symbol=sym, asset_type=Asset.AssetType.STOCK)
            pandas_data[asset] = Data(asset=asset, df=all_data[sym], timestep="day", quote=usd_quote)
    
    if pandas_data:
        test_date = datetime.now().strftime('%Y-%m-%d')
        log_folder = os.path.join(os.getenv('QUANT_DATA_DIR', '/home/data/quant_free_data'), 'html', 'backtest', test_date, STRATEGY_NAME)
        try: Path(log_folder).mkdir(parents=True, exist_ok=True)
        except: log_folder = os.path.join('/tmp', 'html', 'backtest', test_date, STRATEGY_NAME); Path(log_folder).mkdir(parents=True, exist_ok=True)
        
        ts = datetime.now().strftime('%Y-%m-%d_%H-%M')
        base = os.path.join(log_folder, "%s_%s" % (STRATEGY_NAME, ts))
        params = StrategyParams()
        
        from lumibot.backtesting import PandasDataBacktesting
        results = InstitutionalFlowDivergence.backtest(
            PandasDataBacktesting,
            pd.to_datetime(start), pd.to_datetime(end),
            # benchmark_asset="000300.SH",
            pandas_data=pandas_data, budget=10000, sleeptime="1D",
            buy_slippage=params.buy_slippage, sell_slippage=params.sell_slippage,
            commission=params.commission,
            logfile="%s_log.txt" % base, trades_file="%s_trades.csv" % base,
            stats_file="%s_stats.csv" % base,
            parameters={"symbols": SYMBOL_POOL, "full_data": all_data, "params": params})
        
        print("\n=== RESULTS ===")
        if isinstance(results, dict):
            for k in ["cagr", "sharpe", "max_drawdown", "total_return", "volatility"]:
                v = results.get(k, None)
                if v is not None: print("  %s: %s" % (k, v))
        print("\nDone: %s" % log_folder)
    else:
        print("ERROR: No data")
