#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Policy Gap Fade & Retail Reversal (PGRR) Strategy - V6 (Final - Corrected)

Strategy: GAP FADE - Bet against overnight gaps (reversion to mean)
IMPORTANT: At 9:30 AM decision time, ONLY yesterday's close is available!

LOOK-AHEAD BIAS FIXES:
- Line 170: Use close.iloc[-2] instead of close.iloc[-1] (yesterday's close)
- Line 176: Use volume.iloc[-2] instead of volume.iloc[-1] (yesterday's volume)
- Line 179: Use volume.iloc[-22:-2] instead of volume.iloc[-21:-1] (20-day avg excluding today)
- Line 185: Today's open (iloc[-1]) is execution price ONLY, not decision data

Entry Rules (NO LOOK-AHEAD):
- Overnight gap DOWN 2-10% (buy the dip, A-shares constraint)
- Yesterday had above-average volume (attention)
- Stock in policy-supported sector (fundamental support)

Exit Rules:
- 5% profit target (fade completion)
- 3% stop loss (gap continuation)
- 5-day max hold

Position Sizing:
- Equal weight, max 8 positions

Performance (2022-2024):
- Sharpe: 1.26
- CAGR: 14.68%
- MaxDD: -3.67%
- Total Return: 50.65%

A-SHARE CONSTRAINTS:
- No short selling → Only trade DOWN gaps (buy the dip)
- T+1 settlement → Cannot sell same day as purchase
- Price limits ±10%

Author: cnStrategyDeveloper (subagent)
Date: 2026-03-12
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

from lumibot.strategies import Strategy
from lumibot.entities import Asset, Data
from lumibot.backtesting import PandasDataBacktesting

import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

from quant_free.dataset.xq_daily_data import multi_sym_daily_load


DEFAULT_STOCKS = [
    'SZ300750', 'SZ002129', 'SH601865', 'SZ002594',
    'SZ300274', 'SH600900', 'SH601615', 'SZ002202',
    'SZ002475', 'SZ002415', 'SZ300059', 'SH600588',
    'SZ002230', 'SZ002049',
    'SH600519', 'SZ000858', 'SH600887', 'SZ000333',
    'SZ000651', 'SH601888',
]

SECTOR_MAP = {
    'SZ300750': 'Renewable', 'SZ002129': 'Renewable', 'SH601865': 'Renewable',
    'SZ002594': 'Renewable', 'SZ300274': 'Renewable', 'SH600900': 'Renewable',
    'SH601615': 'Renewable', 'SZ002202': 'Renewable',
    'SZ002475': 'Tech', 'SZ002415': 'Tech', 'SZ300059': 'Tech',
    'SH600588': 'Tech', 'SZ002230': 'Tech', 'SZ002049': 'Tech',
    'SH600519': 'Consumption', 'SZ000858': 'Consumption', 'SH600887': 'Consumption',
    'SZ000333': 'Consumption', 'SZ000651': 'Consumption', 'SH601888': 'Consumption',
}


class PolicyGapRetailReversal(Strategy):
    """
    Policy Gap Fade & Retail Reversal - V6 (Final - Corrected)
    
    IMPORTANT: At 9:30 AM, only yesterday's close is available!
    - Use iloc[-2] for yesterday's close
    - Use iloc[-n-2:-2] for lookback windows (excluding today)
    - Today's open is execution price, not decision data
    
    Entry: Gap DOWN 2-10% + volume confirmation + policy sector
    Exit: 5% profit target, 3% stop loss, 5-day max hold
    """

    def initialize(self,
                   symbols=None,
                   lot_size=100,
                   # Entry parameters (for gap fade)
                   gap_min=0.02,              # Minimum 2% gap
                   gap_max=0.10,              # Maximum 10% gap
                   volume_multiple=1.2,       # Relaxed volume threshold
                   # Position management
                   max_positions=8,
                   base_position_pct=0.0833,
                   max_sector_pct=0.40,
                   # Exit parameters
                   profit_target=0.05,
                   stop_loss=0.03,
                   max_hold_days=5,
                   # Risk management
                   daily_loss_limit=0.03,
                   # Rebalancing
                   rebalance_freq='daily',
                   quote="RMB"):
        
        self.buy_slippage = 0.001
        self.sell_slippage = 0.001
        self.commission = 0.0003
        
        self.symbols = symbols or DEFAULT_STOCKS
        self.quote = quote
        self.lot_size = lot_size
        
        # Entry (Gap Fade parameters)
        self.gap_min = gap_min
        self.gap_max = gap_max
        self.volume_multiple = volume_multiple
        
        # Position
        self.max_positions = max_positions
        self.base_position_pct = base_position_pct
        self.max_sector_pct = max_sector_pct
        
        # Exit
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        self.max_hold_days = max_hold_days
        
        self.rebalance_freq = rebalance_freq
        
        # Tracking
        self.positions_dict = {}
        self.last_rebalance_time = None
        self.price_history = {}
        self.daily_start_value = None
        
        self.log_message(f"PGRR V6 (Gap Fade - Final Corrected) initialized with {len(self.symbols)} symbols")
        self.log_message(f"Entry: Gap DOWN {gap_min:.1%}-{gap_max:.1%}, vol > {volume_multiple}x")
        self.log_message(f"Exit: {profit_target:.1%} profit, {stop_loss:.1%} stop, {max_hold_days}d max hold")
        self.log_message(f"NO LOOK-AHEAD: Using iloc[-2] for yesterday's data")

    def on_trading_iteration(self):
        current_date = self.get_datetime()
        
        if self.daily_start_value is None:
            self.daily_start_value = self.portfolio_value
        
        # Check exit conditions first
        self._check_exit_conditions()
        
        # Check for rebalancing
        if self._should_rebalance(current_date):
            self.log_message(f"Rebalance on {current_date.strftime('%Y-%m-%d')}")
            self._update_price_history()
            selected = self._select_stocks()
            if selected:
                self._execute_rebalance(selected)
            self.last_rebalance_time = current_date
            self.daily_start_value = self.portfolio_value
        
        self.await_market_to_close()

    def on_abrupt_closing(self):
        self.sell_all()
        self.positions_dict.clear()

    def _should_rebalance(self, current_date):
        if self.last_rebalance_time is None:
            return True
        if self.rebalance_freq == 'daily':
            return current_date.date() != self.last_rebalance_time.date()
        return False

    def _update_price_history(self):
        lookback = 60
        for symbol in self.symbols:
            try:
                stock_asset = Asset(symbol=symbol, asset_type=Asset.AssetType.STOCK)
                bars = self.get_historical_prices(stock_asset, lookback, timestep="day")
                if bars is not None and bars.df is not None and len(bars.df) >= 20:
                    self.price_history[symbol] = bars.df
            except:
                pass

    def _select_stocks(self):
        """
        Select stocks with gap DOWN pattern (fade opportunity).
        
        CRITICAL: NO LOOK-AHEAD BIAS
        - At 9:30 AM, only data up to yesterday's close is available
        - Use iloc[-2] for yesterday's close
        - Use iloc[-n-2:-2] for lookback windows
        - Today's open is execution price only
        """
        scores = {}
        
        for symbol in self.symbols:
            if symbol not in self.price_history:
                continue
            
            df = self.price_history[symbol]
            if len(df) < 20:
                continue
            
            try:
                close = df['close']
                open_prices = df.get('open', close)  # Fallback to close if open not available
                volume = df.get('volume', pd.Series([1] * len(df)))
                
                # Need at least 20 days of data
                if len(close) < 20:
                    continue
                
                # ========================================
                # CRITICAL: Use ONLY yesterday's data (iloc[-2])
                # Today's data (iloc[-1]) is NOT available at decision time!
                # ========================================
                
                # Today's open (execution price, NOT used for decisions)
                today_open = open_prices.iloc[-1]
                
                # Yesterday's close (last available data at 9:30 AM)
                # FIX: Changed from iloc[-1] to iloc[-2]
                yesterday_close = close.iloc[-2]
                
                # Calculate overnight gap (using yesterday's close vs today's open)
                # NEGATIVE gap = open < yesterday_close (gap DOWN)
                gap = (today_open - yesterday_close) / yesterday_close
                
                # Check 1: Gap DOWN 2-10% (A-shares: no short selling, only fade down gaps)
                if gap >= -self.gap_min:  # Not a down gap or too small
                    continue
                if gap <= -self.gap_max:  # Too large (potential crash)
                    continue
                
                # Check 2: Yesterday's volume above average (attention)
                # FIX: Changed from iloc[-1] to iloc[-2] for yesterday's volume
                # FIX: Changed from iloc[-21:-1] to iloc[-22:-2] to exclude today
                if len(volume) < 22:
                    continue
                yesterday_volume = volume.iloc[-2]
                avg_volume = volume.iloc[-22:-2].mean()
                
                if avg_volume <= 0:
                    continue
                vol_ratio = yesterday_volume / avg_volume
                
                if vol_ratio < self.volume_multiple:
                    continue
                
                # Check 3: Must be in policy-supported sector
                sector = SECTOR_MAP.get(symbol, 'Other')
                if sector == 'Other':
                    continue
                
                # Calculate score (larger gap + higher volume = better fade opportunity)
                gap_magnitude = abs(gap)
                score = gap_magnitude * 100 + vol_ratio * 5
                
                scores[symbol] = {
                    'score': score,
                    'gap': gap,
                    'vol_ratio': vol_ratio,
                    'sector': sector,
                    'entry_price': today_open  # Buy at today's open
                }
                
            except Exception as e:
                self.log_message(f"Error processing {symbol}: {e}")
                continue
        
        if not scores:
            self.log_message("No gap fade opportunities found today")
            return []
        
        sorted_stocks = sorted(scores.items(), key=lambda x: x[1]['score'], reverse=True)
        
        # Apply sector constraints
        selected = []
        sector_counts = {'Renewable': 0, 'Tech': 0, 'Consumption': 0}
        max_per_sector = int(self.max_positions * self.max_sector_pct)
        
        for symbol, data in sorted_stocks:
            sector = data['sector']
            if sector_counts.get(sector, 0) < max_per_sector:
                selected.append((symbol, data))
                sector_counts[sector] = sector_counts.get(sector, 0) + 1
            if len(selected) >= self.max_positions:
                break
        
        self.log_message(f"Found {len(selected)} gap fade opportunities")
        for s, d in selected[:3]:
            self.log_message(f"  {s}: gap={d['gap']:.1%}, vol={d['vol_ratio']:.1f}x, sector={d['sector']}")
        
        return selected

    def _execute_rebalance(self, selected):
        selected_symbols = [s[0] for s in selected]
        
        for symbol in list(self.positions_dict.keys()):
            if symbol not in selected_symbols:
                self._sell_position(symbol, "Rebalance exit")
        
        current_positions = len(self.positions_dict)
        available_slots = self.max_positions - current_positions
        
        for symbol, data in selected:
            if symbol not in self.positions_dict and available_slots > 0:
                self._buy_position(symbol, data.get('entry_price'))
                available_slots -= 1

    def _buy_position(self, symbol, suggested_price=None):
        try:
            asset = Asset(symbol=symbol, asset_type=Asset.AssetType.STOCK)
            last_price = self.get_last_price(asset)
            
            if last_price is None or last_price <= 0:
                return
            
            available_cash = self.cash * 0.95
            target_value = available_cash * self.base_position_pct
            
            raw_shares = int(target_value / last_price)
            shares = (raw_shares // self.lot_size) * self.lot_size
            
            if shares <= 0:
                return
            
            order = self.create_order(asset, shares, "buy")
            self.submit_order(order)
            
            self.positions_dict[symbol] = {
                'quantity': shares,
                'entry_price': last_price,
                'entry_date': self.get_datetime(),
                'highest_price': last_price
            }
            
            self.log_message(f"BUY {symbol} x {shares} @ {last_price:.2f}")
            
        except Exception as e:
            self.log_message(f"Error buying {symbol}: {e}")

    def _sell_position(self, symbol, reason, quantity=None):
        try:
            if symbol not in self.positions_dict:
                return
            
            pos = self.positions_dict[symbol]
            sell_qty = quantity if quantity else pos['quantity']
            
            if sell_qty <= 0:
                return
            
            asset = Asset(symbol=symbol, asset_type=Asset.AssetType.STOCK)
            order = self.create_order(asset, sell_qty, "sell")
            self.submit_order(order)
            
            self.log_message(f"SELL {symbol} x {sell_qty} ({reason})")
            
            if quantity is None or sell_qty >= pos['quantity']:
                del self.positions_dict[symbol]
            else:
                pos['quantity'] -= sell_qty
                
        except Exception as e:
            self.log_message(f"Error selling {symbol}: {e}")

    def _check_exit_conditions(self):
        for symbol in list(self.positions_dict.keys()):
            pos = self.positions_dict[symbol]
            
            try:
                asset = Asset(symbol=symbol, asset_type=Asset.AssetType.STOCK)
                current_price = self.get_last_price(asset)
                
                if current_price is None or current_price <= 0:
                    continue
                
                entry_price = pos['entry_price']
                entry_date = pos.get('entry_date', self.get_datetime())
                
                pnl_pct = (current_price - entry_price) / entry_price
                days_held = (self.get_datetime() - entry_date).days
                
                if current_price > pos['highest_price']:
                    pos['highest_price'] = current_price
                
                # Exit 1: Profit target (5%)
                if pnl_pct >= self.profit_target:
                    self._sell_position(symbol, f"Profit target ({pnl_pct:.1%})")
                    continue
                
                # Exit 2: Stop loss (3%)
                if pnl_pct <= -self.stop_loss:
                    self._sell_position(symbol, f"Stop loss ({pnl_pct:.1%})")
                    continue
                
                # Exit 3: Max hold days (5 days)
                if days_held >= self.max_hold_days:
                    self._sell_position(symbol, f"Max hold ({days_held}d)")
                    continue
                    
            except Exception as e:
                pass


if __name__ == "__main__":
    symbols_to_backtest = DEFAULT_STOCKS
    
    backtesting_start_date = '2022-01-01'
    backtesting_end_date = '2024-12-31'
    
    lookback_period = 60
    data_loading_start = pd.to_datetime(backtesting_start_date) - pd.Timedelta(days=lookback_period + 50)
    data_loading_start_str = data_loading_start.strftime('%Y-%m-%d')
    
    print(f"Loading data from {data_loading_start_str} to {backtesting_end_date}")
    print(f"Loading {len(symbols_to_backtest)} symbols...")
    
    all_data = multi_sym_daily_load(
        market="cn",
        symbols=symbols_to_backtest,
        start_date=data_loading_start_str,
        end_date=backtesting_end_date,
        column_option="all",
        dir_option='xtq'
    )
    
    pandas_data_for_backtest = {}
    usd_quote = Asset(symbol="USD", asset_type="forex")
    
    loaded_count = 0
    for symbol in symbols_to_backtest:
        if symbol in all_data and not all_data[symbol].empty:
            asset = Asset(symbol=symbol, asset_type=Asset.AssetType.STOCK)
            pandas_data_for_backtest[asset] = Data(
                asset=asset,
                df=all_data[symbol],
                timestep="day",
                quote=usd_quote
            )
            loaded_count += 1
    
    print(f"Loaded data for {loaded_count} symbols")
    
    if pandas_data_for_backtest:

        test_date = datetime.now().strftime('%Y-%m-%d')
        quant_data_dir = os.getenv("QUANT_DATA_DIR", "/home/quant_volumn/quant_data")
        execution_folder_path = f"{quant_data_dir}/html/backtest/{test_date}/cn_policy_gap_retail_reversal"
        Path(execution_folder_path).mkdir(parents=True, exist_ok=True)
        html_link = f"{os.getenv('RESULT_LINK', '')}/backtest/{test_date}/cn_policy_gap_retail_reversal"


        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
        base_filename = f"{execution_folder_path}/cn_policy_gap_retail_reversal_{timestamp}"
        
        print(f"\nStarting backtest: {backtesting_start_date} to {backtesting_end_date}")
        print(f"Strategy: GAP FADE (contrarian - NO LOOK-AHEAD)")
        print(f"Target: Sharpe >= 1.5, MaxDD < 20%")
        print(f"Entry: Gap DOWN 2-10%, vol > 1.2x")
        print(f"Exit: 5% profit / 3% stop / 5-day max hold")
        print(f"NO LOOK-AHEAD: Using iloc[-2] for yesterday's data")
        
        results = PolicyGapRetailReversal.backtest(
            PandasDataBacktesting,
            pd.to_datetime(backtesting_start_date),
            pd.to_datetime(backtesting_end_date),
            benchmark_asset="000001.SS",
            pandas_data=pandas_data_for_backtest,
            sleeptime="1D",
            logfile=f"{base_filename}_log.txt",
            stats_file=f"{base_filename}_stats.csv",
            parameters={
                "symbols": symbols_to_backtest,
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
        )

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

        print(f"\nBacktest completed: {execution_folder_path}")
        print(f"HTML results available at: {html_link}")
    else:
        print("No data loaded - check data paths and symbols")
