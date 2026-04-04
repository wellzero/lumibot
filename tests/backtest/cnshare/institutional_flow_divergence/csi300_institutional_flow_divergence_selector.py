"""
CSI300 Institutional Flow Divergence Stock Selector
Universe: CSI300 Large-Cap A-Shares
Selection: 8-15 stocks with strongest volume-price divergence signals

Pre-Filters:
1. CSI300 constituent (large-cap)
2. 20-day avg turnover >= 200M RMB
3. Price >= 5 RMB
4. Market cap >= 30B RMB
5. Listed >= 250 trading days
6. Not ST, not suspended
"""

import pandas as pd
import numpy as np
import os
from typing import List, Dict, Tuple

DATA_DIR = "/home/data/quant_free_data"

PRE_FILTER = {
    'min_price': 5.0,
    'min_amount_20d': 200e6,     # 200M RMB daily turnover
    'min_market_cap': 30e9,      # 30B RMB
    'min_list_days': 250,
    'min_data_rows': 100,        # Need 62+ for signals
}

SIGNAL = {
    'price_window': 20,
    'weekly_window': 60,
    'vol_decline_threshold': -0.02,
    'rsi_period': 14,
    'rsi_oversold': 40,
    'reversal_window': 5,
    'vol_surge_mult': 1.8,
    'threshold_accum': 0.001,
    'threshold_momentum': 0.0005,
    'max_stocks': 50,            # Select top 50 for backtest
}

# Default CSI300 large-cap universe (hardcoded fallback)
DEFAULT_UNIVERSE = [
    "SH600000","SH600016","SH600019","SH600028","SH600029","SH600030","SH600036","SH600048",
    "SH600050","SH600061","SH600085","SH600089","SH600104","SH600109","SH600111","SH600115",
    "SH600150","SH600153","SH600160","SH600166","SH600176","SH600177","SH600183","SH600196",
    "SH600208","SH600219","SH600233","SH600271","SH600276","SH600282","SH600298","SH600299",
    "SH600309","SH600332","SH600346","SH600369","SH600383","SH600390","SH600406","SH600426",
    "SH600436","SH600438","SH600456","SH600460","SH600486","SH600489","SH600498","SH600501",
    "SH600519","SH600547","SH600570","SH600585","SH600588","SH600596","SH600600","SH600606",
    "SH600690","SH600703","SH600745","SH600809","SH600837","SH600845","SH600859","SH600867",
    "SH600886","SH600887","SH600893","SH600900","SH600905","SH600908","SH600918","SH600919",
    "SH600926","SH600941","SH601006","SH601012","SH601066","SH601088","SH601111","SH601117",
    "SH601138","SH601166","SH601225","SH601228","SH601236","SH601288","SH601318","SH601328",
    "SH601336","SH601390","SH601398","SH601601","SH601615","SH601628","SH601633","SH601668",
    "SH601669","SH601688","SH601728","SH601766","SH601788","SH601799","SH601818","SH601838",
    "SH601857","SH601877","SH601881","SH601888","SH601899","SH601901","SH601919","SH601985",
    "SH601989","SH603019","SH603087","SH603098","SH603160","SH603195","SH603259","SH603288",
    "SH603369","SH603377","SH603501","SH603569","SH603596","SH603613","SH603717","SH603799",
    "SH603833","SH603899","SH603986","SH605117",
    "SZ000001","SZ000002","SZ000063","SZ000066","SZ000069","SZ000100","SZ000157","SZ000166",
    "SZ000333","SZ000338","SZ000425","SZ000538","SZ000568","SZ000596","SZ000625","SZ000651",
    "SZ000661","SZ000708","SZ000725","SZ000768","SZ000776","SZ000786","SZ000800","SZ000858",
    "SZ000876","SZ000895","SZ000938","SZ000963","SZ000977","SZ001979","SZ002001","SZ002007",
    "SZ002008","SZ002024","SZ002027","SZ002032","SZ002049","SZ002120","SZ002142","SZ002153",
    "SZ002179","SZ002230","SZ002236","SZ002241","SZ002249","SZ002271","SZ002304","SZ002311",
    "SZ002352","SZ002371","SZ002410","SZ002415","SZ002456","SZ002460","SZ002466","SZ002475",
    "SZ002493","SZ002475","SZ002555","SZ002594","SZ002601","SZ002602","SZ002607","SZ002709",
    "SZ002714","SZ002736","SZ002812","SZ002841","SZ002916",
    "SZ300003","SZ300014","SZ300015","SZ300033","SZ300059","SZ300124","SZ300142","SZ300274",
    "SZ300347","SZ300408","SZ300413","SZ300418","SZ300433","SZ300496","SZ300676","SZ300760",
    "SZ300750","SZ300782","SZ300832","SZ301269",
]


def load_stock_data(stock_codes: List[str]) -> Dict[str, pd.DataFrame]:
    """Load daily OHLCV data from CSV files."""
    df_dict = {}
    for code in stock_codes:
        path = os.path.join(DATA_DIR, 'cn', 'equity', code, 'xq', 'daily.csv')
        try:
            if os.path.exists(path):
                df = pd.read_csv(path)
                df_dict[code] = df
        except Exception:
            continue
    return df_dict


def apply_pre_filters(df_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Apply fundamental and liquidity pre-filters.

    IMPORTANT: After _truncate_to_end_date, df.iloc[-1] IS the last available
    data point (end_date or before). This is the data available for decision
    making. Using [-1] not [-2] to use all available data.
    """
    filtered = {}
    for code, df in df_dict.items():
        if len(df) < PRE_FILTER['min_data_rows']:
            continue

        # Last row close (the most recent available data)
        last_close = df['close'].iloc[-1]
        if last_close < PRE_FILTER['min_price']:
            continue

        # 20-day avg amount ending at last available data
        amount_20d = df['amount'].iloc[-21:-1].mean() if len(df) >= 21 else df['amount'].iloc[:-1].mean()
        if amount_20d < PRE_FILTER['min_amount_20d']:
            continue

        # Market cap (last available)
        if 'market_capital' in df.columns:
            last_mcap = df['market_capital'].iloc[-1]
            if pd.notna(last_mcap) and last_mcap < PRE_FILTER['min_market_cap']:
                continue

        # Check for sufficient data
        if len(df) < PRE_FILTER['min_list_days']:
            continue

        filtered[code] = df
    return filtered


def compute_rsi(series: np.ndarray, period: int = 14) -> float:
    """Compute RSI from a price series."""
    if len(series) < period + 1:
        return 50.0
    deltas = np.diff(series[-(period+1):])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def compute_divergence_score(df: pd.DataFrame, code: str) -> Dict:
    """Compute Volume-Price Divergence Score for a single stock.

    IMPORTANT: After truncation, df.iloc[-1] is the last available data point.
    We use [-1] for current values and [-N-1:-1] for windows to use all
    available data without lookahead bias.
    """
    close = df['close'].values
    volume = df['volume'].values
    amount = df['amount'].values
    high = df['high'].values
    low = df['low'].values
    n = len(close)

    if n < 70:
        return None

    # --- VTDS: Volume Trend Divergence Score ---
    window = SIGNAL['price_window']
    pw = SIGNAL['price_window'] + 1  # Window ending at last available data
    ww = SIGNAL['weekly_window'] + 1

    if n < ww:
        return None

    # Price trend (20-day, ending at last available data point)
    close_20d = close[-pw:-1]
    x_price = np.arange(len(close_20d))
    price_slope = np.polyfit(x_price, close_20d, 1)[0]
    price_slope_norm = price_slope / np.mean(close_20d)

    # Volume trend (20-day)
    vol_20d = volume[-pw:-1]
    vol_slope = np.polyfit(np.arange(len(vol_20d)), vol_20d, 1)[0]
    vol_slope_norm = vol_slope / np.mean(vol_20d)

    # VTDS = divergence between price and volume trends
    VTDS = -price_slope_norm * vol_slope_norm

    # --- IFS: Institutional Flow Score ---
    tp_20d = (high[-pw:-1] + low[-pw:-1] + close[-pw:-1]) / 3
    vwap_20d = np.sum(tp_20d * vol_20d) / np.sum(vol_20d)
    last_close = close[-1]
    vwap_dev = (last_close - vwap_20d) / vwap_20d

    # OBV 5-day (using last 5 days of available data)
    obv_5d_sum = 0
    for i in range(5):
        idx = -(i + 1)
        if -idx - 1 >= n:
            break
        sign = 1 if close[idx] > close[idx - 1] else (-1 if close[idx] < close[idx - 1] else 0)
        obv_5d_sum += sign * volume[idx]

    IFS = np.sign(vwap_dev) * (abs(vwap_dev) * 10 + np.sign(obv_5d_sum))

    # --- WTA: Weekly Trend Alignment ---
    close_60d = close[-ww:-1]
    slope_60d = np.polyfit(np.arange(len(close_60d)), close_60d, 1)[0]
    slope_60d_norm = slope_60d / np.mean(close_60d)

    WTA = np.sign(price_slope_norm) * np.sign(slope_60d_norm)

    # --- RSI (using all available data) ---
    rsi = compute_rsi(close, SIGNAL['rsi_period'])

    # --- Reversal check (last close vs 5 days ago) ---
    reversal = close[-1] > close[-(1 + SIGNAL['reversal_window'])]

    # --- Volume surge check ---
    vol_avg_20d = np.mean(vol_20d)
    vol_surge = volume[-1] > vol_avg_20d * SIGNAL['vol_surge_mult']

    # --- Divergence Score ---
    vt_score = min(abs(VTDS) * 50, 100)
    if_score = min(abs(IFS) * 20, 30)
    trend_score = 10 if WTA > 0 else 0
    rsi_bonus = max(0, (SIGNAL['rsi_oversold'] - rsi) * 0.5)

    total_score = vt_score + if_score + trend_score + rsi_bonus

    # Signal classification
    accumulation_signal = (
        VTDS > SIGNAL['threshold_accum']
        and price_slope_norm < 0
        and vol_slope_norm < SIGNAL['vol_decline_threshold']
        and WTA >= 0
        and rsi < SIGNAL['rsi_oversold']
        and reversal
    )

    momentum_signal = (
        VTDS > SIGNAL['threshold_momentum']
        and price_slope_norm < 0
        and vol_surge
        and close[-1] > close[-2]
        and WTA >= 0
    )

    return {
        'code': code,
        'score': total_score,
        'VTDS': VTDS,
        'IFS': IFS,
        'WTA': WTA,
        'RSI': rsi,
        'price_slope_norm': price_slope_norm,
        'vol_slope_norm': vol_slope_norm,
        'vwap_dev': vwap_dev,
        'reversal': reversal,
        'vol_surge': vol_surge,
        'accumulation_signal': accumulation_signal,
        'momentum_signal': momentum_signal,
        'amount_20d': np.mean(amount[-pw:-1]),
        'market_cap': df['market_capital'].iloc[-1] if 'market_capital' in df.columns else np.nan,
        'close': close[-1],
    }


def _truncate_to_end_date(df_dict: Dict[str, pd.DataFrame], end_date: str) -> Dict[str, pd.DataFrame]:
    """Truncate all DataFrames to data on or before end_date to prevent lookahead."""
    end_dt = pd.Timestamp(end_date)
    truncated = {}
    for code, df in df_dict.items():
        if 'timestamp' in df.columns:
            dates = pd.to_datetime(df['timestamp'])
            mask = dates <= end_dt
            if mask.any():
                truncated[code] = df.loc[mask].copy().reset_index(drop=True)
        else:
            truncated[code] = df
    return truncated


def select_stocks(top_n: int = 50, end_date: str = None) -> Tuple[List[str], pd.DataFrame]:
    """Main selection function.

    Args:
        top_n: Number of stocks to select.
        end_date: Cut-off date (e.g. '2024-12-31'). Only data on or before
                  this date is used. Prevents lookahead bias in backtesting.
                  None = use all available data (live mode).
    """
    print(f"Loading data for {len(DEFAULT_UNIVERSE)} CSI300 constituents...")
    df_dict = load_stock_data(DEFAULT_UNIVERSE)
    print(f"Loaded {len(df_dict)} stocks with data")

    if end_date:
        df_dict = _truncate_to_end_date(df_dict, end_date)
        print(f"Truncated to end_date={end_date}")
    
    print("Applying pre-filters...")
    filtered = apply_pre_filters(df_dict)
    print(f"After pre-filters: {len(filtered)} stocks")
    
    print("Computing divergence scores...")
    results = []
    for code, df in filtered.items():
        try:
            score = compute_divergence_score(df, code)
            if score is not None:
                results.append(score)
        except Exception as e:
            continue
    
    if not results:
        print("WARNING: No stocks passed scoring. Returning defaults.")
        return DEFAULT_UNIVERSE[:30], pd.DataFrame()
    
    # Build dataframe
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('score', ascending=False)
    
    # Select top N
    selected_codes = results_df.head(top_n)['code'].tolist()
    
    # Summary stats
    accum_count = results_df['accumulation_signal'].sum()
    momentum_count = results_df['momentum_signal'].sum()
    
    print(f"\n{'='*60}")
    print(f"CSI300 Institutional Flow Divergence Selector Results")
    print(f"{'='*60}")
    print(f"Stocks scored: {len(results_df)}")
    print(f"Accumulation signals: {accum_count}")
    print(f"Momentum shift signals: {momentum_count}")
    print(f"Selected: top {len(selected_codes)} by divergence score")
    print(f"\nTop 15 stocks:")
    print(results_df.head(15)[['code', 'score', 'VTDS', 'RSI', 'close', 'amount_20d']].to_string(index=False))
    
    # Write report
    report_dir = os.path.dirname(os.path.abspath(__file__))
    report_path = os.path.join(report_dir, '..', 'reports', 'selector.md')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write("# Stock Selection Report: CSI300 Institutional Flow Divergence\n\n")
        f.write(f"**Date:** {end_date or 'latest'}\n")
        f.write(f"**Universe:** CSI300 Large-Cap A-Shares\n")
        f.write(f"**Stocks with data:** {len(df_dict)}\n")
        f.write(f"**After pre-filters:** {len(filtered)}\n")
        f.write(f"**Accumulation signals:** {int(accum_count)}\n")
        f.write(f"**Momentum signals:** {int(momentum_count)}\n")
        f.write(f"**Selected:** {len(selected_codes)} stocks\n\n")
        
        f.write("## Pre-Filters Applied\n\n")
        f.write(f"- Min price: {PRE_FILTER['min_price']} RMB\n")
        f.write(f"- Min 20d avg turnover: {PRE_FILTER['min_amount_20d']/1e6:.0f}M RMB\n")
        f.write(f"- Min market cap: {PRE_FILTER['min_market_cap']/1e9:.0f}B RMB\n")
        f.write(f"- Min listing days: {PRE_FILTER['min_list_days']}\n\n")
        
        f.write("## Selected Stocks\n\n")
        f.write("| # | Code | Score | VTDS | IFS | WTA | RSI | Close | Amount(20d) |\n")
        f.write("|---|------|-------|------|-----|-----|-----|-------|-------------|\n")
        for i, row in results_df.head(top_n).iterrows():
            f.write(f"| {i+1} | {row['code']} | {row['score']:.1f} | {row['VTDS']:.6f} | "
                    f"{row['IFS']:.3f} | {row['WTA']:.0f} | {row['RSI']:.1f} | {row['close']:.2f} | "
                    f"{row['amount_20d']/1e6:.0f}M |\n")
        
        f.write(f"\n## Signal Statistics\n\n")
        f.write(f"- Mean score: {results_df['score'].mean():.1f}\n")
        f.write(f"- Median score: {results_df['score'].median():.1f}\n")
        f.write(f"- Mean RSI: {results_df['RSI'].mean():.1f}\n")
        f.write(f"- Mean VTDS: {results_df['VTDS'].mean():.6f}\n\n")
        
        if accum_count > 0:
            accum_stocks = results_df[results_df['accumulation_signal'] == True]['code'].tolist()
            f.write(f"## Active Accumulation Signals\n\n")
            f.write(f"Stocks with active accumulation divergence: {', '.join(accum_stocks)}\n\n")
        
        if momentum_count > 0:
            mom_stocks = results_df[results_df['momentum_signal'] == True]['code'].tolist()
            f.write(f"## Active Momentum Shift Signals\n\n")
            f.write(f"Stocks with active momentum shift: {', '.join(mom_stocks)}\n")
    
    print(f"\nReport written to: {report_path}")
    return selected_codes, results_df


if __name__ == "__main__":
    codes, df = select_stocks()
    print(f"\n\nSelected {len(codes)} stocks for backtest:")
    print(codes)
