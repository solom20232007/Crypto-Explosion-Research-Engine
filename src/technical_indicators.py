"""Broad, causal technical-indicator feature set for pre-breakout research.

All indicator values use the current and prior candles only. The phase label is
a descriptive, rule-based annotation; it is not itself a model input.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


MODEL_TECHNICAL_COLUMNS = [
    "ret_2h", "ret_4h", "roc_5h", "roc_10h",
    "rsi_6", "rsi_21", "stoch_rsi_k", "stoch_rsi_d",
    "ema_ratio_9", "ema_ratio_21", "ema_ratio_100", "ema_ratio_200",
    "sma_ratio_100", "sma_ratio_200", "macd_line_pct", "macd_hist_slope",
    "atr_7_pct", "atr_28_pct", "bb_width_40", "bb_position_40",
    "donchian_position_20", "donchian_position_55", "breakout_20",
    "breakout_55", "adx_14", "plus_di_14", "minus_di_14",
    "cci_20", "williams_r_14", "mfi_14", "cmf_20", "obv_slope_20",
    "volume_zscore_20", "volume_ratio_5_20", "volume_ratio_50",
    "vwap_ratio_24", "keltner_position_20", "keltner_width_20",
    "body_pct", "upper_wick_pct", "lower_wick_pct", "close_location",
    "return_skew_24", "downside_volatility_24", "upside_volatility_24",
]


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    result = 100 - 100 / (1 + rs)
    result = result.mask((loss == 0) & (gain > 0), 100)
    result = result.mask((gain == 0) & (loss > 0), 0)
    return result


def add_technical_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Return causal indicators and breakout-phase annotations, row-aligned.

    Required columns: open, high, low, close, volume, quote_volume.
    """
    required = {"open", "high", "low", "close", "volume", "quote_volume"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing indicator input columns: {', '.join(missing)}")

    x = frame.copy()
    o, h, l, c = (pd.to_numeric(x[k], errors="coerce") for k in ("open", "high", "low", "close"))
    v = pd.to_numeric(x["volume"], errors="coerce").clip(lower=0)
    qv = pd.to_numeric(x["quote_volume"], errors="coerce").clip(lower=0)
    out = pd.DataFrame(index=x.index)
    ret1 = c.pct_change()
    for n in (2, 4, 5, 10, 18, 36, 168):
        out[f"ret_{n}h" if n in (2, 4) else f"roc_{n}h"] = c.pct_change(n)

    # RSI and Stochastic RSI.
    rsi6, rsi14, rsi21 = _rsi(c, 6), _rsi(c, 14), _rsi(c, 21)
    out["rsi_6"], out["rsi_21"] = rsi6, rsi21
    rsi_min, rsi_max = rsi14.rolling(14).min(), rsi14.rolling(14).max()
    stoch = (rsi14 - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan) * 100
    out["stoch_rsi_k"] = stoch
    out["stoch_rsi_d"] = stoch.rolling(3).mean()

    # Moving-average trend, alignment and MACD family.
    for n in (9, 21, 50, 100, 200):
        out[f"ema_ratio_{n}"] = c / c.ewm(span=n, adjust=False, min_periods=n).mean() - 1
    for n in (100, 200):
        out[f"sma_ratio_{n}"] = c / c.rolling(n).mean() - 1
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    out["macd_line_pct"] = macd / c.replace(0, np.nan)
    out["macd_hist_slope"] = (macd - signal).diff(3) / c.replace(0, np.nan)

    # True range / ATR.
    prev = c.shift(1)
    tr = pd.concat([(h-l), (h-prev).abs(), (l-prev).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    for n in (7, 14, 28):
        out[f"atr_{n}_pct"] = tr.rolling(n).mean() / c.replace(0, np.nan)

    # Bollinger and Donchian volatility/compression/breakout.
    for n in (20, 40):
        mid, sd = c.rolling(n).mean(), c.rolling(n).std()
        width = 4 * sd
        out[f"bb_width_{n}"] = width / mid.replace(0, np.nan)
        out[f"bb_position_{n}"] = (c - (mid - 2 * sd)) / width.replace(0, np.nan)
    for n in (20, 55):
        prior_high = h.shift(1).rolling(n).max()
        prior_low = l.shift(1).rolling(n).min()
        out[f"donchian_position_{n}"] = (c-prior_low) / (prior_high-prior_low).replace(0, np.nan)
        out[f"breakout_{n}"] = (c > prior_high).astype("int8")

    # Directional movement and ADX.
    up, down = h.diff(), -l.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    atr_for_di = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1/14, min_periods=14, adjust=False).mean() / atr_for_di.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1/14, min_periods=14, adjust=False).mean() / atr_for_di.replace(0, np.nan)
    dx = 100 * (plus_di-minus_di).abs() / (plus_di+minus_di).replace(0, np.nan)
    out["plus_di_14"], out["minus_di_14"] = plus_di, minus_di
    out["adx_14"] = dx.ewm(alpha=1/14, min_periods=14, adjust=False).mean()

    # Oscillators.
    typical = (h+l+c) / 3
    typical_mean = typical.rolling(20).mean()
    mean_dev = typical.rolling(20).apply(lambda a: np.mean(np.abs(a - np.mean(a))), raw=True)
    out["cci_20"] = (typical-typical_mean) / (0.015*mean_dev).replace(0, np.nan)
    hh14, ll14 = h.rolling(14).max(), l.rolling(14).min()
    out["williams_r_14"] = -100*(hh14-c)/(hh14-ll14).replace(0, np.nan)
    # Money Flow Index.
    raw_flow = typical * v
    flow_sign = typical.diff()
    pos_flow = raw_flow.where(flow_sign > 0, 0.0).rolling(14).sum()
    neg_flow = raw_flow.where(flow_sign < 0, 0.0).rolling(14).sum()
    mfr = pos_flow / neg_flow.replace(0, np.nan)
    out["mfi_14"] = (100 - 100/(1+mfr)).mask((neg_flow == 0) & (pos_flow > 0), 100)
    # Chaikin Money Flow.
    mfm = ((c-l)-(h-c)) / (h-l).replace(0, np.nan)
    out["cmf_20"] = (mfm.fillna(0)*v).rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)

    # Volume/flow, OBV and rolling VWAP.
    vmean20, vstd20 = v.rolling(20).mean(), v.rolling(20).std()
    out["volume_zscore_20"] = (v-vmean20) / vstd20.replace(0, np.nan)
    out["volume_ratio_5_20"] = v.rolling(5).mean() / vmean20.replace(0, np.nan)
    out["volume_ratio_50"] = v / v.rolling(50).mean().replace(0, np.nan)
    obv = (np.sign(c.diff()).fillna(0)*v).cumsum()
    out["obv_slope_20"] = obv.diff(20) / v.rolling(20).sum().replace(0, np.nan)
    vwap24 = (typical*v).rolling(24).sum() / v.rolling(24).sum().replace(0, np.nan)
    out["vwap_ratio_24"] = c / vwap24 - 1

    # Keltner channel.
    ema20 = c.ewm(span=20, adjust=False, min_periods=20).mean()
    atr_ema = tr.ewm(alpha=1/20, min_periods=20, adjust=False).mean()
    upper, lower = ema20 + 2*atr_ema, ema20 - 2*atr_ema
    out["keltner_position_20"] = (c-lower) / (upper-lower).replace(0, np.nan)
    out["keltner_width_20"] = (upper-lower) / ema20.replace(0, np.nan)

    # Candle anatomy and return-distribution context.
    candle_range = (h-l).replace(0, np.nan)
    out["body_pct"] = (c-o) / o.replace(0, np.nan)
    out["upper_wick_pct"] = (h-pd.concat([o,c],axis=1).max(axis=1)) / c.replace(0, np.nan)
    out["lower_wick_pct"] = (pd.concat([o,c],axis=1).min(axis=1)-l) / c.replace(0, np.nan)
    out["close_location"] = (c-l) / candle_range
    out["return_skew_24"] = ret1.rolling(24).skew()
    out["downside_volatility_24"] = ret1.clip(upper=0).rolling(24).std()
    out["upside_volatility_24"] = ret1.clip(lower=0).rolling(24).std()

    # Causal phase scores: no future candles are used.
    prior_high20 = h.shift(1).rolling(20).max()
    recent_high_distance = c / prior_high20.replace(0, np.nan) - 1
    compression = out["bb_width_20"] < out["bb_width_20"].rolling(100, min_periods=30).quantile(0.25)
    volume_build = out["volume_ratio_5_20"] >= 1.15
    momentum_build = (rsi14.between(48, 68)) & (out["macd_hist_slope"] > 0)
    near_resistance = recent_high_distance.between(-0.04, 0.005)
    out["prebreakout_score"] = (
        compression.fillna(False).astype(int)
        + volume_build.fillna(False).astype(int)
        + momentum_build.fillna(False).astype(int)
        + near_resistance.fillna(False).astype(int)
        + (out["plus_di_14"] > out["minus_di_14"]).fillna(False).astype(int)
    ) / 5.0
    out["early_breakout_score"] = (
        (out["breakout_20"] == 1).astype(int)
        + (out["volume_zscore_20"] >= 1.5).fillna(False).astype(int)
        + (ret1 > 0).fillna(False).astype(int)
        + (c.pct_change(4) > 0.02).fillna(False).astype(int)
        + (out["adx_14"] >= 20).fillna(False).astype(int)
    ) / 5.0
    move4, move12, move24 = c.pct_change(4), c.pct_change(12), c.pct_change(24)
    out["move_4h_pct"], out["move_12h_pct"], out["move_24h_pct"] = move4, move12, move24
    active = (move4 >= 0.05) | (move12 >= 0.10) | (move24 >= 0.15)
    early = (out["breakout_20"] == 1) & (out["early_breakout_score"] >= 0.6) & ~active
    setup = (out["prebreakout_score"] >= 0.6) & ~early & ~active
    out["breakout_phase"] = np.select([active, early, setup], ["active_explosion", "early_breakout", "prebreakout_setup"], default="normal")
    return out.replace([np.inf, -np.inf], np.nan)
