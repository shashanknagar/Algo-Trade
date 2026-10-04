"""Market structure (HH/HL/LH/LL), break of structure, change of character, and
rejection-then-resolution at a level.

WHAT THIS ENCODES, and where each rule comes from:

  MARKET STRUCTURE (Dow theory, restated by the "smart money" school). Label every
  confirmed swing: a swing high is HH if above the previous swing high else LH; a
  swing low is HL if above the previous swing low else LL. Uptrend = HH + HL,
  downtrend = LH + LL, anything else = range.

  BREAK OF STRUCTURE (BOS). In an uptrend, a close above the last swing high --
  continuation. Mirror in a downtrend.

  CHANGE OF CHARACTER (CHoCH). In an uptrend, a close BELOW the last higher low --
  the protective swing -- is the earliest structural sign of reversal. Practitioners
  are explicit that a bare CHoCH fakes out often and that follow-through is needed:
  after the CHoCH, price should print a lower high and then break the low it made.
  Both the bare and the confirmed versions are produced so the value of waiting can
  be measured rather than assumed.

  REJECTION AT A LEVEL, THEN RESOLUTION. The specific question asked: an upper wick
  at resistance shows sellers defending it. Two things can happen next --
    * price closes ABOVE the level anyway: the selling was absorbed -> breakout (long)
    * price never clears it, and a later candle closes below the rejection candle's
      low: the defence held -> confirmed reversal (short)
  Wicks in the middle of a range are excluded; the literature is unanimous that only
  wicks AT a level carry information, and our own measurements agree.

Everything is built from swings that are confirmed SWING_N bars after the fact, so
nothing here is knowable before it actually was.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .price_action import SWING_N, swing_flags, wick_ratios

NEAR = 0.01            # "at" a level = within 1%
REJECT_WICK = 0.5      # wick must be >= half the candle's range
REJECT_WINDOW = 5      # a rejection within the last N bars counts as recent


def _swing_sequence(high: pd.Series, low: pd.Series, n: int = SWING_N) -> pd.DataFrame:
    """Per bar: the last confirmed swing high/low, the one before each, and labels.
    All values are ffilled from the confirmation bar, so they are point-in-time."""
    hi_f, lo_f = swing_flags(high, low, n)
    pos = pd.Series(np.arange(len(high)), index=high.index, dtype=float)

    def last_two(values, flags):
        v = values.shift(n).where(flags)
        p = (pos - n).where(flags)
        only_v, only_p = v.dropna(), p.dropna()
        return (only_v.shift(1).reindex(values.index).ffill(), v.ffill(),
                only_p.shift(1).reindex(values.index).ffill(), p.ffill())

    sh_prev, sh, sh_prev_i, sh_i = last_two(high, hi_f)
    sl_prev, sl, sl_prev_i, sl_i = last_two(low, lo_f)

    out = pd.DataFrame(index=high.index)
    out["sh"], out["sh_prev"], out["sh_i"] = sh, sh_prev, sh_i
    out["sl"], out["sl_prev"], out["sl_i"] = sl, sl_prev, sl_i
    out["sh_is_hh"] = sh > sh_prev
    out["sl_is_hl"] = sl > sl_prev
    out["new_sh"], out["new_sl"] = hi_f, lo_f
    return out


def build_structure(df: pd.DataFrame) -> pd.DataFrame:
    """All structure states and entry rules for one instrument (sorted by date)."""
    d = df.copy()
    o, h, l, c, v = d["open"], d["high"], d["low"], d["close"], d["volume"]
    s = _swing_sequence(h, l)
    vol_ok = (v / v.rolling(20).mean()) > 1.3

    # ---- trend state from the swing sequence --------------------------------
    up = s["sh_is_hh"] & s["sl_is_hl"]
    down = (~s["sh_is_hh"]) & (~s["sl_is_hl"]) & s["sh"].notna() & s["sl_prev"].notna()
    d["trend"] = np.select([up, down], ["up", "down"], default="range")
    d["trend"] = d["trend"].where(s["sh_prev"].notna() & s["sl_prev"].notna(), "range")

    def first_cross(cond: pd.Series) -> pd.Series:
        return cond & ~cond.shift(1, fill_value=False)

    # ---- BOS: continuation break of the last swing WITH the trend -----------
    d["bos_up"] = first_cross((d["trend"] == "up") & (c > s["sh"]))
    d["bos_down"] = first_cross((d["trend"] == "down") & (c < s["sl"]))

    # ---- CHoCH: break of the PROTECTIVE swing against the trend --------------
    d["choch_bear"] = first_cross((d["trend"] == "up") & (c < s["sl"]))
    d["choch_bull"] = first_cross((d["trend"] == "down") & (c > s["sh"]))

    # ---- Confirmed CHoCH: after the break, a lower high forms, then the post-
    # break low is taken out (mirror for bullish). The follow-through the
    # literature insists on before trusting a reversal.
    d["choch_bear_confirmed"] = _confirmed_reversal(d, s, c, bearish=True)
    d["choch_bull_confirmed"] = _confirmed_reversal(d, s, c, bearish=False)

    # ---- Rejection at a level, then resolution -------------------------------
    lower_w, upper_w, cpos = wick_ratios(o, h, l, c)
    res, sup = s["sh"], s["sl"]
    # Rejection candle at resistance: upper wick >= half the range, high reached the
    # level (within 1% or pierced it), close finished below it.
    rej_top = (upper_w >= REJECT_WICK) & (h >= res * (1 - NEAR)) & (c < res)
    rej_bot = (lower_w >= REJECT_WICK) & (l <= sup * (1 + NEAR)) & (c > sup)
    d["rejection_at_resistance"] = rej_top.fillna(False)
    d["rejection_at_support"] = rej_bot.fillna(False)

    recent_rej_top = rej_top.rolling(REJECT_WINDOW).max().shift(1).fillna(0).astype(bool)
    recent_rej_bot = rej_bot.rolling(REJECT_WINDOW).max().shift(1).fillna(0).astype(bool)
    # The rejection candle's own low/high, carried forward for the confirmation test.
    rej_top_low = l.where(rej_top).ffill()
    rej_bot_high = h.where(rej_bot).ffill()

    # Absorbed rejection -> breakout: sellers defended, buyers won anyway.
    d["absorbed_breakout_long"] = first_cross(recent_rej_top & (c > res))
    d["absorbed_breakdown_short"] = first_cross(recent_rej_bot & (c < sup))
    # Held rejection -> confirmed reversal: never cleared the level, and a later
    # candle closes beyond the rejection candle's extreme.
    d["confirmed_reversal_short"] = first_cross(recent_rej_top & (c < res) & (c < rej_top_low))
    d["confirmed_reversal_long"] = first_cross(recent_rej_bot & (c > sup) & (c > rej_bot_high))

    # ---- Volume-confirmed variants of every entry ---------------------------
    for col in ("bos_up", "bos_down", "choch_bear", "choch_bull", "absorbed_breakout_long",
                "absorbed_breakdown_short", "confirmed_reversal_short", "confirmed_reversal_long"):
        d[f"{col}_vol"] = d[col] & vol_ok

    # ---- Structure + trend-aligned wick: the full "read the chart" rule -----
    # Long only in an uptrend on a held rejection at support; short only in a
    # downtrend on a held rejection at resistance.
    d["trend_aligned_long"] = d["confirmed_reversal_long"] & (d["trend"] == "up")
    d["trend_aligned_short"] = d["confirmed_reversal_short"] & (d["trend"] == "down")
    return d


def _confirmed_reversal(d: pd.DataFrame, s: pd.DataFrame, close: pd.Series, bearish: bool) -> pd.Series:
    """Bearish: after a bearish CHoCH, the next confirmed swing high is a LOWER high
    and price then closes below the lowest low printed since the CHoCH. Bullish is the
    mirror. Implemented as a small state machine per bar."""
    choch = d["choch_bear"] if bearish else d["choch_bull"]
    new_swing = s["new_sh"] if bearish else s["new_sl"]
    swing_val = s["sh"] if bearish else s["sl"]
    swing_prev = s["sh_prev"] if bearish else s["sl_prev"]

    out = pd.Series(False, index=d.index)
    armed, saw_swing, extreme = False, False, np.nan
    lows, highs = d["low"].values, d["high"].values
    cv, chv, nsv, svv, spv = close.values, choch.values, new_swing.values, swing_val.values, swing_prev.values
    for i in range(len(d)):
        if chv[i]:
            armed, saw_swing = True, False
            extreme = lows[i] if bearish else highs[i]
            continue
        if not armed:
            continue
        if nsv[i]:
            ok = (svv[i] < spv[i]) if bearish else (svv[i] > spv[i])   # lower high / higher low
            if ok:
                saw_swing = True
            else:
                armed = False           # structure did not follow through; disarm
                continue
        # Test against the extreme printed on PRIOR bars, then fold today's bar in.
        # Updating first would require a close below today's own low -- impossible.
        if saw_swing and ((cv[i] < extreme) if bearish else (cv[i] > extreme)):
            out.iloc[i] = True
            armed = False
            continue
        extreme = min(extreme, lows[i]) if bearish else max(extreme, highs[i])
    return out


LONG_SIGNALS = ["bos_up", "bos_up_vol", "choch_bull", "choch_bull_vol", "choch_bull_confirmed",
                "absorbed_breakout_long", "absorbed_breakout_long_vol",
                "confirmed_reversal_long", "confirmed_reversal_long_vol", "trend_aligned_long"]
SHORT_SIGNALS = ["bos_down", "bos_down_vol", "choch_bear", "choch_bear_vol", "choch_bear_confirmed",
                 "absorbed_breakdown_short", "absorbed_breakdown_short_vol",
                 "confirmed_reversal_short", "confirmed_reversal_short_vol", "trend_aligned_short"]
