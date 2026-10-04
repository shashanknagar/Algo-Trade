"""Live paper-trading loop: captures the option chain every minute, evaluates a
prediction every 5 minutes, and records every recommendation and its outcome.

NO ORDERS ARE EVER PLACED. no_trade_guard.py structurally blocks order methods on the
GrowwAPI class itself, so this cannot execute a trade even if instructed to. Every
recommendation lands in the `paper_trades` table and nowhere else.

TWO SEPARATE CADENCES, deliberately:
  * DATA capture runs every 60s. Groww's option chain is live-snapshot-only -- there
    is no historical chain API -- so anything not captured now is lost forever. This
    cadence exists to build the dataset, and it is the most valuable thing this loop
    does.
  * PREDICTION runs every 5 minutes. Note that sampling more often does NOT create
    more independent information: what buys statistical power is a shorter HOLDING
    period, not a shorter sampling interval. Fetching every minute is for data
    accumulation and for tracking open positions, not for prediction quality.

HONESTY ABOUT THE SIGNAL: the directional model recorded here has never beaten a coin
flip out of sample (35-46% vs a 40% no-edge baseline). It is logged as paper_v1 so
that its live performance can finally be measured on data it has never seen. Treat
every recommendation as an experiment being recorded, not as advice.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import date, datetime

from growwapi import GrowwAPI

from . import db, option_chain_capture, signal_generator
from .config import Config
from .logging_utils import get_logger
from .market_calendar import IST, MARKET_CLOSE, is_market_open, is_trading_day, now_ist

FETCH_INTERVAL_SECONDS = 60
PREDICT_EVERY_N_CYCLES = 5           # -> a prediction every 5 minutes
STOP_LOSS_PCT = 0.30                 # on the option premium
TARGET_PCT = STOP_LOSS_PCT * signal_generator.RISK_REWARD_RATIO   # keeps R:R at 1:1.5
TARGET_ABS_DELTA = 0.40              # strike selection: ~0.40 delta is a common balance
UNDERLYINGS = ("NIFTY", "BANKNIFTY")


@dataclass
class PaperRecommendation:
    underlying: str
    direction: str
    option_type: str
    strike_price: float
    expiry_date: date
    entry_price: float
    underlying_spot: float | None
    target_price: float
    stop_loss_price: float
    confidence_score: float
    rationale: dict

    @property
    def text(self) -> str:
        return (
            f"Buy {self.underlying} {self.strike_price:.0f} {self.option_type} @ {self.entry_price:.2f} "
            f"| target {self.target_price:.2f} | stop-loss {self.stop_loss_price:.2f}"
        )


def select_strike(chain_rows: list[dict], direction: str) -> dict | None:
    """Picks the contract whose |delta| is closest to TARGET_ABS_DELTA, using the REAL
    Greeks the paid chain provides -- unlike option_selection.py, which had to model
    them from a volatility proxy because no option data was available."""
    if direction not in ("bullish", "bearish"):
        return None
    want = "CE" if direction == "bullish" else "PE"
    candidates = [
        r for r in chain_rows
        if r.get("option_type") == want and r.get("ltp") and r.get("delta") is not None
    ]
    if not candidates:
        # Fall back to nearest-the-money when Greeks are absent.
        spot = next((r.get("underlying_spot") for r in chain_rows if r.get("underlying_spot")), None)
        typed = [r for r in chain_rows if r.get("option_type") == want and r.get("ltp")]
        if not typed or spot is None:
            return None
        return min(typed, key=lambda r: abs(r["strike_price"] - spot))
    return min(candidates, key=lambda r: abs(abs(r["delta"]) - TARGET_ABS_DELTA))


def build_recommendation(underlying: str, signal, chain_rows: list[dict],
                         expiry: date) -> PaperRecommendation | None:
    if signal.direction == "neutral":
        return None
    leg = select_strike(chain_rows, signal.direction)
    if not leg:
        return None
    entry = float(leg["ltp"])
    if entry <= 0:
        return None
    return PaperRecommendation(
        underlying=underlying,
        direction=signal.direction,
        option_type=leg["option_type"],
        strike_price=float(leg["strike_price"]),
        expiry_date=expiry,
        entry_price=entry,
        underlying_spot=leg.get("underlying_spot"),
        # Barriers sit on the PREMIUM, so they are directly evaluable from the chain
        # snapshots this loop is already capturing.
        target_price=round(entry * (1 + TARGET_PCT), 2),
        stop_loss_price=round(entry * (1 - STOP_LOSS_PCT), 2),
        confidence_score=signal.confidence_score,
        rationale={
            "factors": signal.factor_breakdown,
            "delta": leg.get("delta"),
            "iv": leg.get("iv"),
            "oi": leg.get("oi"),
            "model_caveat": "directional model unvalidated -- 35-46% out of sample vs 40% baseline",
        },
    )


def record_recommendation(conn, rec: PaperRecommendation) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO paper_trades
                (underlying, recommendation, option_type, strike_price, expiry_date,
                 direction, entry_price, underlying_spot, target_price, stop_loss_price,
                 confidence_score, rationale, model_version, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'paper_v1', 'open')
            RETURNING id
            """,
            (rec.underlying, rec.text, rec.option_type, rec.strike_price, rec.expiry_date,
             rec.direction, rec.entry_price, rec.underlying_spot, rec.target_price,
             rec.stop_loss_price, rec.confidence_score, json.dumps(rec.rationale, default=str)),
        )
        (trade_id,) = cur.fetchone()
    conn.commit()
    return trade_id


def has_open_trade(conn, underlying: str) -> bool:
    """One open paper position per underlying at a time -- both realistic and the
    thing that keeps recorded trades statistically independent."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM paper_trades WHERE underlying = %s AND status = 'open' LIMIT 1",
            (underlying,),
        )
        return cur.fetchone() is not None


def evaluate_open_trades(conn, latest_prices: dict[tuple, float], logger) -> int:
    """Resolves open paper trades against the premiums just captured."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, underlying, option_type, strike_price, expiry_date, entry_price, "
            "target_price, stop_loss_price FROM paper_trades WHERE status = 'open'"
        )
        open_rows = cur.fetchall()

    resolved = 0
    for tid, und, opt, strike, expiry, entry, target, stop in open_rows:
        price = latest_prices.get((und, expiry, float(strike), opt))
        if price is None:
            continue
        status = None
        if price >= float(target):
            status = "target_hit"
        elif price <= float(stop):
            status = "stop_hit"
        if not status:
            continue
        risk = float(entry) - float(stop)
        r_multiple = (price - float(entry)) / risk if risk else None
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE paper_trades SET status=%s, resolved_at=now(), resolved_price=%s, "
                "r_multiple=%s WHERE id=%s",
                (status, price, r_multiple, tid),
            )
        conn.commit()
        resolved += 1
        logger.info("Paper trade %s resolved: %s at %.2f (R=%.2f)", tid, status, price,
                    r_multiple if r_multiple is not None else float("nan"))
    return resolved


def run(config: Config, groww: GrowwAPI, once: bool = False) -> None:
    logger = get_logger("live_paper_loop")
    if not is_trading_day():
        logger.info("Not a trading day; nothing to do.")
        return

    expiries: dict[str, date] = {}
    for und in UNDERLYINGS:
        exp = option_chain_capture.nearest_expiry(groww, GrowwAPI.EXCHANGE_NSE, und, logger)
        if exp:
            expiries[und] = exp
            logger.info("%s: using nearest expiry %s", und, exp)
        else:
            logger.warning("%s: could not resolve an expiry -- skipping this underlying", und)

    if not expiries:
        logger.error("No expiries resolved; aborting.")
        return

    cycle = 0
    raw_logged: set[str] = set()
    last_direction: dict[str, str] = {}

    while True:
        cycle += 1
        latest_prices: dict[tuple, float] = {}

        for und, expiry in expiries.items():
            try:
                resp = groww.get_option_chain(
                    exchange=GrowwAPI.EXCHANGE_NSE, underlying=und,
                    expiry_date=expiry.isoformat(), timeout=20,
                )
            except Exception:
                logger.exception("Option chain fetch failed for %s", und)
                continue

            rows = option_chain_capture.normalize_chain(resp, und, expiry)
            if not rows:
                logger.warning("Chain for %s produced 0 normalized rows -- storing raw "
                               "payload so the shape can be corrected", und)
            # Store the raw payload once per underlying per run, for shape verification.
            raw_sample = resp if und not in raw_logged else None
            raw_logged.add(und)

            with db.get_connection(config) as conn:
                written = option_chain_capture.upsert_snapshots(conn, rows, raw_sample)
            if cycle == 1 or cycle % 15 == 0:
                logger.info("%s: captured %d chain rows (cycle %d)", und, written, cycle)

            for r in rows:
                if r.get("ltp") is not None:
                    latest_prices[(und, expiry, float(r["strike_price"]), r["option_type"])] = float(r["ltp"])

        with db.get_connection(config) as conn:
            evaluate_open_trades(conn, latest_prices, logger)

        # ---- prediction cadence -------------------------------------------
        if cycle % PREDICT_EVERY_N_CYCLES == 0:
            for und, expiry in expiries.items():
                try:
                    with db.get_connection(config) as conn:
                        signal = signal_generator.generate_signal(
                            conn, und, now_ist().date(), "intraday"
                        )
                except Exception:
                    logger.exception("Signal generation failed for %s", und)
                    continue

                prev = last_direction.get(und)
                if prev and prev != signal.direction:
                    logger.info("DIRECTIONAL SHIFT %s: %s -> %s", und, prev, signal.direction)
                last_direction[und] = signal.direction

                if signal.direction == "neutral":
                    continue
                with db.get_connection(config) as conn:
                    if has_open_trade(conn, und):
                        continue
                    chain_rows = [
                        {"underlying_spot": None, "option_type": k[3], "strike_price": k[2],
                         "ltp": v, "delta": None}
                        for k, v in latest_prices.items() if k[0] == und
                    ]
                    rec = build_recommendation(und, signal, chain_rows, expiry)
                    if rec:
                        tid = record_recommendation(conn, rec)
                        logger.info("PAPER RECOMMENDATION #%s: %s (confidence %.0f) "
                                    "[NOT A REAL ORDER]", tid, rec.text, rec.confidence_score)

        if once:
            break
        if not is_market_open():
            logger.info("Market closed; stopping loop.")
            break
        time.sleep(FETCH_INTERVAL_SECONDS)
