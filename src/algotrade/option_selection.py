"""Theoretical/informational strike + Greeks suggestion layer.

IMPORTANT SCOPE LIMIT: this is NOT backed by real option market data and is NOT
backtested against real premiums (no free source for historical option prices was
found -- see README "Known unknowns" / the master plan's Open Items). It computes
Black-Scholes Greeks using:
  - live/historical SPOT price (real)
  - a REALIZED-VOLATILITY PROXY from signal_factors.realized_volatility (real
    historical price data, but NOT the market's actual implied volatility for any
    specific option contract)
  - an assumed risk-free rate (a static approximation, not fetched live)

Treat this as a rough guide to which strike is "roughly how far OTM/ATM" for a given
directional signal -- not a substitute for checking the real option chain (which
needs Groww's paid plan; see README) before actually trading anything.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

ASSUMED_RISK_FREE_RATE = 0.065  # static approximation (~RBI repo-linked short rate), not fetched live

# NSE index strike spacing conventions (approximate, may drift from current exchange
# rules -- verify against the real instrument master / option chain before trading).
STRIKE_STEP = {"NIFTY": 50, "BANKNIFTY": 100, "SENSEX": 100}


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


@dataclass
class Greeks:
    delta: float
    gamma: float
    theta: float  # per calendar day
    vega: float  # per 1% vol change


def black_scholes_greeks(
    spot: float, strike: float, days_to_expiry: float, volatility: float,
    option_type: str, risk_free_rate: float = ASSUMED_RISK_FREE_RATE,
) -> Greeks | None:
    """option_type: 'CE' (call) or 'PE' (put). Returns None if inputs are degenerate
    (zero/negative time or volatility)."""
    if days_to_expiry <= 0 or volatility <= 0 or spot <= 0 or strike <= 0:
        return None

    t = days_to_expiry / 365.0
    sqrt_t = math.sqrt(t)
    d1 = (math.log(spot / strike) + (risk_free_rate + 0.5 * volatility**2) * t) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t

    if option_type == "CE":
        delta = _norm_cdf(d1)
        theta_annual = (
            -(spot * _norm_pdf(d1) * volatility) / (2 * sqrt_t)
            - risk_free_rate * strike * math.exp(-risk_free_rate * t) * _norm_cdf(d2)
        )
    elif option_type == "PE":
        delta = _norm_cdf(d1) - 1
        theta_annual = (
            -(spot * _norm_pdf(d1) * volatility) / (2 * sqrt_t)
            + risk_free_rate * strike * math.exp(-risk_free_rate * t) * _norm_cdf(-d2)
        )
    else:
        raise ValueError(f"option_type must be 'CE' or 'PE', got {option_type!r}")

    gamma = _norm_pdf(d1) / (spot * volatility * sqrt_t)
    vega = spot * _norm_pdf(d1) * sqrt_t / 100  # per 1% (0.01) vol change
    theta_per_day = theta_annual / 365.0

    return Greeks(delta=delta, gamma=gamma, theta=theta_per_day, vega=vega)


def suggest_strike(
    spot: float, direction: str, days_to_expiry: float, volatility: float,
    target_abs_delta: float = 0.4, instrument: str = "NIFTY",
) -> dict | None:
    """Scans a strike grid around spot for the one whose |delta| is closest to
    target_abs_delta (a common heuristic: ~0.4 delta balances directional exposure
    against premium cost/theta decay). direction: 'bullish' -> CE, 'bearish' -> PE."""
    if direction not in ("bullish", "bearish"):
        return None
    option_type = "CE" if direction == "bullish" else "PE"
    step = STRIKE_STEP.get(instrument, 50)

    best = None
    best_diff = None
    for i in range(-20, 21):
        strike = round((spot + i * step) / step) * step
        greeks = black_scholes_greeks(spot, strike, days_to_expiry, volatility, option_type)
        if greeks is None:
            continue
        diff = abs(abs(greeks.delta) - target_abs_delta)
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best = {"strike": strike, "option_type": option_type, "greeks": greeks}

    return best
