"""HARD BLOCK on order placement.

Every order-placing method on GrowwAPI is replaced at import time with a function
that raises. This is deliberately structural rather than a convention: "we simply
never call place_order" is a promise that a future refactor, a copied snippet, or an
LLM-generated helper can silently break. A raise cannot be broken silently.

This guard is applied by groww_client.py, which is the only route through which this
codebase constructs a GrowwAPI instance -- so any client that reaches application
code is already guarded.

TO EVER ENABLE LIVE TRADING you must delete this module's call site in
groww_client.py by hand. That is intentional friction: per the project plan, live
order routing also requires a broker-registered exchange Algo-ID (mandatory under
SEBI's framework since April 2026), so flipping this on is a compliance decision, not
a code convenience.
"""
from __future__ import annotations

from growwapi import GrowwAPI

BLOCKED_METHODS = (
    "place_order",
    "modify_order",
    "cancel_order",
    "create_smart_order",
    "modify_smart_order",
    "cancel_smart_order",
)


class TradingBlockedError(RuntimeError):
    """Raised when anything attempts to place, modify or cancel a real order."""


def _blocked(name: str):
    def _raise(*args, **kwargs):
        raise TradingBlockedError(
            f"GrowwAPI.{name}() is blocked. This system is research/paper-trading only "
            f"and must never place real orders. See src/algotrade/no_trade_guard.py."
        )
    _raise.__name__ = name
    _raise.__doc__ = f"BLOCKED: {name} is disabled by no_trade_guard."
    return _raise


def install() -> list[str]:
    """Patches the GrowwAPI class itself, so every instance -- however constructed --
    is guarded. Returns the methods actually blocked."""
    blocked = []
    for name in BLOCKED_METHODS:
        if hasattr(GrowwAPI, name):
            setattr(GrowwAPI, name, _blocked(name))
            blocked.append(name)
    return blocked
