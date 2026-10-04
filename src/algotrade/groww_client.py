"""Groww API authentication.

Groww's SDK supports two distinct key types (confirmed against the installed
growwapi 1.5.0 source -- GrowwAPI.get_access_token(api_key, totp=None, secret=None)):

- **TOTP-type key**: GROWW_API_KEY is a short TOTP token, and a rotating 6-digit code
  (derived via pyotp from a base32 GROWW_API_SECRET) is required on every call.
- **Approval-type key**: GROWW_API_KEY is a long-lived bearer token (a JWT -- starts
  with "eyJ..."), and GROWW_API_SECRET is passed straight through to the token
  endpoint with no TOTP/pyotp involved. This is the type this account actually has.

Which flow applies is detected automatically from the shape of GROWW_API_SECRET (valid
base32 -> TOTP flow; anything else -> approval flow) so this doesn't need a separate
config flag. Both key/secret values come from OS environment variables -- see
config.py -- never hardcoded here.
"""
from __future__ import annotations

import re

import pyotp
from growwapi import GrowwAPI, GrowwFeed

from . import no_trade_guard
from .config import Config

# Applied at import: order placement raises rather than executing. This system
# is paper-trading only -- see no_trade_guard.py.
_BLOCKED_METHODS = no_trade_guard.install()

_BASE32_RE = re.compile(r"^[A-Z2-7]+=*$")


def _is_base32(value: str) -> bool:
    return bool(_BASE32_RE.fullmatch(value.strip().upper())) if value.strip() else False


def resolve_access_token(api_key: str, secret: str) -> str:
    """The auth-flow-detection logic, independent of Config -- so callers that only
    have the raw key/secret (e.g. scripts/probe_api_capabilities.py, which
    deliberately doesn't require POSTGRES_* like the rest of the app does) don't need
    to duplicate it."""
    if _is_base32(secret):
        totp = pyotp.TOTP(secret).now()
        return GrowwAPI.get_access_token(api_key=api_key, totp=totp)
    return GrowwAPI.get_access_token(api_key=api_key, secret=secret)


def get_access_token(config: Config) -> str:
    return resolve_access_token(config.groww_api_key, config.groww_api_secret)


def get_api_client(config: Config) -> GrowwAPI:
    return GrowwAPI(get_access_token(config))


def get_feed_client(config: Config, api_client: GrowwAPI | None = None) -> GrowwFeed:
    """GrowwFeed wraps a GrowwAPI instance (not a raw token) -- confirmed against the
    installed SDK's source, since the published docs/README disagree with each other
    and with the actual 1.5.0 code on this point."""
    return GrowwFeed(api_client or get_api_client(config))
