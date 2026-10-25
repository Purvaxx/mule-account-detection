"""
Foreign-exchange seam — declared, deliberately unimplemented
============================================================

The frozen Experiment 3 model consumes ``amount_usd``.  A dataset denominated in
several currencies therefore cannot be scored without converting its amounts,
and converting them honestly requires **historical, dated, attributable** rates:
a transaction on 2018-04-11 in EUR must be converted at that day's rate, and the
resulting figure must be traceable to the source that supplied it.

The repository has no such source.  Inventing rates — a hard-coded table, a
present-day rate applied retroactively, or a rate scraped without provenance —
would produce a confidently-wrong ``amount_usd`` feeding two of the five
baseline scalars for every entity in the dataset.  That is explicitly refused
; ``CurrencyPolicy.EXTERNAL_FX_REQUIRED`` reports
``INCOMPATIBLE`` instead.

**This module implements no conversion.**  It exists so that the shape of a
future provenance-backed FX layer is fixed now, and so that exactly one place in
the codebase answers "is FX conversion available?".  When a real provider is
introduced it implements :class:`FxRateProvider` and is returned by
:func:`get_fx_provider`; nothing else in the dataset path needs to change.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date
from typing import Any, Dict, Optional, Protocol, runtime_checkable

__all__ = [
    "FxQuote",
    "FxRateProvider",
    "get_fx_provider",
    "fx_availability",
]


@dataclass(frozen=True)
class FxQuote:
    """A single dated rate **with its provenance**.

    ``source`` and ``retrieved_at`` are not optional decoration: a converted
    amount that cannot say where its rate came from is not evidence, and the
    compatibility report would have no honest way to describe it.
    """

    currency: str
    #: Units of ``currency`` per 1 USD, on ``as_of``.
    rate: float
    as_of: date
    #: Identifier of the publishing authority (e.g. a central bank series id).
    source: str
    #: When this quote was retrieved from ``source``.
    retrieved_at: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["as_of"] = self.as_of.isoformat()
        return d


@runtime_checkable
class FxRateProvider(Protocol):
    """The contract a future FX implementation must satisfy.

    Implementations must be *historical* (the rate for the transaction's own
    date, not today's) and must refuse rather than interpolate: a provider that
    has no quote for a currency/date raises, so the dataset is reported
    incompatible instead of silently converted at a nearby rate.
    """

    #: Identifies the provider in the compatibility report and manifest.
    name: str

    def quote(self, currency: str, on: date) -> FxQuote:
        """Return the dated quote, or raise if this provider cannot supply it."""
        ...


def get_fx_provider() -> Optional[FxRateProvider]:
    """The configured FX provider, or ``None`` when there is none.

    There is currently no provider and this always returns ``None``.  It is the
    single point the dataset policy layer consults, so introducing a real
    provider is a change here and nowhere else.
    """
    return None


def fx_availability() -> Dict[str, Any]:
    """Describe FX support for the compatibility report and the UI."""
    provider = get_fx_provider()
    if provider is None:
        return {
            "available": False,
            "provider": None,
            "reason": (
                "No provenance-backed FX provider is configured. Converting "
                "multi-currency amounts requires dated, attributable historical "
                "rates, which this repository does not have. Rates will not be "
                "fabricated."
            ),
        }
    return {
        "available": True,
        "provider": getattr(provider, "name", provider.__class__.__name__),
        "reason": None,
    }
