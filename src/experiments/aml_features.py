"""
Layer 6 - Scalable AML Feature Families
=========================================

Six independently versioned families that cover the evidence an AML analyst
actually reasons about, computed from the edge table with NumPy and SciPy so
that a 9.5-million-transaction, 855-thousand-account benchmark is a minute of
work rather than an afternoon.

    aml_txn_v1        volume and amount behaviour
    aml_temporal_v1   timing, velocity, burstiness, dormancy, recency
    aml_temporal_v2   the same, measured on the window's own activity clock
    aml_flow_v1       fan-in / fan-out, imbalance, pass-through, turnaround
    aml_network_v1    degree, counterparty structure, reciprocity, component
    aml_cycle_v1      reciprocity, triangles, closed 4-walks, ordered layering
    aml_context_v1    payment-channel, currency and jurisdiction mix

They are separate registry entries, not one block, because Part IX of the
model-selection programme asks which *information source* contributes: an
ablation can only answer that if the sources are separable.  ``aml_core_v1`` in
:mod:`src.experiments.features` stays exactly as it was - it is pinned by
existing artifacts - and these do not replace it.

Temporal semantics
------------------
Every column here is a summary of transactions present in
``FeatureRequest.edge_table``.  Under ``FeatureSpec.scope == "train_window"``
the runner has already truncated that table to rows strictly before the fold's
boundary, so no column can describe behaviour that had not happened yet.  Two
columns need an explicit "now" and take it from ``FeatureRequest.as_of`` when
the scope supplies one, falling back to the latest observed timestamp and
recording that in the column documentation:

``aml_temporal_v1::recency_days``   ``as_of`` minus the entity's last activity
``aml_temporal_v1::tenure_days``    ``as_of`` minus the entity's first activity

Every other column is invariant to the choice of "now".

Undefined values
----------------
A statistic with no meaning for an entity - an inter-arrival time from a single
transaction, a ratio over an empty denominator - is ``NaN``.  It is never 0
(indistinguishable from a real zero) and never a sentinel (a number a tree
splits on).  Imputation is the candidate's train-fitted business.

Amounts
-------
Read through :func:`src.experiments.sources.amount_column`, which prefers
``amount_usd`` and falls back to the native ``amount``.  Nothing here converts
a currency; on a multi-currency benchmark the amount columns are comparable
only within a currency and the dataset spec says so.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.experiments.features import FeatureRequest, FeatureSet, register_feature_set

logger = logging.getLogger(__name__)

__all__ = ["EventView", "event_view"]

_EPS = 1e-12
_SEC_PER_DAY = 86400.0
_HOUR = 3600.0
#: Days since the epoch, plus this, taken modulo 7, gives Monday = 0.
#: 1970-01-01 was a Thursday, which is index 3.
_EPOCH_DOW_OFFSET = 3


# ---------------------------------------------------------------------------
# Shared per-request computation
# ---------------------------------------------------------------------------


@dataclass
class EventView:
    """Edge table flattened into one event per (transaction, endpoint).

    Built once per :class:`FeatureRequest` and cached on it, because all five
    families need the same flattening and it is the expensive part: on SAML-D
    it is 19 million rows.

    Attributes
    ----------
    n:
        Number of entities.  Every array indexed by entity has this length.
    src, dst:
        Positional entity indices of each transaction.
    amount:
        Per-transaction amount, or ``None`` when the source has no amount.
    seconds:
        Per-transaction timestamp as float seconds since the epoch, ``NaN``
        where the transaction is undated, or ``None`` when the source is
        entirely undated.
    ent, cp, is_in:
        The flattened event arrays: acting entity, its counterparty, and
        whether the entity received (``True``) or sent (``False``).  Each
        transaction contributes two events, so these are twice the length of
        ``src``.
    ev_amount, ev_seconds:
        ``amount`` and ``seconds`` repeated to match the flattened arrays, or
        ``None`` when the source has no amount / no timestamps.
    """

    n: int
    src: np.ndarray
    dst: np.ndarray
    amount: Optional[np.ndarray]
    seconds: Optional[np.ndarray]
    ent: np.ndarray
    cp: np.ndarray
    is_in: np.ndarray
    ev_amount: Optional[np.ndarray]
    ev_seconds: Optional[np.ndarray]

    # -- reductions ------------------------------------------------------

    def count_by(self, mask: Optional[np.ndarray] = None) -> np.ndarray:
        keys = self.ent if mask is None else self.ent[mask]
        return np.bincount(keys, minlength=self.n).astype(np.float64)

    def sum_by(self, values: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        keys = self.ent if mask is None else self.ent[mask]
        vals = values if mask is None else values[mask]
        finite = np.isfinite(vals)
        return np.bincount(keys[finite], weights=vals[finite], minlength=self.n)

    def nan_ratio(self, numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        """Elementwise ratio, ``NaN`` where the denominator is zero."""
        out = np.full(self.n, np.nan, dtype=np.float64)
        ok = denominator > 0
        out[ok] = numerator[ok] / denominator[ok]
        return out


def _as_seconds(edges: pd.DataFrame) -> Optional[np.ndarray]:
    if "date" not in edges.columns:
        return None
    dates = pd.to_datetime(edges["date"], errors="coerce")
    values = dates.to_numpy(dtype="datetime64[s]").astype(np.float64)
    values[dates.isna().to_numpy()] = np.nan
    if not np.isfinite(values).any():
        return None
    return values


def event_view(request: FeatureRequest) -> Optional[EventView]:
    """Build (or return the cached) :class:`EventView` for ``request``.

    Returns ``None`` when the request carries no transactions, which is a
    legitimate state - an early time-forward fold can have an empty window -
    and every family must handle it by returning an all-``NaN`` block rather
    than by raising.
    """
    cached = request.cache.get("event_view")
    if cached is not None:
        return cached if isinstance(cached, EventView) else None

    edges = request.edge_table
    if edges is None or len(edges) == 0:
        request.cache["event_view"] = False
        return None

    from src.experiments.sources import amount_column, entity_codes

    src, dst = entity_codes(edges, request.entity_ids)
    src = src.astype(np.int64, copy=False)
    dst = dst.astype(np.int64, copy=False)
    n = len(request.entity_ids)

    amount_name = amount_column(edges)
    amount = (
        pd.to_numeric(edges[amount_name], errors="coerce").to_numpy(dtype=np.float64)
        if amount_name
        else None
    )
    if amount is not None:
        amount = np.abs(amount)
    seconds = _as_seconds(edges)

    ent = np.concatenate([src, dst])
    cp = np.concatenate([dst, src])
    is_in = np.concatenate(
        [np.zeros(len(src), dtype=bool), np.ones(len(dst), dtype=bool)]
    )
    ev_amount = None if amount is None else np.concatenate([amount, amount])
    ev_seconds = None if seconds is None else np.concatenate([seconds, seconds])

    view = EventView(
        n=n,
        src=src,
        dst=dst,
        amount=amount,
        seconds=seconds,
        ent=ent,
        cp=cp,
        is_in=is_in,
        ev_amount=ev_amount,
        ev_seconds=ev_seconds,
    )
    request.cache["event_view"] = view
    return view


def _empty(request: FeatureRequest, columns: list[str]) -> pd.DataFrame:
    index = pd.Index(request.entity_ids, name="entity_id")
    return pd.DataFrame(np.nan, index=index, columns=columns, dtype=np.float64)


def _frame(request: FeatureRequest, data: dict[str, np.ndarray]) -> pd.DataFrame:
    index = pd.Index(request.entity_ids, name="entity_id")
    return pd.DataFrame(data, index=index, dtype=np.float64)


def _grouped_stat(
    keys: np.ndarray, values: np.ndarray, n: int, how: str
) -> np.ndarray:
    """Per-entity ``min`` / ``max`` of ``values``, ``NaN`` where absent.

    ``np.minimum.at`` is an unbuffered ufunc call and is roughly two orders of
    magnitude slower than a groupby at 19 million rows, so this goes through
    pandas.
    """
    out = np.full(n, np.nan, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return out
    series = pd.Series(values[finite]).groupby(keys[finite], sort=False)
    agg = series.min() if how == "min" else series.max()
    out[agg.index.to_numpy()] = agg.to_numpy()
    return out


def _nunique(keys: np.ndarray, values: np.ndarray, n: int) -> np.ndarray:
    """Per-entity count of distinct ``values``. Zero where the entity is absent."""
    out = np.zeros(n, dtype=np.float64)
    agg = pd.Series(values).groupby(keys, sort=False).nunique()
    out[agg.index.to_numpy()] = agg.to_numpy()
    return out


def _mode_share(keys: np.ndarray, values: np.ndarray, n: int) -> np.ndarray:
    """Per-entity share of the most frequent ``values`` code. ``NaN`` where absent."""
    out = np.full(n, np.nan, dtype=np.float64)
    frame = pd.DataFrame({"k": keys, "v": values})
    counts = frame.groupby(["k", "v"], sort=False).size()
    top = counts.groupby(level=0, sort=False).max()
    total = counts.groupby(level=0, sort=False).sum()
    idx = top.index.to_numpy()
    out[idx] = top.to_numpy() / np.maximum(total.to_numpy(), 1)
    return out


def _grouped_std(
    keys: np.ndarray, values: np.ndarray, mean: np.ndarray, n: int
) -> np.ndarray:
    """Population standard deviation per entity, computed in two passes.

    The one-pass identity ``E[X^2] - E[X]^2`` is the obvious way to do this with
    a single ``bincount``, and it is the wrong way here.  SAML-D amounts reach
    1.26e7, so their squares reach 1.6e14 and a busy entity's sum of squares
    reaches ~1e17 - past the point where float64 represents consecutive integers
    exactly.  Subtracting two nearly equal numbers of that size destroys most of
    the significant digits, and it destroys them precisely for the entities
    whose amounts barely vary, which is the structuring signature this feature
    exists to catch.

    Summing squared deviations from an already-computed mean costs one extra
    pass over the same array and has no such cancellation.

    ``NaN`` where the entity has no defined value; 0 where it has exactly one.
    """
    out = np.full(n, np.nan, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return out
    k = keys[finite]
    counts = np.bincount(k, minlength=n).astype(np.float64)
    deviation = values[finite] - mean[k]
    sum_sq_dev = np.bincount(k, weights=deviation**2, minlength=n)
    ok = counts > 0
    out[ok] = np.sqrt(np.maximum(sum_sq_dev[ok] / counts[ok], 0.0))
    return out


def _now_seconds(request: FeatureRequest, view: EventView) -> tuple[float, str]:
    """The reference "now" for recency, and where it came from."""
    if request.as_of is not None:
        return float(pd.Timestamp(request.as_of).timestamp()), "as_of"
    if view.ev_seconds is not None and np.isfinite(view.ev_seconds).any():
        return float(np.nanmax(view.ev_seconds)), "max_observed"
    return float("nan"), "undefined"


# ---------------------------------------------------------------------------
# aml_txn_v1 - volume and amount
# ---------------------------------------------------------------------------

_TXN_COLUMNS = [
    "txn_in", "txn_out", "txn_total", "in_out_txn_ratio",
    "amount_in_sum", "amount_out_sum", "amount_total",
    "amount_in_mean", "amount_out_mean", "amount_mean",
    "amount_max", "amount_min", "amount_std", "amount_cv",
    "amount_log_mean", "amount_top1_share", "round_amount_share",
]


def _aml_txn(request: FeatureRequest) -> pd.DataFrame:
    """Counts and amount statistics. No timing, no structure.

    ``round_amount_share`` is the fraction of an entity's transactions whose
    amount is an exact multiple of 100 - the crudest structuring tell, and one
    that costs nothing to compute.  ``amount_top1_share`` is the largest single
    transaction as a fraction of total throughput, which separates an entity
    with one big movement from one with steady traffic.
    """
    view = event_view(request)
    if view is None:
        return _empty(request, _TXN_COLUMNS)

    n = view.n
    txn_in = view.count_by(view.is_in)
    txn_out = view.count_by(~view.is_in)
    txn_total = txn_in + txn_out

    data: dict[str, np.ndarray] = {
        "txn_in": txn_in,
        "txn_out": txn_out,
        "txn_total": txn_total,
        "in_out_txn_ratio": view.nan_ratio(txn_in, txn_in + txn_out),
    }

    if view.ev_amount is None:
        for column in _TXN_COLUMNS:
            data.setdefault(column, np.full(n, np.nan))
        return _frame(request, data)[_TXN_COLUMNS]

    amounts = view.ev_amount
    amount_in_sum = view.sum_by(amounts, view.is_in)
    amount_out_sum = view.sum_by(amounts, ~view.is_in)
    amount_total = amount_in_sum + amount_out_sum
    defined = np.isfinite(amounts)
    n_defined = np.bincount(view.ent[defined], minlength=n).astype(np.float64)
    mean = view.nan_ratio(amount_total, n_defined)
    std = _grouped_std(view.ent, amounts, mean, n)

    amount_max = _grouped_stat(view.ent, amounts, n, "max")
    log_sum = np.bincount(
        view.ent[defined], weights=np.log1p(amounts[defined]), minlength=n
    )
    is_round = defined & (np.remainder(amounts, 100.0) == 0)
    round_count = np.bincount(view.ent[is_round], minlength=n).astype(np.float64)

    data.update(
        {
            "amount_in_sum": np.where(txn_in > 0, amount_in_sum, np.nan),
            "amount_out_sum": np.where(txn_out > 0, amount_out_sum, np.nan),
            "amount_total": np.where(n_defined > 0, amount_total, np.nan),
            "amount_in_mean": view.nan_ratio(amount_in_sum, txn_in),
            "amount_out_mean": view.nan_ratio(amount_out_sum, txn_out),
            "amount_mean": mean,
            "amount_max": amount_max,
            "amount_min": _grouped_stat(view.ent, amounts, n, "min"),
            "amount_std": np.where(n_defined > 1, std, np.nan),
            "amount_cv": view.nan_ratio(std, np.where(mean > 0, mean, 0.0)),
            "amount_log_mean": view.nan_ratio(log_sum, n_defined),
            "amount_top1_share": view.nan_ratio(
                amount_max, np.where(amount_total > 0, amount_total, 0.0)
            ),
            "round_amount_share": view.nan_ratio(round_count, n_defined),
        }
    )
    return _frame(request, data)[_TXN_COLUMNS]


# ---------------------------------------------------------------------------
# aml_temporal_v1
# ---------------------------------------------------------------------------

_TEMPORAL_COLUMNS = [
    "active_days", "span_days", "txn_per_active_day", "txn_per_span_day",
    "mean_iat_hours", "std_iat_hours", "min_iat_hours", "max_gap_days",
    "burstiness", "max_daily_txn", "daily_concentration",
    "night_share", "weekend_share", "second_half_share",
    "recency_days", "tenure_days",
]


def _aml_temporal(request: FeatureRequest) -> pd.DataFrame:
    """Timing behaviour: velocity, regularity, concentration, dormancy, recency.

    ``burstiness`` is the Goh-Barabasi coefficient ``(sigma - mu) / (sigma +
    mu)`` over inter-arrival times: -1 for perfectly regular timing, 0 for
    Poisson, +1 for maximally bursty.  ``second_half_share`` is the fraction of
    an entity's transactions that fall in the later half of *its own* active
    span, which is how a dormant-to-active transition shows up without needing
    a global reference point.

    ``recency_days`` and ``tenure_days`` are measured from ``request.as_of``
    when the experiment scoped features to a training window, and from the
    latest observed transaction otherwise; both readings are stated in the
    module docstring.
    """
    view = event_view(request)
    if view is None or view.ev_seconds is None:
        return _empty(request, _TEMPORAL_COLUMNS)

    n = view.n
    seconds = view.ev_seconds
    dated = np.isfinite(seconds)
    ent = view.ent[dated]
    t = seconds[dated]

    counts = np.bincount(ent, minlength=n).astype(np.float64)
    t_min = _grouped_stat(view.ent, seconds, n, "min")
    t_max = _grouped_stat(view.ent, seconds, n, "max")
    span_days = (t_max - t_min) / _SEC_PER_DAY

    days = np.floor(t / _SEC_PER_DAY)
    active_days = _nunique(ent, days, n)
    active_days[counts == 0] = np.nan

    daily = pd.Series(np.ones(len(ent))).groupby([ent, days], sort=False).size()
    max_daily = np.full(n, np.nan)
    grouped_max = daily.groupby(level=0, sort=False).max()
    max_daily[grouped_max.index.to_numpy()] = grouped_max.to_numpy()

    # Inter-arrival statistics: sort once by (entity, time) and diff inside
    # each entity's segment.  The first event of each segment has no gap.
    order = np.lexsort((t, ent))
    ent_sorted = ent[order]
    t_sorted = t[order]
    same = np.empty(len(t_sorted), dtype=bool)
    same[0] = False
    same[1:] = ent_sorted[1:] == ent_sorted[:-1]
    gaps = np.empty(len(t_sorted), dtype=np.float64)
    gaps[0] = np.nan
    gaps[1:] = t_sorted[1:] - t_sorted[:-1]
    gaps[~same] = np.nan
    gap_ok = np.isfinite(gaps)
    n_gap = np.bincount(ent_sorted[gap_ok], minlength=n).astype(np.float64)
    gap_sum = np.bincount(ent_sorted[gap_ok], weights=gaps[gap_ok], minlength=n)
    mean_gap = view.nan_ratio(gap_sum, n_gap)
    std_gap = _grouped_std(ent_sorted, gaps, mean_gap, n)
    min_gap = _grouped_stat(ent_sorted, gaps, n, "min")
    max_gap = _grouped_stat(ent_sorted, gaps, n, "max")
    denominator = std_gap + mean_gap
    burstiness = np.full(n, np.nan)
    ok = np.isfinite(denominator) & (np.abs(denominator) > _EPS)
    burstiness[ok] = (std_gap[ok] - mean_gap[ok]) / denominator[ok]

    # Hour of day is read straight off the epoch, so it is UTC. The benchmarks
    # ship naive local timestamps with no zone, and inventing one would be a
    # fabrication; ``night_share`` is therefore "the small hours of the
    # timestamp as recorded", which is what the data supports.
    hour = np.floor(np.remainder(t, _SEC_PER_DAY) / _HOUR)
    night = np.bincount(ent[hour < 6], minlength=n).astype(np.float64)
    # 1970-01-01 was a Thursday. Offsetting by _EPOCH_DOW_OFFSET puts Monday at
    # 0, so Saturday is 5 and Sunday is 6.
    dow = np.remainder(days + _EPOCH_DOW_OFFSET, 7)
    weekend = np.bincount(ent[dow >= 5], minlength=n).astype(np.float64)

    midpoint = (t_min + t_max) / 2.0
    later = t >= midpoint[ent]
    second_half = np.bincount(ent[later], minlength=n).astype(np.float64)

    now, _ = _now_seconds(request, view)
    recency = (now - t_max) / _SEC_PER_DAY
    tenure = (now - t_min) / _SEC_PER_DAY

    data = {
        "active_days": active_days,
        "span_days": span_days,
        "txn_per_active_day": view.nan_ratio(counts, np.nan_to_num(active_days)),
        "txn_per_span_day": view.nan_ratio(counts, np.where(span_days > 0, span_days, 0.0)),
        "mean_iat_hours": mean_gap / _HOUR,
        "std_iat_hours": std_gap / _HOUR,
        "min_iat_hours": min_gap / _HOUR,
        "max_gap_days": max_gap / _SEC_PER_DAY,
        "burstiness": burstiness,
        "max_daily_txn": max_daily,
        "daily_concentration": view.nan_ratio(np.nan_to_num(max_daily), counts),
        "night_share": view.nan_ratio(night, counts),
        "weekend_share": view.nan_ratio(weekend, counts),
        "second_half_share": view.nan_ratio(second_half, counts),
        "recency_days": recency,
        "tenure_days": tenure,
    }
    return _frame(request, data)[_TEMPORAL_COLUMNS]


# ---------------------------------------------------------------------------
# aml_temporal_v2 - window-relative timing
# ---------------------------------------------------------------------------

_TEMPORAL_V2_COLUMNS = [
    "active_bin_fraction", "span_activity", "txn_per_active_bin",
    "txn_per_span_activity",
    "mean_iat_hours", "std_iat_hours", "min_iat_hours", "max_gap_activity",
    "burstiness", "max_bin_txn", "activity_concentration",
    "night_share", "window_half_share",
    "recency_activity", "tenure_activity",
]

#: Deciles.  Ten equal-activity bins replace ``aml_temporal_v1``'s calendar
#: days: a calendar day is an alignment artefact of where the window boundaries
#: fall, and its volume differs between windows (IBM HI-Small: a Friday carries
#: 3.1x a Saturday).
_ACTIVITY_BINS = 10


def _activity_clock(view: EventView) -> Optional[np.ndarray]:
    """Per-event position on the window's own activity clock, in ``[0, 1)``.

    ``tau(t)`` is the share of the window's transactions dated strictly before
    ``t``.  Time is measured in "how much happened" rather than in hours, so a
    window whose volume sits mostly on one of its days is comparable with one
    whose volume is spread evenly.

    Computed from the transactions in ``view`` and nothing else: no labels, no
    rows outside the window, no reference to another window, and no
    ``FeatureRequest.as_of``.  ``None`` when the window carries no dated
    transaction; ``NaN`` for an individual undated event.
    """
    if view.seconds is None or view.ev_seconds is None:
        return None
    txn = view.seconds[np.isfinite(view.seconds)]
    if not len(txn):
        return None
    ordered = np.sort(txn)
    tau = np.full(len(view.ev_seconds), np.nan, dtype=np.float64)
    dated = np.isfinite(view.ev_seconds)
    tau[dated] = (
        np.searchsorted(ordered, view.ev_seconds[dated], side="left") / len(ordered)
    )
    return tau


def _aml_temporal_v2(request: FeatureRequest) -> pd.DataFrame:
    """Timing behaviour measured on the detection window's own activity clock.

    The window-relative counterpart of ``aml_temporal_v1``, built for
    cross-period transfer: R1 (``docs/r1_detection_transfer.md``) found that the
    features drifting hardest between two detection windows are those counting
    calendar days or measuring position against a wall clock, and that the fold
    carrying that drift lost 89 % of its in-period average precision.

    What changes and why is tabulated in ``docs/r6_temporal_v2.md`` §4.  In
    short: position, span, dormancy and concentration move to the activity clock
    (``recency_activity``, ``tenure_activity``, ``span_activity``,
    ``max_gap_activity``, ``activity_concentration``, ``max_bin_txn``,
    ``active_bin_fraction``, ``txn_per_active_bin``, ``txn_per_span_activity``,
    ``window_half_share``); inter-arrival velocity, burstiness and
    ``night_share`` stay in wall-clock units because that is what they mean; and
    ``weekend_share`` is dropped because in a short window it encodes which
    calendar days the window covers rather than anything about the entity.

    Undefined stays ``NaN``, never 0, exactly as in v1.
    """
    view = event_view(request)
    if view is None or view.ev_seconds is None:
        return _empty(request, _TEMPORAL_V2_COLUMNS)
    tau = _activity_clock(view)
    if tau is None:
        return _empty(request, _TEMPORAL_V2_COLUMNS)

    n = view.n
    seconds = view.ev_seconds
    dated = np.isfinite(seconds) & np.isfinite(tau)
    ent = view.ent[dated]
    t = seconds[dated]
    u = tau[dated]

    counts = np.bincount(ent, minlength=n).astype(np.float64)
    u_min = _grouped_stat(view.ent, tau, n, "min")
    u_max = _grouped_stat(view.ent, tau, n, "max")
    span_activity = u_max - u_min

    bins = np.minimum(np.floor(u * _ACTIVITY_BINS), _ACTIVITY_BINS - 1)
    active_bins = _nunique(ent, bins, n)
    active_bins[counts == 0] = np.nan

    per_bin = pd.Series(np.ones(len(ent))).groupby([ent, bins], sort=False).size()
    max_bin = np.full(n, np.nan)
    grouped_max = per_bin.groupby(level=0, sort=False).max()
    max_bin[grouped_max.index.to_numpy()] = grouped_max.to_numpy()

    # Velocity stays on the wall clock: "one transaction every twenty minutes"
    # is an AML signal in hours, not in activity share.  Same computation as v1.
    order = np.lexsort((t, ent))
    ent_sorted = ent[order]
    t_sorted = t[order]
    same = np.empty(len(t_sorted), dtype=bool)
    same[0] = False
    same[1:] = ent_sorted[1:] == ent_sorted[:-1]
    gaps = np.empty(len(t_sorted), dtype=np.float64)
    gaps[0] = np.nan
    gaps[1:] = t_sorted[1:] - t_sorted[:-1]
    gaps[~same] = np.nan
    gap_ok = np.isfinite(gaps)
    n_gap = np.bincount(ent_sorted[gap_ok], minlength=n).astype(np.float64)
    gap_sum = np.bincount(ent_sorted[gap_ok], weights=gaps[gap_ok], minlength=n)
    mean_gap = view.nan_ratio(gap_sum, n_gap)
    std_gap = _grouped_std(ent_sorted, gaps, mean_gap, n)
    min_gap = _grouped_stat(ent_sorted, gaps, n, "min")
    denominator = std_gap + mean_gap
    burstiness = np.full(n, np.nan)
    ok = np.isfinite(denominator) & (np.abs(denominator) > _EPS)
    burstiness[ok] = (std_gap[ok] - mean_gap[ok]) / denominator[ok]

    # Dormancy, by contrast, is relative to how much happened in the window.
    order_u = np.lexsort((u, ent))
    ent_u = ent[order_u]
    u_sorted = u[order_u]
    same_u = np.empty(len(u_sorted), dtype=bool)
    same_u[0] = False
    same_u[1:] = ent_u[1:] == ent_u[:-1]
    gaps_u = np.empty(len(u_sorted), dtype=np.float64)
    gaps_u[0] = np.nan
    gaps_u[1:] = u_sorted[1:] - u_sorted[:-1]
    gaps_u[~same_u] = np.nan
    max_gap_activity = _grouped_stat(ent_u, gaps_u, n, "max")

    # Hour of day is read straight off the epoch, so it is UTC, exactly as v1
    # documents: the benchmarks ship naive timestamps and inventing a zone
    # would be a fabrication.
    hour = np.floor(np.remainder(t, _SEC_PER_DAY) / _HOUR)
    night = np.bincount(ent[hour < 6], minlength=n).astype(np.float64)

    # The window's activity midpoint, not the entity's own span midpoint: the
    # latter is degenerate for a single-transaction entity and shifts meaning
    # with the entity's span.
    half = np.bincount(ent[u >= 0.5], minlength=n).astype(np.float64)

    data = {
        "active_bin_fraction": active_bins / float(_ACTIVITY_BINS),
        "span_activity": span_activity,
        "txn_per_active_bin": view.nan_ratio(counts, np.nan_to_num(active_bins)),
        "txn_per_span_activity": view.nan_ratio(
            counts, np.where(span_activity > 0, span_activity, 0.0)
        ),
        "mean_iat_hours": mean_gap / _HOUR,
        "std_iat_hours": std_gap / _HOUR,
        "min_iat_hours": min_gap / _HOUR,
        "max_gap_activity": max_gap_activity,
        "burstiness": burstiness,
        "max_bin_txn": max_bin,
        "activity_concentration": view.nan_ratio(np.nan_to_num(max_bin), counts),
        "night_share": view.nan_ratio(night, counts),
        "window_half_share": view.nan_ratio(half, counts),
        "recency_activity": 1.0 - u_max,
        "tenure_activity": 1.0 - u_min,
    }
    return _frame(request, data)[_TEMPORAL_V2_COLUMNS]


# ---------------------------------------------------------------------------
# aml_flow_v1
# ---------------------------------------------------------------------------

_FLOW_COLUMNS = [
    "fan_in", "fan_out", "fan_ratio", "fan_product_norm", "fan_min",
    "inflow_outflow_ratio", "flow_imbalance", "pass_through_ratio",
    "min_in_to_out_hours", "median_in_to_out_hours",
    "rapid_turnover_share", "instant_turnaround_share",
]

#: An outgoing transaction counts as a turnaround when money arrived at the
#: same entity within this many hours before it.
_RAPID_TURNAROUND_HOURS = 24.0
#: The tighter band: money leaving within an hour of arriving. Distinct from
#: ``_RAPID_TURNAROUND_HOURS`` rather than a duplicate of it, which is why the
#: column is named for the interval it actually tests.
_INSTANT_TURNAROUND_HOURS = 1.0


def _time_offsets(seconds: np.ndarray) -> tuple[np.ndarray, int]:
    """Integer second offsets from the earliest timestamp, plus the span.

    The turnaround joins below build a globally monotone ``entity * span +
    offset`` key so that one ``searchsorted`` answers a per-entity question for
    every entity at once.  The key is built in ``int64``, not ``float64``: at
    855 thousand entities and a ten-month span the float version is still exact,
    but the product grows with both dimensions and a silent loss of
    second-level resolution on a larger dataset would corrupt the join without
    raising anything.  ``int64`` has room for roughly 9.2e18, which is 10^11
    entities over a decade.

    Undated rows get offset 0; every caller masks them out before use.
    """
    finite = np.isfinite(seconds)
    base = int(np.nanmin(seconds))
    offset = np.zeros(len(seconds), dtype=np.int64)
    offset[finite] = seconds[finite].astype(np.int64) - base
    span = int(offset[finite].max()) + 1 if finite.any() else 1
    return offset, span


def _preceding_inflow_gap(view: EventView) -> Optional[np.ndarray]:
    """Hours from the most recent inflow to each outflow, per outflow event.

    Vectorised across all entities at once.  Times are mapped to a globally
    monotone key ``entity * span + offset`` so that a single ``searchsorted``
    over every entity's inflows answers "the latest inflow at *this* entity at
    or before this outflow" - a per-entity Python loop over 855 thousand
    entities would dominate the whole feature computation.

    ``NaN`` where an outflow has no inflow before it.
    """
    if view.seconds is None:
        return None
    t = view.seconds
    dated = np.isfinite(t)
    if not dated.any():
        return None

    offset, span = _time_offsets(t)

    in_key = view.dst[dated] * span + offset[dated]
    in_time = t[dated]
    in_ent = view.dst[dated]
    order = np.argsort(in_key, kind="stable")
    in_key = in_key[order]
    in_time = in_time[order]
    in_ent = in_ent[order]

    out_key = view.src[dated] * span + offset[dated]
    out_time = t[dated]
    out_ent = view.src[dated]

    pos = np.searchsorted(in_key, out_key, side="right") - 1
    gap = np.full(len(out_key), np.nan, dtype=np.float64)
    valid = pos >= 0
    matched = np.zeros(len(out_key), dtype=bool)
    matched[valid] = in_ent[pos[valid]] == out_ent[valid]
    take = valid & matched
    gap[take] = (out_time[take] - in_time[pos[take]]) / _HOUR
    return gap


def _aml_flow(request: FeatureRequest) -> pd.DataFrame:
    """Directional flow behaviour.

    ``fan_product_norm`` is ``fan_in * fan_out / txn_total``: the gather-scatter
    signature, high for an entity that collects from many and redistributes to
    many without simply being busy.  ``min_in_to_out_hours``,
    ``rapid_turnover_share`` (within 24 hours) and ``instant_turnaround_share``
    (within one hour) are the layering signature - money that leaves almost as
    soon as it arrives.
    """
    view = event_view(request)
    if view is None:
        return _empty(request, _FLOW_COLUMNS)

    n = view.n
    txn_in = view.count_by(view.is_in)
    txn_out = view.count_by(~view.is_in)
    txn_total = txn_in + txn_out
    # Self-loops are excluded, matching aml_network_v1 and aml_cycle_v1: an
    # account paying itself is not a counterparty, and counting it would make
    # fan_in / fan_out disagree with counterparty_count on the same entity.
    not_loop = view.src != view.dst
    fan_in = _nunique(view.dst[not_loop], view.src[not_loop], n)
    fan_out = _nunique(view.src[not_loop], view.dst[not_loop], n)

    data: dict[str, np.ndarray] = {
        "fan_in": fan_in,
        "fan_out": fan_out,
        "fan_ratio": view.nan_ratio(fan_in, fan_in + fan_out),
        "fan_product_norm": view.nan_ratio(fan_in * fan_out, txn_total),
        "fan_min": np.minimum(fan_in, fan_out).astype(np.float64),
    }

    if view.ev_amount is not None:
        inflow = view.sum_by(view.ev_amount, view.is_in)
        outflow = view.sum_by(view.ev_amount, ~view.is_in)
        total = inflow + outflow
        data["inflow_outflow_ratio"] = view.nan_ratio(inflow, np.where(outflow > 0, outflow, 0.0))
        data["flow_imbalance"] = view.nan_ratio(inflow - outflow, total)
        data["pass_through_ratio"] = view.nan_ratio(
            np.minimum(inflow, outflow), np.maximum(inflow, outflow)
        )
    else:
        for column in (
            "inflow_outflow_ratio", "flow_imbalance", "pass_through_ratio",
        ):
            data[column] = np.full(n, np.nan)

    gap = _preceding_inflow_gap(view)
    if gap is None:
        for column in (
            "min_in_to_out_hours", "median_in_to_out_hours",
            "rapid_turnover_share", "instant_turnaround_share",
        ):
            data[column] = np.full(n, np.nan)
    else:
        dated = np.isfinite(view.seconds)
        out_ent = view.src[dated]
        data["min_in_to_out_hours"] = _grouped_stat(out_ent, gap, n, "min")
        median = np.full(n, np.nan)
        finite = np.isfinite(gap)
        if finite.any():
            agg = pd.Series(gap[finite]).groupby(out_ent[finite], sort=False).median()
            median[agg.index.to_numpy()] = agg.to_numpy()
        data["median_in_to_out_hours"] = median
        rapid = finite & (gap <= _RAPID_TURNAROUND_HOURS)
        instant = finite & (gap <= _INSTANT_TURNAROUND_HOURS)
        n_out_dated = np.bincount(out_ent, minlength=n).astype(np.float64)
        data["rapid_turnover_share"] = view.nan_ratio(
            np.bincount(out_ent[rapid], minlength=n).astype(np.float64), n_out_dated
        )
        data["instant_turnaround_share"] = view.nan_ratio(
            np.bincount(out_ent[instant], minlength=n).astype(np.float64), n_out_dated
        )

    return _frame(request, data)[_FLOW_COLUMNS]


# ---------------------------------------------------------------------------
# aml_network_v1
# ---------------------------------------------------------------------------

_NETWORK_COLUMNS = [
    "counterparty_count", "counterparty_diversity",
    "top_counterparty_share", "repeat_pair_mean",
    "reciprocal_partners", "reciprocal_ratio",
    "component_size", "component_edge_density",
    "neighbour_txn_mean", "neighbour_txn_max", "neighbour_counterparty_mean",
]


def _simple_adjacency(request: FeatureRequest, view: EventView):
    """Cached SciPy CSR adjacency of the *simple, loop-free* directed graph.

    Two reductions, both of which change what the structural columns mean and
    are therefore stated here rather than left to be inferred:

    * **Parallel edges are collapsed**, with the multiplicity kept separately.
      This is what makes the structural families affordable: SAML-D's 9.5
      million transactions become 887 thousand distinct ordered pairs, and every
      sparse product below is computed on those.
    * **Self-loops are dropped.** An account paying itself is not a counterparty
      relationship, and leaving the diagonal in would let a self-loop close a
      degenerate "cycle" - ``i -> i -> j -> i`` would be counted as a directed
      triangle. The Reference Dataset has 33 self-loops and the IBM AML extract
      1,709, so this is not hypothetical. SAML-D has none, so its columns are
      unaffected either way.
    """
    cached = request.cache.get("simple_adjacency")
    if cached is not None:
        return cached
    from scipy.sparse import csr_matrix

    n = view.n
    keep = view.src != view.dst
    adjacency = csr_matrix(
        (
            np.ones(int(keep.sum()), dtype=np.int32),
            (view.src[keep], view.dst[keep]),
        ),
        shape=(n, n),
    )
    adjacency.sum_duplicates()
    multiplicity = adjacency.copy()
    adjacency.data[:] = 1
    request.cache["simple_adjacency"] = (adjacency, multiplicity)
    return adjacency, multiplicity


def _aml_network(request: FeatureRequest) -> pd.DataFrame:
    """Counterparty structure and one hop of neighbourhood aggregation.

    ``counterparty_count`` *is* the undirected degree.  Degree centrality is
    not a separate column: normalising by ``n - 1``, a constant within a run,
    produces a rescaled copy with an identical ranking, and a duplicated axis
    would take a share of every importance attribution for nothing.

    ``top_counterparty_share`` and ``repeat_pair_mean`` measure repeated fund
    movement along the same relationship, which is what distinguishes a mule
    shuttling money to one destination from an account with genuinely diverse
    traffic.  The three ``neighbour_*`` columns are the only genuinely
    relational features in the tabular families: they summarise the entity's
    counterparties, which is the information a one-layer GNN would learn to
    aggregate.
    """
    view = event_view(request)
    if view is None:
        return _empty(request, _NETWORK_COLUMNS)

    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    n = view.n
    adjacency, multiplicity = _simple_adjacency(request, view)
    undirected = (adjacency + adjacency.T).astype(bool).astype(np.int32)
    undirected.setdiag(0)
    undirected.eliminate_zeros()

    txn_total = view.count_by()
    counterparty_count = np.diff(undirected.indptr).astype(np.float64)

    pair_counts = multiplicity + multiplicity.T
    top_pair = np.zeros(n, dtype=np.float64)
    if pair_counts.nnz:
        top_pair = np.asarray(pair_counts.max(axis=1).todense()).ravel().astype(np.float64)
    pair_totals = np.asarray(pair_counts.sum(axis=1)).ravel().astype(np.float64)
    n_pairs = np.diff(pair_counts.tocsr().indptr).astype(np.float64)

    reciprocal = adjacency.multiply(adjacency.T)
    reciprocal_partners = np.diff(reciprocal.tocsr().indptr).astype(np.float64)

    n_components, membership = connected_components(undirected, directed=False)
    component_size = np.bincount(membership, minlength=n_components)[membership].astype(
        np.float64
    )
    component_edges = np.bincount(
        membership, weights=counterparty_count, minlength=n_components
    )[membership] / 2.0
    possible = component_size * (component_size - 1.0) / 2.0
    density = np.full(n, np.nan)
    ok = possible > 0
    density[ok] = component_edges[ok] / possible[ok]

    neighbour_txn_mean = view.nan_ratio(undirected @ txn_total, counterparty_count)
    neighbour_cp_mean = view.nan_ratio(undirected @ counterparty_count, counterparty_count)
    neighbour_txn_max = np.full(n, np.nan)
    if undirected.nnz:
        weighted = csr_matrix(
            (txn_total[undirected.indices], undirected.indices, undirected.indptr),
            shape=undirected.shape,
        )
        neighbour_txn_max = np.asarray(weighted.max(axis=1).todense()).ravel().astype(
            np.float64
        )
        neighbour_txn_max[counterparty_count == 0] = np.nan

    data = {
        "counterparty_count": counterparty_count,
        "counterparty_diversity": view.nan_ratio(counterparty_count, txn_total),
        "top_counterparty_share": view.nan_ratio(top_pair, pair_totals),
        "repeat_pair_mean": view.nan_ratio(pair_totals, n_pairs),
        "reciprocal_partners": reciprocal_partners,
        "reciprocal_ratio": view.nan_ratio(reciprocal_partners, counterparty_count),
        "component_size": component_size,
        "component_edge_density": density,
        "neighbour_txn_mean": neighbour_txn_mean,
        "neighbour_txn_max": neighbour_txn_max,
        "neighbour_counterparty_mean": neighbour_cp_mean,
    }
    return _frame(request, data)[_NETWORK_COLUMNS]


# ---------------------------------------------------------------------------
# aml_cycle_v1
# ---------------------------------------------------------------------------

_CYCLE_COLUMNS = [
    "two_path_through", "cycle3_count", "cycle4_count",
    "layering_events", "layering_share", "layering_amount_match_share",
]

#: Window inside which an outflow is treated as a continuation of an inflow
#: through the same entity - the "layering" hop.
_LAYERING_HOURS = 72.0
#: An outflow whose amount is within this fraction of the preceding inflow is
#: treated as the same money moving on.
_AMOUNT_MATCH_TOLERANCE = 0.10


def _aml_cycle(request: FeatureRequest) -> pd.DataFrame:
    """Circular and layering structure.

    Columns, and exactly what each counts on the simple, loop-free directed
    graph:

    ``two_path_through``
        ``in_degree * out_degree`` on the simple graph: how many ways money
        could pass *through* this entity in two hops. A pass-through position,
        not a cycle. It equals ``aml_flow_v1::fan_in * aml_flow_v1::fan_out``,
        which a linear model cannot form from those two columns and a tree
        would have to approximate with splits; that is why it is here and not
        left implicit.

    Directed 2-cycles are **not** here. They are
    ``aml_network_v1::reciprocal_partners``, which is the same count; putting
    it in both families would duplicate a column in every matrix that uses
    both.
    ``cycle3_count``
        Directed triangles through the entity, ``i -> a -> j -> i``.
    ``cycle4_count``
        Closed **walks** of length four through the entity, less the degenerate
        ``j == i`` term. It is not a count of *simple* 4-cycles: the case where
        both intermediaries are the same node (``i -> a -> j -> a -> i``) is
        included. Separating them costs another sparse product for a
        distinction nothing downstream has needed; this note is the honest
        version of the cheaper computation.
    ``layering_events`` / ``layering_share`` / ``layering_amount_match_share``
        Temporally ordered pass-through: an outflow within
        :data:`_LAYERING_HOURS` of an inflow at the same entity, and the share
        of those whose amount is within :data:`_AMOUNT_MATCH_TOLERANCE` of the
        inflow that preceded it.

    Cost note: all three cycle counts are sparse products on the *deduplicated*
    graph.  On SAML-D that graph has 887 thousand edges and ``A @ A`` has 8.0
    million non-zeros, so the whole family is a few seconds.  On a denser graph
    ``A @ A`` is the term that grows, and this family - not the others - is
    what would need a budget.
    """
    view = event_view(request)
    if view is None:
        return _empty(request, _CYCLE_COLUMNS)

    n = view.n
    adjacency, _ = _simple_adjacency(request, view)

    two_step = (adjacency @ adjacency).tocsr()
    in_degree = np.asarray(adjacency.sum(axis=0)).ravel().astype(np.float64)
    out_degree = np.asarray(adjacency.sum(axis=1)).ravel().astype(np.float64)
    cycle3 = np.asarray(two_step.multiply(adjacency.T).sum(axis=1)).ravel().astype(
        np.float64
    )
    cycle4 = np.asarray(two_step.multiply(two_step.T).sum(axis=1)).ravel().astype(
        np.float64
    )
    # i -> i two-step terms are not 4-cycles through distinct intermediaries.
    cycle4 -= two_step.diagonal().astype(np.float64) ** 2

    data: dict[str, np.ndarray] = {
        "two_path_through": in_degree * out_degree,
        "cycle3_count": cycle3,
        "cycle4_count": np.maximum(cycle4, 0.0),
    }

    gap = _preceding_inflow_gap(view)
    if gap is None:
        for column in ("layering_events", "layering_share", "layering_amount_match_share"):
            data[column] = np.full(n, np.nan)
    else:
        dated = np.isfinite(view.seconds)
        out_ent = view.src[dated]
        n_out = np.bincount(out_ent, minlength=n).astype(np.float64)
        hit = np.isfinite(gap) & (gap <= _LAYERING_HOURS)
        events = np.bincount(out_ent[hit], minlength=n).astype(np.float64)
        data["layering_events"] = events
        data["layering_share"] = view.nan_ratio(events, n_out)
        if view.amount is None:
            data["layering_amount_match_share"] = np.full(n, np.nan)
        else:
            matched = _amount_matched_layering(view, gap, hit)
            data["layering_amount_match_share"] = view.nan_ratio(
                np.bincount(out_ent[matched], minlength=n).astype(np.float64),
                np.where(events > 0, events, 0.0),
            )

    return _frame(request, data)[_CYCLE_COLUMNS]


def _amount_matched_layering(
    view: EventView, gap: np.ndarray, hit: np.ndarray
) -> np.ndarray:
    """Which layering events move an amount close to the inflow that preceded them."""
    t = view.seconds
    dated = np.isfinite(t)
    offset, span = _time_offsets(t)

    in_key = view.dst[dated] * span + offset[dated]
    order = np.argsort(in_key, kind="stable")
    in_key = in_key[order]
    in_amount = view.amount[dated][order]

    out_key = view.src[dated] * span + offset[dated]
    out_amount = view.amount[dated]
    pos = np.searchsorted(in_key, out_key, side="right") - 1

    matched = np.zeros(len(out_key), dtype=bool)
    usable = hit & (pos >= 0)
    if not usable.any():
        return matched
    reference = in_amount[pos[usable]]
    candidate = out_amount[usable]
    ok = np.isfinite(reference) & np.isfinite(candidate) & (reference > 0)
    ratio = np.zeros(len(reference))
    ratio[ok] = np.abs(candidate[ok] - reference[ok]) / reference[ok]
    matched[np.flatnonzero(usable)] = ok & (ratio <= _AMOUNT_MATCH_TOLERANCE)
    return matched


# ---------------------------------------------------------------------------
# aml_context_v1
# ---------------------------------------------------------------------------

#: Edge-attribute columns this family reads when the dataset carries them.
#: Absent columns produce absent features, never invented ones.
#: Publishers spell these differently and the family matches by exact name, so
#: every supported spelling has to be listed. A name that is absent from these
#: tuples is not a small loss: the column silently becomes all-NaN and the
#: signal is discarded without any error. IBM HI-Small was running with all six
#: context columns dead until 2026-09-06 because its ``Payment Format`` /
#: ``Payment Currency`` / ``Receiving Currency`` (space-separated) matched none
#: of the underscore spellings. :func:`_aml_context` now reports which source
#: columns it resolved, and the runner records that in the artifact.
_CONTEXT_CHANNEL_COLUMNS = (
    "Payment_type", "payment_type", "payment_channel",
    "Payment Format", "payment_format",          # IBM AMLworld
)
_CONTEXT_CURRENCY_PAIR = (
    ("Payment_currency", "Received_currency"),
    ("currency_orig", "currency_dest"),
    ("Payment Currency", "Receiving Currency"),  # IBM AMLworld
)
_CONTEXT_LOCATION_PAIR = (
    ("Sender_bank_location", "Receiver_bank_location"),
    ("payer_bank_country", "beneficiary_bank_country"),
)


def _first_present(edges: pd.DataFrame, names: tuple[str, ...]) -> Optional[str]:
    for name in names:
        if name in edges.columns:
            return name
    return None


def _first_pair(
    edges: pd.DataFrame, pairs: tuple[tuple[str, str], ...]
) -> Optional[tuple[str, str]]:
    for left, right in pairs:
        if left in edges.columns and right in edges.columns:
            return left, right
    return None


def _aml_context(request: FeatureRequest) -> pd.DataFrame:
    """Channel, currency and jurisdiction mix, where the dataset carries them.

    Columns that the dataset cannot support are returned as all-``NaN`` rather
    than omitted, so the matrix has the same shape whichever benchmark it is
    computed on and two artifacts remain column-comparable.  Which of them were
    actually populated is visible in the artifact's feature block as the count
    of non-null cells.
    """
    columns = [
        "n_payment_types", "top_payment_type_share",
        "currency_change_share", "n_currencies",
        "cross_border_share", "n_counterparty_locations",
    ]
    view = event_view(request)
    edges = request.edge_table
    if view is None or edges is None:
        return _empty(request, columns)
    resolved: dict[str, Any] = {}

    n = view.n
    data = {name: np.full(n, np.nan) for name in columns}
    ent = view.ent

    channel = _first_present(edges, _CONTEXT_CHANNEL_COLUMNS)
    resolved["channel"] = channel
    if channel is not None:
        codes = _codes(edges[channel])
        both = np.concatenate([codes, codes])
        data["n_payment_types"] = _nunique(ent, both, n)
        data["top_payment_type_share"] = _mode_share(ent, both, n)

    currencies = _first_pair(edges, _CONTEXT_CURRENCY_PAIR)
    resolved["currency"] = currencies
    if currencies is not None:
        left = _codes(edges[currencies[0]])
        right = _codes(edges[currencies[1]])
        changed = np.concatenate([left != right, left != right])
        counts = view.count_by()
        data["currency_change_share"] = view.nan_ratio(
            np.bincount(ent[changed], minlength=n).astype(np.float64), counts
        )
        own = np.concatenate([left, right])
        data["n_currencies"] = _nunique(ent, own, n)

    locations = _first_pair(edges, _CONTEXT_LOCATION_PAIR)
    resolved["location"] = locations
    if locations is not None:
        left = _codes(edges[locations[0]])
        right = _codes(edges[locations[1]])
        cross = np.concatenate([left != right, left != right])
        counts = view.count_by()
        data["cross_border_share"] = view.nan_ratio(
            np.bincount(ent[cross], minlength=n).astype(np.float64), counts
        )
        counterparty_location = np.concatenate([right, left])
        data["n_counterparty_locations"] = _nunique(ent, counterparty_location, n)

    request.cache["aml_context_v1_resolved"] = resolved
    if not any(v is not None for v in resolved.values()):
        logger.warning(
            "aml_context_v1 resolved none of its source columns on this "
            "dataset; all six of its columns will be NaN. Edge columns present: "
            "%s. Add this dataset's spelling to _CONTEXT_* rather than "
            "accepting the silent loss.",
            [str(c) for c in edges.columns],
        )
    return _frame(request, data)[columns]


def _codes(series: pd.Series) -> np.ndarray:
    """Integer codes for a categorical or text column."""
    if isinstance(series.dtype, pd.CategoricalDtype):
        return series.cat.codes.to_numpy()
    return pd.factorize(series, use_na_sentinel=True)[0]


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register_feature_set(
    FeatureSet(
        name="aml_txn_v1",
        version="1.0",
        family="transaction",
        description=(
            "Per-entity transaction volume and amount behaviour: directional "
            "counts, sums, means, dispersion, largest-transaction share and "
            "round-amount share. No timing and no structure."
        ),
        compute=_aml_txn,
        requires_graph=False,
    )
)

register_feature_set(
    FeatureSet(
        name="aml_temporal_v1",
        version="1.0",
        family="temporal",
        description=(
            "Timing behaviour: active days, span, velocity, inter-arrival mean "
            "and dispersion, burstiness, daily concentration, night and weekend "
            "share, dormant-to-active shift, recency and tenure."
        ),
        compute=_aml_temporal,
        requires_graph=False,
    )
)

register_feature_set(
    FeatureSet(
        name="aml_temporal_v2",
        version="2.0",
        family="temporal",
        description=(
            "Window-relative timing: position, span, dormancy and concentration "
            "measured on the detection window's own activity clock; wall-clock "
            "velocity, burstiness and night share unchanged; no weekend share "
            "and no dependence on a reference 'now'. See docs/r6_temporal_v2.md."
        ),
        compute=_aml_temporal_v2,
        requires_graph=False,
    )
)

register_feature_set(
    FeatureSet(
        name="aml_flow_v1",
        version="1.0",
        family="flow",
        description=(
            "Directional flow: fan-in, fan-out, gather-scatter signature, "
            "inflow/outflow imbalance, pass-through ratio, source and sink "
            "scores, and inflow-to-outflow turnaround latency."
        ),
        compute=_aml_flow,
        requires_graph=False,
    )
)

register_feature_set(
    FeatureSet(
        name="aml_network_v1",
        version="1.0",
        family="network",
        description=(
            "Counterparty structure: counterparty count (the undirected "
            "degree), diversity, repeated-pair concentration, reciprocity, "
            "connected-component size and density, and one hop of neighbour "
            "aggregation."
        ),
        compute=_aml_network,
        requires_graph=False,
    )
)

register_feature_set(
    FeatureSet(
        name="aml_cycle_v1",
        version="1.0",
        family="structure",
        description=(
            "Circularity and layering: reciprocal edges, two-paths through the "
            "entity, directed triangles, closed length-4 walks, and temporally "
            "ordered inflow-to-outflow pass-through events with amount match. "
            "Computed on the simple, loop-free directed graph."
        ),
        compute=_aml_cycle,
        requires_graph=False,
    )
)

register_feature_set(
    FeatureSet(
        name="aml_context_v1",
        version="1.0",
        family="context",
        description=(
            "Payment-channel, currency and jurisdiction mix, populated only "
            "from edge-attribute columns the dataset actually carries."
        ),
        compute=_aml_context,
        requires_graph=False,
    )
)
