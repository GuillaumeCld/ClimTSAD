"""Peak extraction and characterisation for comparative anomaly detectors.

Extraction uses a fixed per-detector percentile plus runs declustering. This
fixes the raw exceedance rate across detectors, while the diagnostics describe
persistence, seasonal concentration, and cross-detector agreement without
requiring labels.
"""

from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

def _flag_clusters(flags, gap):
    indices = np.flatnonzero(flags)
    if len(indices) == 0:
        return []
    return np.split(indices, np.flatnonzero(np.diff(indices) > gap) + 1)


def _decluster(values, flags, gap):
    """Group exceedances at most ``gap`` apart and retain each run maximum."""
    clusters = _flag_clusters(flags, gap)
    return np.array(
        [cluster[np.argmax(values[cluster])] for cluster in clusters], dtype=int
    )


def cluster_points(flags, event, gap):
    """Return all raw exceedance positions in a declustered event's run."""
    for cluster in _flag_clusters(flags, gap):
        if event in cluster:
            return cluster
    raise ValueError(f"event {event} is not a flagged point")


def extract_peaks(values, quantile=0.99, decluster=10):
    """Return a constant threshold, raw flags, and declustered peak positions."""
    values = np.asarray(values, dtype=float)
    if not 0 < quantile < 1:
        raise ValueError("quantile must be between 0 and 1")
    if decluster < 0:
        raise ValueError("decluster must be non-negative")
    if not np.isfinite(values).any():
        raise ValueError("scores contain no finite values")
    cutoff = np.nanquantile(values, quantile)
    flags = np.isfinite(values) & (values > cutoff)
    threshold = np.full(len(values), cutoff)
    return threshold, flags, _decluster(values, flags, decluster)


def analyze_scores(scores, quantile=0.99, decluster=10):
    """Extract peaks independently from every named detector score array."""
    return {
        name: extract_peaks(score, quantile=quantile, decluster=decluster)
        for name, score in scores.items()
    }


def _index_for(values, index=None, start_date="2000-01-01"):
    if index is not None:
        result = pd.DatetimeIndex(index)
        if len(result) < len(values):
            raise ValueError(
                f"index has {len(result)} dates for {len(values)} values"
            )
        return result[:len(values)]
    return pd.date_range(start_date, periods=len(values), freq="D")



def seasonal_strength(values, period=365):
    values = np.asarray(values, float)


    y = pd.Series(values)#.interpolate(limit_direction="both")

    result = STL(y, period=365, robust=True).fit()

    denom = np.var(result.seasonal + result.resid, ddof=1)
    return 0.0 if denom == 0 else float(
        np.clip(1 - np.var(result.resid, ddof=1) / denom, 0, 1)
    )


def _indicator(events, n, window):
    indicator = np.zeros(n, dtype=bool)
    for event in events:
        indicator[max(0, event - window):min(n, event + window + 1)] = True
    return indicator


def _jaccard(left, right):
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else 0.0


def diagnostics(scores, results, index=None, start_date="2000-01-01"):
    """Characterise persistence and seasonal concentration by detector."""
    rows = []
    for name, score in scores.items():
        values = np.asarray(score, dtype=float)
        dates = _index_for(values, index, start_date)
        _, flags, events = results[name]
        n_raw = int(flags.sum())
        rows.append({
            "model": name,
            "raw": n_raw,
            "events": len(events),
            "theta": round(len(events) / n_raw, 3) if n_raw else np.nan,
            "seasonal_strength": round(
                seasonal_strength(values, dates), 3),
        })
    return pd.DataFrame(rows).set_index("model")


def month_histogram(results, n, index=None, start_date="2000-01-01"):
    """Return month-of-year event counts for every detector with events."""
    dates = _index_for(np.empty(n), index, start_date)
    output = {}
    for name, (_, _, events) in results.items():
        if len(events):
            output[name] = (
                pd.Series(dates[events].month).value_counts()
                .reindex(range(1, 13), fill_value=0).to_numpy()
            )
    return pd.DataFrame(output, index=range(1, 13)).T


def agreement(results, n=None):
    """Return raw-exceedance Jaccard on each pair's shared time span.

    Flags are compared before declustering and are not widened. Result lengths
    are inferred from their threshold arrays. When ``n`` is
    supplied for compatibility, it acts only as an upper bound. This prevents
    unavailable trailing dates from being counted as negative detections for a
    shorter score series.
    """
    names = list(results)
    lengths = {
        name: min(len(results[name][0]), n) if n is not None
        else len(results[name][0])
        for name in names
    }
    flags = {
        name: np.asarray(results[name][1], dtype=bool)[:lengths[name]]
        for name in names
    }
    matrix = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for left, right in combinations(names, 2):
        shared = min(lengths[left], lengths[right])
        matrix.loc[left, right] = matrix.loc[right, left] = round(
            _jaccard(flags[left][:shared], flags[right][:shared]), 3
        )
    return matrix


def consensus_events(results, n, window=10, min_detectors=3,
                     index=None, start_date="2000-01-01"):
    """Return intervals supported by at least ``min_detectors`` detectors."""
    names = [name for name, result in results.items() if len(result[2])]
    if not names:
        return pd.DataFrame()
    indicators = {
        name: _indicator(results[name][2], n, window) for name in names
    }
    counts = np.sum([indicators[name] for name in names], axis=0)
    dates = _index_for(np.empty(n), index, start_date)
    hot = counts >= min_detectors
    if not hot.any():
        return pd.DataFrame()

    indices = np.flatnonzero(hot)
    rows = []
    for cluster in np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1):
        peak = cluster[np.argmax(counts[cluster])]
        rows.append({
            "start": dates[cluster[0]],
            "end": dates[cluster[-1]],
            "peak": dates[peak],
            "n_detectors": int(counts[peak]),
            "detectors": ",".join(sorted(
                name for name in names if indicators[name][cluster].any()
            )),
        })
    return pd.DataFrame(rows).sort_values("n_detectors", ascending=False)


def strongest_events(score, peak_result, cap=10):
    """Rank declustered cluster representatives by threshold excess."""
    score = np.asarray(score, dtype=float)
    threshold, _, events = peak_result
    excess = score[events] - threshold[events]
    return events[np.argsort(excess)[::-1][:cap]]
