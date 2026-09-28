"""
detect.py -- give it a time series, get anomaly scores from every detector.

    python detect.py --input tas.npy --start 1950-01-01 --frequency D --split_year 2000

    # or from Python
    import detect
    s = detect.load_series("tas.npy", "1950-01-01", frequency="D")
    scores = detect.run(s, split_year=2000)

Every detector is a plain function (train, test) -> one score per test day,
higher = more anomalous. Statistics come from `train` only; test subsequences
are compared only to the training period (outer join).
"""

import argparse
import importlib
import os
import sys

import numpy as np
import pandas as pd

from .utils import burst_compute

# disable gpu (stumpy crashed)  (comment if gpu is needed)


sys.path.insert(0, '../MP_CPP/build/src/')
try:
    libimp = importlib.import_module("libimp")
except ImportError as exc:                     # keep the script usable without
    libimp = None                              # the C++ build
    print(f"[warn] libimp unavailable ({exc}); the 'imp' detector is disabled")


# --------------------------------------------------------------------------
# input
# --------------------------------------------------------------------------


def load_series(path, start, end, frequency="D"):
    """Load a .npy array and attach a DatetimeIndex.

    `end` is optional: with `start` and `frequency` the index is inferred from
    the array length, which avoids silent off-by-one mismatches. If `end` is
    given it is checked against the data.
    """
    v = np.load(path)
    v = np.squeeze(v)
    if v.ndim != 1:
        raise ValueError(f"expected a 1-D series, got shape {v.shape}")

    idx = pd.date_range(start=start, end=end, freq=frequency)
    # remove leap days if the input is daily and the array length is 365 * n
    if frequency == "D" and len(v) % 365 == 0 and len(idx) != len(v):
        idx = idx[~((idx.month == 2) & (idx.day == 29))]
    if len(idx) != len(v):
        raise ValueError(f"index length {len(idx)} != array length {len(v)}")

    return pd.Series(v.astype(np.float64), index=idx, name="value")


def samples_per_day(frequency):
    """How many samples make up one day, so window sizes stay in real time
    units when the series is hourly rather than daily."""
    probe = pd.date_range("2001-01-01", periods=2, freq=frequency)
    step = probe[1] - probe[0]
    return max(1, int(round(pd.Timedelta("1D") / step)))


# --------------------------------------------------------------------------
# seasonal climatology (fitted on train only)
# --------------------------------------------------------------------------


def _doy(index):
    """Day of year, with Feb 29 collapsed so the climatology has length 365."""
    d = np.asarray(index.dayofyear).copy()
    leap = np.asarray(index.is_leap_year)
    d[leap & (d > 59)] -= 1
    return np.clip(d, 1, 365) - 1


def climatology(train, window=15):
    """Mean and sd for each calendar day, pooled over +/- `window` days."""
    doy, v = _doy(train.index), train.to_numpy(float)
    mean, std = np.full(365, np.nan), np.full(365, np.nan)
    for d in range(365):
        off = np.abs(((doy - d + 182) % 365) - 182)
        sel = v[(off <= window) & np.isfinite(v)]
        if sel.size >= 10:
            mean[d], std[d] = sel.mean(), sel.std(ddof=1)

    def fill(a): return pd.Series(a).interpolate(
        limit_direction="both").to_numpy()
    return fill(mean), np.maximum(fill(std), 1e-6)


def zscore(x, clim):
    mean, std = clim
    d = _doy(x.index)
    return (x.to_numpy(float) - mean[d]) / std[d]


def _windows(v, m):
    return np.lib.stride_tricks.sliding_window_view(np.nan_to_num(v), m)


def _spread(win_scores, m, n):
    """Give each day the max score of any window covering it."""
    out = np.full(n, -np.inf)
    for i, val in enumerate(win_scores):
        out[i:i + m] = np.maximum(out[i:i + m], val)
    return out



# --------------------------------------------------------------------------
# detectors: (train, test) -> score per test day
# --------------------------------------------------------------------------


def doy_zscore(train, test):
    """Baseline: how unusual is this day for this time of year."""
    return zscore(test, climatology(train))


def burst(train, test, min_length=3, max_length=30, threshold=3.0, period_length=365):
    """Seasonal burst 
    returns a list of (start, end, z_score) tuples for each detected burst.
    """
    ts = np.concatenate((train, test))
    split = len(train) // period_length
    n_periods = len(ts) // period_length

    print(f"split={split}, n_periods={n_periods}, period_length={period_length}, ")
    all_bursts = burst_compute(
        ts, n_periods, threshold, period_length, min_length, max_length, split)
    return all_bursts


def imp(train, test, m, interval_length, period_length=365):
    """Interval Matrix Profile (C++ implementation).

    retruns the score and the indice of the closest match 
    """
    if libimp is None:
        raise RuntimeError("libimp not importable")

    train_period_starts = [i for i in range(0, len(train), 365)]
    test_period_starts = [i for i in range(0, len(test), 365)]

    score, indices = libimp.outer_BIMP_double(
        query_series=test, target_series=train, window_size=m,
        query_period_starts=test_period_starts,
        target_period_starts=train_period_starts,
        interval_length=interval_length)
    return score, indices


def matrix_profile(train, test, m=15, normalize=False):
    """Matrix Profile (stumpy implementation).
    retruns the score and the indice of the closest match
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import stumpy
    res = stumpy.stump(test, m, train, ignore_trivial=False, normalize=normalize)
    score, indices = res[:, 0], res[:, 1]
    # reactivate GPU for the rest of the script
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    return score, indices


def iforest(train, test, m=15, seed=0):
    """Isolation Forest on windows of deseasonalized values."""
    from sklearn.ensemble import IsolationForest
    # clim = climatology(train)
    model = IsolationForest(n_estimators=300, random_state=seed)
    model.fit(_windows(train, m))
    return -model.score_samples(_windows(test, m))


def autoencoder(train, test, m=32, hidden=8, seed=0):
    """Reconstruction error of a bottleneck network trained on the past."""
    from sklearn.neural_network import MLPRegressor
    clim = climatology(train)
    Xtr = _windows(zscore(train, clim), m)
    net = MLPRegressor(hidden_layer_sizes=(32, hidden, 32), max_iter=300,
                       random_state=seed, early_stopping=True).fit(Xtr, Xtr)
    Xte = _windows(zscore(test, clim), m)
    return ((net.predict(Xte) - Xte) ** 2).mean(axis=1)


def build_detectors(spd=1, m_days=15, ae_days=32, interval_days=30,
                    burst_threshold=4.5):
    """Detector registry with window sizes given in *days* and converted to
    samples, so the same settings work for daily and hourly input."""
    m, ae_m, iv = m_days * spd, ae_days * spd, interval_days * spd

    def burst_flags(tr, te):
        intervals = burst(
            tr,
            te,
            min_length=3 * spd,
            max_length=30 * spd,
            threshold=burst_threshold,
            period_length=365 * spd,
        )
        scores = np.zeros(len(te), dtype=float)
        offset = len(tr)
        for start, end, _ in intervals:
            if end < offset or start >= offset + len(te):
                continue
            lo = max(start, offset) - offset
            hi = min(end, offset + len(te) - 1) - offset
            scores[lo:hi + 1] = 1.0
        return scores

    return {
        "doy_zscore": doy_zscore,
        "burst": burst_flags,
        "mp_raw": lambda tr, te: matrix_profile(
            tr, te, m=m, normalize=False)[0],
        "mp_znorm": lambda tr, te: matrix_profile(
            tr, te, m=m, normalize=True)[0],
        "imp": lambda tr, te: imp(
            tr, te, m=m, interval_length=iv)[0],
        "iforest": lambda tr, te: iforest(tr, te, m=m),
        "autoencoder": lambda tr, te: autoencoder(tr, te, m=ae_m),
    }


DETECTORS = build_detectors()


# --------------------------------------------------------------------------


def run(series, split_year=2000, detectors=None, rank=True):
    """Fit on everything before `split_year`, score everything from it onward.

    Returns a DataFrame indexed by test date. Window-level scores are spread
    back over the days each window covers, so every column has one value per
    test timestamp regardless of the detector's window size.

    `rank=True` converts each column to a percentile within the test period,
    so detectors with different score scales can be compared and thresholded
    the same way.
    """
    tr = series[series.index.year < split_year]
    te = series[series.index.year >= split_year]
    if len(tr) == 0 or len(te) == 0:
        raise ValueError(
            f"split_year={split_year} leaves an empty train or test set")

    out = {}
    for name, fn in (detectors or DETECTORS).items():
        try:
            s = np.asarray(fn(tr, te), dtype=float)
            if len(s) < len(te):                       # window-level -> per-day
                s = _spread(s, len(te) - len(s) + 1, len(te))
            elif len(s) > len(te):
                raise ValueError(
                    f"returned {len(s)} scores for {len(te)} points")
            out[name] = s
        except Exception as exc:
            print(f"[skip] {name}: {type(exc).__name__}: {exc}")

    df = pd.DataFrame(out, index=te.index)
    return df.rank(pct=True) if rank else df


def top_events(scores, column, k=10, gap=5):
    """The k most anomalous episodes for one detector, merging nearby days."""
    s = scores[column].sort_values(ascending=False)
    picked = []
    for date in s.index:
        if all(abs((date - d).days) > gap for d in picked):
            picked.append(date)
        if len(picked) == k:
            break
    return scores.loc[sorted(picked), [column]]


def demo():
    idx = pd.date_range("1950-01-01", "2025-12-31", freq="D")
    rng = np.random.default_rng(0)
    warming = 0.03 * np.arange(len(idx)) / 365.25
    s = pd.Series(12 + 9 * np.sin(2 * np.pi * (idx.dayofyear - 110) / 365.25)
                  + warming + rng.normal(0, 2.5, len(idx)), index=idx)
    sc = run(s)
    print(sc.describe().round(3))
    print("\nmost anomalous episodes (burst):")
    print(top_events(sc, "burst").round(4))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run anomaly detectors on a time series.")
    parser.add_argument("--input", help="time series in npy format")
    parser.add_argument(
        "--start", help="start date of the time series (YYYY-MM-DD)")
    parser.add_argument(
        "--end", help="end date of the time series (YYYY-MM-DD)")
    parser.add_argument("--frequency", default="D",
                        help="frequency of the time series (e.g., 'D' for daily, 'h' for hourly)")
    parser.add_argument("--split_year", type=int,
                        default=2000, help="year to split train/test")

    parser.add_argument("--output", help="write the score table to this CSV")
    args = parser.parse_args()

    series = load_series(args.input, args.start, args.end, args.frequency)
    spd = samples_per_day(args.frequency)
    registry = build_detectors(spd)

    print(f"{len(series)} points, {series.index[0].date()}..{series.index[-1].date()}, "
          f"freq={args.frequency} ({spd} sample(s)/day), split at {args.split_year}")
    scores = run(series, split_year=args.split_year, detectors=registry)

    print(scores.describe().round(3))
    for col in scores.columns:
        print(f"\ntop episodes -- {col}")
        print(top_events(scores, col, k=5).round(4))

    if args.output:
        scores.to_csv(args.output)
        print(f"\nwrote {args.output}")
