"""Data preparation and numerical analysis helpers for example.py."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PreparedSeries:
    dates: pd.DatetimeIndex
    train_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex
    train: np.ndarray
    test: np.ndarray
    split: int
    folded_train: np.ndarray
    folded_test: np.ndarray
    average_train: np.ndarray


def prepare_daily_series(input_path, source_start, source_end, series_name,
                         analysis_start="1970-01-01",
                         analysis_end="2023-12-31", split_year=2000):
    """Align a raw/preprocessed daily array and create the fixed train/test split."""
    values = np.load(input_path).astype(np.float64)
    calendar = pd.date_range(source_start, source_end, freq="D")
    not_leap_day = ~((calendar.month == 2) & (calendar.day == 29))
    if len(values) == len(calendar):
        values, dates = values[not_leap_day], calendar[not_leap_day]
    elif len(values) == not_leap_day.sum():
        dates = calendar[not_leap_day]
    else:
        raise ValueError(
            f"{len(values)} values cannot be aligned with {len(calendar)} "
            f"calendar days or {not_leap_day.sum()} non-leap days")

    counts = pd.Series(1, index=dates).groupby(dates.year).sum()
    complete_years = counts[counts == 365].index
    complete = np.isin(dates.year, complete_years)
    values, dates = values[complete], dates[complete]

    keep = ((dates >= pd.Timestamp(analysis_start)) &
            (dates <= pd.Timestamp(analysis_end)))
    values, dates = values[keep], dates[keep]
    expected_days = 54 * 365
    if len(values) != expected_days:
        raise ValueError(
            f"{series_name} has {len(values)} aligned days in 1970--2023; "
            f"expected {expected_days}")

    split = dates.searchsorted(pd.Timestamp(f"{split_year}-01-01"))
    if split == len(dates) or dates[split].year != split_year:
        raise ValueError(f"split year {split_year} is not present in the data")
    if split != 30 * 365:
        raise ValueError(
            f"training period must contain 30 complete 365-day years, "
            f"got {split} days")

    train, test = values[:split], values[split:]
    folded_train = train.reshape(-1, 365)
    folded_test = test.reshape(-1, 365)
    return PreparedSeries(
        dates=dates,
        train_dates=dates[:split],
        test_dates=dates[split:],
        train=train,
        test=test,
        split=split,
        folded_train=folded_train,
        folded_test=folded_test,
        average_train=folded_train.mean(axis=0),
    )


@dataclass(frozen=True)
class AgreementAnalysis:
    names: list
    selected: np.ndarray
    agreement: np.ndarray
    jaccard: np.ndarray
    event_counts: dict


def analyze_agreement(all_bursts, scores, pot, n_test, split,
                      subsequence_length):
    """Build masks and pairwise Jaccard overlap on shared score coverage."""
    selections = {"Burst": np.zeros(n_test, dtype=bool)}
    lengths = {"Burst": n_test}
    for start, end, _ in all_bursts:
        lo, hi = max(0, start - split), min(n_test, end - split + 1)
        selections["Burst"][lo:hi] = True
    raw_selections = {"Burst": selections["Burst"].copy()}

    event_counts = {"Burst": len(all_bursts)}
    for name in scores:
        selected = np.zeros(n_test, dtype=bool)
        length = min(len(scores[name]), n_test)
        events = pot[name][2]
        raw = np.zeros(n_test, dtype=bool)
        raw[:length] = np.asarray(pot[name][1], dtype=bool)[:length]
        for index in events:
            selected[index:min(index + subsequence_length, length)] = True
        selections[name] = selected
        raw_selections[name] = raw
        lengths[name] = length
        event_counts[name] = len(events)

    names = list(selections)
    selected = np.vstack(list(selections.values()))
    agreement = selected.sum(axis=0)
    raw_selected = np.vstack([raw_selections[name] for name in names])
    jaccard = np.zeros((len(names), len(names)))
    for i, left in enumerate(raw_selected):
        for j, right in enumerate(raw_selected):
            if i == j:
                jaccard[i, j] = 1.0
                continue
            shared = min(lengths[names[i]], lengths[names[j]])
            union = np.logical_or(left[:shared], right[:shared]).sum()
            intersection = np.logical_and(left[:shared], right[:shared]).sum()
            jaccard[i, j] = intersection / max(union, 1)
    return AgreementAnalysis(names, selected, agreement, jaccard, event_counts)


def agreement_summary(series_name, test_dates, analysis, subsequence_length,
                      seed, peak_quantile):
    """Create the compact per-method agreement summary table."""
    rows = []
    for i, name in enumerate(analysis.names):
        mask = analysis.selected[i]
        dates_selected = test_dates[mask]
        others = np.delete(analysis.jaccard[i], i)
        consensus = mask & (analysis.agreement >= 2)
        rows.append({
            "series": series_name,
            "method": name,
            "subsequence_length": subsequence_length,
            "seed": seed,
            "peak_quantile": peak_quantile if name != "Burst" else np.nan,
            "event_count": analysis.event_counts[name],
            "selected_days": int(mask.sum()),
            "exclusive_days": int((mask & (analysis.agreement == 1)).sum()),
            "consensus_days": int(consensus.sum()),
            "consensus_rate": float(consensus.sum() / max(mask.sum(), 1)),
            "mean_jaccard": float(others.mean()),
            "first_selection": (dates_selected[0].date()
                                if len(dates_selected) else None),
            "last_selection": (dates_selected[-1].date()
                               if len(dates_selected) else None),
        })
    return pd.DataFrame(rows)


def selection_masks(test_dates, analysis):
    return pd.DataFrame({
        "date": np.tile(test_dates.date, len(analysis.names)),
        "method": np.repeat(analysis.names, len(test_dates)),
        "selected": analysis.selected.reshape(-1),
    })


def burst_strength(event, split, folded_train, test):
    start, end, _ = event
    index, length = start - split, end - start + 1
    days = np.arange(start, end + 1) % 365
    reference = folded_train[:, days].sum(axis=1)
    observed = test[index:index + length].sum()
    return abs((observed - reference.mean()) /
               max(reference.std(ddof=1), 1e-8))


def combine_series_summaries(output_root, series_names):
    frames = [pd.read_csv(Path(output_root) / name / "agreement_summary.csv")
              for name in series_names]
    return pd.concat(frames, ignore_index=True)


def cross_series_analysis_text(summary):
    lines = ["Cross-series anomaly analysis", ""]
    for name, group in summary.groupby("series"):
        active = group[group.selected_days > 0]
        most = active.loc[active.consensus_rate.idxmax()]
        least = active.loc[active.consensus_rate.idxmin()]
        lines += [
            f"{name}:",
            f"- strongest consensus: {most.method} "
            f"({most.consensus_rate:.1%} of selected days)",
            f"- most method-specific: {least.method} "
            f"({least.consensus_rate:.1%} confirmed)",
            "",
        ]
    return "\n".join(lines)
