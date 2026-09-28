"""Run the fixed-scale comparative anomaly-detection experiment."""

import argparse
import os
import random
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Support both ``python case_study/example.py`` and package entry points.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for import_path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

# Configure writable plotting caches before importing the plotting module.
CACHE = PROJECT_ROOT / ".cache"
CACHE.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE))

from case_study.deep import (
    reconstruct_models,
    reconstruction_error_summary,
    train_models,
)
from anomaly_detection.detect import burst, imp, matrix_profile
from case_study.analysis import (
    agreement_summary,
    analyze_agreement,
    burst_strength,
    combine_series_summaries,
    cross_series_analysis_text,
    prepare_daily_series,
    selection_masks,
)
from case_study.plotting import (
    plot_agreement_matrix,
    plot_agreement_timeline,
    plot_anomaly_scores,
    plot_burst_interpretation,
    plot_cross_series_comparison,
    plot_deep_interpretation,
    plot_discord_interpretation,
    plot_event_month_histogram,
    plot_folded_overview,
    plot_method_score,
    plot_test_anomalies,
    save_figure,
)
from case_study.pot import (
    agreement as peak_agreement,
    analyze_scores,
    cluster_points as event_cluster_points,
    consensus_events,
    diagnostics as peak_diagnostics,
    month_histogram,
    strongest_events,
)


SERIES = {
    "temperature": (PROJECT_ROOT / "data/temperature_time_series.npy",
                    "1950-01-01", "2023-12-31"),
    "precipitation": (PROJECT_ROOT / "data/preci_time_series.npy",
                      "1970-01-01", "2023-12-31"),
    "pressure": (PROJECT_ROOT / "data/pressure_time_series.npy",
                 "1950-01-01", "2025-09-30"),
}
OUTPUT_ROOT = PROJECT_ROOT / "outputs/figures"
SUBSEQUENCE_LENGTH = 15
INTERPRETATION_CAP = 5


def plotting_units(series_name, data, reconstructions):
    """Return plotting-only data in conventional meteorological units."""
    if series_name == "temperature":
        transform = lambda values: np.asarray(values, dtype=float) - 273.15
        value_label = r"temperature ($^\circ$C)"
    elif series_name == "pressure":
        transform = lambda values: np.asarray(values, dtype=float) / 100.0
        value_label = "pressure (hPa)"
    else:
        transform = lambda values: np.asarray(values, dtype=float) * 1000.0
        value_label = "Total precipitation (mm)"

    plotting_data = replace(
        data,
        train=transform(data.train),
        test=transform(data.test),
        folded_train=transform(data.folded_train),
        folded_test=transform(data.folded_test),
        average_train=transform(data.average_train),
    )
    plotting_reconstructions = {
        name: transform(values) for name, values in reconstructions.items()
    }
    return plotting_data, plotting_reconstructions, value_label


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--series", choices=SERIES)
    parser.add_argument("--peak-quantile", type=float, default=.99,
                        help="per-detector percentile used to extract peaks")
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed for stochastic detectors and deep models")
    args = parser.parse_args()
    if not 0 < args.peak_quantile < 1:
        parser.error("--peak-quantile must be between 0 and 1")
    return args


def run_all_series(args):
    """Run each series in an isolated process, then combine their summaries."""
    for name in SERIES:
        print(f"Running series {name}")
        subprocess.run([
            sys.executable, str(Path(__file__).resolve()), "--series", name,
            "--seed", str(args.seed),
            "--peak-quantile", str(args.peak_quantile),
        ], check=True)

    summary = combine_series_summaries(OUTPUT_ROOT, SERIES)
    summary.to_csv(OUTPUT_ROOT / "cross_series_summary.csv", index=False)
    fig = plot_cross_series_comparison(summary)
    save_figure(fig, OUTPUT_ROOT, "cross_series", "method_comparison")
    analysis = cross_series_analysis_text(summary)
    (OUTPUT_ROOT / "cross_series" / "analysis.txt").write_text(analysis)


def classical_detection(train, test, m):
    """Run the active classical subsequence detectors."""
    print(f"Running classical detection with subsequence length {m}")
    print(f"Running IMP")
    imp_score, imp_indices = imp(train, test, m=m, interval_length=30)
    print(f"Running MP")
    mp_score, mp_indices = matrix_profile(train, test, m=m, normalize=True)
    print(f"Running AAMP")
    aamp_score, aamp_indices = matrix_profile(train, test, m=m, normalize=False)
    scores = {
        "IMP": imp_score,
        "MP": mp_score,
        "AAMP": aamp_score,
    }
    neighbours = {
        "IMP": (imp_score, imp_indices, "IMP"),
        "MP (z-normalized)": (mp_score, mp_indices, "MP"),
        "AAMP": (aamp_score, aamp_indices, "AAMP"),
    }
    return scores, neighbours


def clear_old_interpretations(output):
    for method in ["burst", "imp", "mp", "aamp", "iforest",
                   "autoencoder", "dlinear", "timesnet"]:
        folder = output / method
        if not folder.exists():
            continue
        if method == "iforest":
            # Isolation Forest is temporarily disabled. Remove every legacy
            # figure when the series is rerun so it cannot be mistaken for a
            # result from the active detector set.
            for extension in ("png", "pdf"):
                for old in folder.glob(f"*.{extension}"):
                    old.unlink()
            continue
        for extension in ("png", "pdf"):
            for old in folder.glob(f"interpret_*.{extension}"):
                old.unlink()
            top = folder / f"top_1_anomaly.{extension}"
            if top.exists():
                top.unlink()


def plot_interpretations(output, data, all_bursts, scores, pot, neighbours,
                         reconstructions, value_label="value"):
    """Create overview and detailed interpretation figures."""
    m = SUBSEQUENCE_LENGTH
    folded_args = (data.folded_train, data.folded_test, data.average_train)
    ranked_bursts = sorted(
        all_bursts,
        key=lambda event: burst_strength(
            event, data.split, data.folded_train, data.test),
        reverse=True,
    )[:INTERPRETATION_CAP]

    fig = plot_folded_overview(
        "burst", data.test, *folded_args, data.split,
        intervals=[(start, end) for start, end, _ in all_bursts],
        value_label=value_label)
    save_figure(fig, output, "burst", "overview")
    burst_focus = None
    burst_length = 1
    if ranked_bursts:
        burst_start, burst_end, _ = ranked_bursts[0]
        burst_focus = burst_start - data.split
        burst_length = burst_end - burst_start + 1
    fig = plot_test_anomalies(
        "Burst", data.test, data.test_dates, focus_index=burst_focus,
        anomaly_length=burst_length, value_label=value_label,
        reference_years=data.folded_train, reference_label="1970--1999")
    save_figure(fig, output, "burst", "test_anomalies", layout_top=.84)

    for name in scores:
        representatives = pot[name][2]
        raw_points = np.flatnonzero(pot[name][1])
        companions = np.setdiff1d(raw_points, representatives)
        fig = plot_folded_overview(
            name, data.test, *folded_args, data.split,
            points=representatives, other_points=companions,
            value_label=value_label)
        save_figure(fig, output, name.lower(), "overview")
        strongest = strongest_events(scores[name], pot[name], 1)
        focus = int(strongest[0]) if len(strongest) else None
        fig = plot_test_anomalies(
            name, data.test, data.test_dates, focus_index=focus,
            anomaly_length=(m if name in {"IMP", "MP", "AAMP"} else 1),
            reconstruction=reconstructions.get(name),
            value_label=value_label, reference_years=data.folded_train,
            reference_label="1970--1999")
        save_figure(
            fig, output, name.lower(), "test_anomalies", layout_top=.84)
        fig = plot_method_score(name, scores[name], data.test_dates, pot[name])
        save_figure(
            fig, output, name.lower(), "score_timeseries", layout_top=.84)

    clear_old_interpretations(output)
    for rank, event in enumerate(ranked_bursts, 1):
        start, end, sign = event
        index, length = start - data.split, end - start + 1
        days = np.arange(start, end + 1) % 365
        reference = data.folded_train[:, days].sum(axis=1)
        observed = data.test[index:index + length].sum()
        strength = burst_strength(
            event, data.split, data.folded_train, data.test)
        fig = plot_burst_interpretation(
            data.test, data.test_dates, *folded_args, index, length, sign,
            strength, reference, observed, value_label=value_label)
        save_figure(fig, output, "burst", f"interpret_{rank:02d}",
                    aliases=("top_1_anomaly",) if rank == 1 else ())

    for label, (score, indices, key) in neighbours.items():
        method = label.split()[0].lower()
        events = strongest_events(score, pot[key], INTERPRETATION_CAP)
        for rank, index in enumerate(events, 1):
            neighbour = int(indices[index])
            fig = plot_discord_interpretation(
                label, data.test, data.train, data.test_dates, data.train_dates,
                *folded_args, index, neighbour, m, score[index],
                value_label=value_label)
            save_figure(fig, output, method, f"interpret_{rank:02d}",
                        aliases=("top_1_anomaly",) if rank == 1 else ())

    for name, reconstruction in reconstructions.items():
        events = strongest_events(scores[name], pot[name], INTERPRETATION_CAP)
        for rank, index in enumerate(events, 1):
            cluster = event_cluster_points(
                pot[name][1], index, SUBSEQUENCE_LENGTH)
            fig = plot_deep_interpretation(
                name, data.test, data.test_dates, reconstruction, *folded_args,
                index, scores[name][index], cluster,
                value_label=value_label)
            save_figure(fig, output, name.lower(), f"interpret_{rank:02d}",
                        aliases=("top_1_anomaly",) if rank == 1 else ())


def run_series(series_name, args):
    """Execute detection, analysis, and plotting for one configured series."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    input_path, source_start, source_end = SERIES[series_name]
    output = OUTPUT_ROOT / series_name
    output.mkdir(parents=True, exist_ok=True)
    data = prepare_daily_series(
        input_path, source_start, source_end, series_name)
    print(f"Split index: {data.split}, Split date: {data.test_dates[0]}")

    print(f"Running burst detection with subsequence length {SUBSEQUENCE_LENGTH}")
    all_bursts = burst(
        data.train, data.test, min_length=15, max_length=15, threshold=3)
    print(f"Detected {len(all_bursts)} bursts")

    scores, neighbours = classical_detection(
        data.train, data.test, SUBSEQUENCE_LENGTH)
    models, deep_scores, standardized_test, mean, std, deep_losses = train_models(
        data.train, data.test)
    pd.DataFrame.from_dict(deep_losses, orient="index").rename_axis(
        "model").to_csv(output / "deep_losses.csv")
    scores.update(deep_scores)
    print(f"Extracting peaks above quantile {args.peak_quantile:g}")
    pot = analyze_scores(
        scores, quantile=args.peak_quantile,
        decluster=SUBSEQUENCE_LENGTH)
    peak_diagnostics(
        scores, pot, index=data.test_dates,
    ).to_csv(output / "peak_diagnostics.csv")
    detector_jaccard = peak_agreement(
        pot, len(data.test))
    detector_jaccard.to_csv(output / "detector_jaccard.csv")
    consensus_events(
        pot, len(data.test), window=SUBSEQUENCE_LENGTH, min_detectors=3,
        index=data.test_dates,
    ).to_csv(output / "consensus_events.csv", index=False)
    event_months = month_histogram(
        pot, len(data.test), index=data.test_dates)
    event_months.to_csv(output / "event_month_histogram.csv")
    print(f"Agreement analysis")
    agreement = analyze_agreement(
        all_bursts, scores, pot, len(data.test), data.split,
        SUBSEQUENCE_LENGTH)

    print(f"Plotting anomaly scores and agreement figures")
    fig = plot_anomaly_scores(
        scores, pot, data.test_dates, args.peak_quantile)
    save_figure(fig, output, "comparison", "anomaly_scores")
    fig = plot_agreement_timeline(
        data.test_dates, agreement.selected, agreement.names)
    save_figure(fig, output, "comparison", "agreement_timeline")
    fig = plot_agreement_matrix(agreement.jaccard, agreement.names)
    save_figure(fig, output, "comparison", "agreement_matrix")
    if len(detector_jaccard):
        fig = plot_agreement_matrix(
            detector_jaccard.to_numpy(), list(detector_jaccard.index))
        save_figure(fig, output, "comparison", "peak_jaccard")
    fig = plot_event_month_histogram(event_months)
    save_figure(fig, output, "comparison", "event_month_histogram")

    summary = agreement_summary(
        series_name, data.test_dates, agreement, SUBSEQUENCE_LENGTH,
        args.seed, args.peak_quantile)
    summary.to_csv(output / "agreement_summary.csv", index=False)
    selection_masks(data.test_dates, agreement).to_csv(
        output / "selection_masks.csv", index=False)

    reconstructions = reconstruct_models(
        models, standardized_test, mean, std)
    pd.DataFrame(reconstructions, index=data.test_dates).rename_axis(
        "date").to_csv(output / "deep_reconstructions.csv")
    pd.DataFrame.from_dict(
        reconstruction_error_summary(data.test, reconstructions),
        orient="index",
    ).rename_axis("model").to_csv(output / "deep_reconstruction_summary.csv")
    plotting_data, plotting_reconstructions, value_label = plotting_units(
        series_name, data, reconstructions)
    plot_interpretations(
        output, plotting_data, all_bursts, scores, pot, neighbours,
        plotting_reconstructions, value_label=value_label)


def main():
    args = parse_args()
    if args.series is None:
        run_all_series(args)
    else:
        run_series(args.series, args)


if __name__ == "__main__":
    main()
