"""Plotting helpers for the reproducible batch experiment in example.py."""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FuncFormatter, MaxNLocator


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "mathtext.fontset": "cm",
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": .2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "font.size": 14,
    "axes.labelsize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.dpi": 120,
    "savefig.dpi": 300,
})


def _figure_legend(fig, axes, ncol=4, y=.99):
    """Place one deduplicated legend in a reserved band above the axes."""
    handles, labels = [], []
    for ax in np.atleast_1d(axes).ravel():
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label and label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        fig.legend(handles, labels, loc="upper center",
                   ncol=min(ncol, len(labels)), bbox_to_anchor=(.5, y))


def _concise_dates(ax, min_ticks=3, max_ticks=6):
    locator = mdates.AutoDateLocator(minticks=min_ticks, maxticks=max_ticks)
    ax.xaxis.set_major_locator(locator)
    # Numeric formats are locale-independent and safe for Matplotlib's TeX
    # wrapper (e.g. a French ``déc.`` tick otherwise fails in math mode).
    formatter = mdates.ConciseDateFormatter(
        locator,
        formats=["%Y", "%Y-%m", "%m-%d", "%m-%d", "%H:%M", "%H:%M:%S"],
        zero_formats=["", "%Y", "%Y-%m", "%m-%d", "%H:%M", "%H:%M:%S"],
        offset_formats=[
            "", "%Y", "%Y-%m", "%Y-%m-%d", "%Y-%m-%d", "%Y-%m-%d",
        ],
    )
    ax.xaxis.set_major_formatter(formatter)


def _legend_above(ax, y=.97, **kwargs):
    """Add a figure-level legend above a single axis when labels exist."""
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.figure.legend(handles, labels, loc="upper center",
                         bbox_to_anchor=(.5, y), **kwargs)


def save_figure(fig, output, method, name, aliases=(), layout_top=.80):
    """Save one figure and optional aliases as both PNG and vector PDF."""
    path = Path(output) / method
    path.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(.01, .01, .99, layout_top))
    for extension in ("png", "pdf"):
        fig.savefig(path / f"{name}.{extension}", bbox_inches="tight")
    for alias in aliases:
        for extension in ("png", "pdf"):
            fig.savefig(path / f"{alias}.{extension}", bbox_inches="tight")
    plt.close(fig)


def plot_cross_series_comparison(summary):
    summary = summary.copy()
    summary["series"] = summary["series"].replace({
        "temperature": "Temperature",
        "precipitation": "Total precipitation",
        "pressure": "Pressure",
    })
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for metric, ax in [
        ("consensus_rate", axes[0]),
        ("mean_jaccard", axes[1]),
    ]:
        summary.pivot(index="method", columns="series", values=metric).plot.bar(
            ax=ax)
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=45)
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    _figure_legend(fig, axes, ncol=3)
    return fig


def plot_anomaly_scores(scores, pot, test_dates, quantile):
    n = len(scores)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 2.6 * nrows),
                             sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, (name, score) in zip(axes, scores.items()):
        score = np.asarray(score, dtype=float)
        threshold, flags, events = pot[name]
        points = np.flatnonzero(flags)
        ax.plot(test_dates[:len(score)], score, color="#24476b", lw=.9,
                label="score")
        ax.plot(test_dates[:len(score)], threshold, color="#222222",
                ls="--", lw=1.2, label="global percentile threshold")
        ax.scatter(test_dates[points], score[points], color="#e69f00",
                   marker="x", s=18, linewidths=.9, zorder=3,
                   label="raw abnormal points")
        ax.scatter(test_dates[events], score[events], facecolors="none",
                   edgecolors="#b2182b", marker="D", s=38, linewidths=1.2,
                   zorder=4, label="declustered representatives")
        ax.text(0, 1.025, name, transform=ax.transAxes, ha="left", va="bottom",
                fontweight="bold")
        ax.set_ylabel("score")
        _concise_dates(ax)
    for ax in axes[n:]:
        ax.set_visible(False)
    axes[n - 1].set_xlabel("evaluation date")
    _figure_legend(fig, axes[:n], ncol=4, y=.90)
    return fig


def plot_agreement_timeline(test_dates, selected, names):
    agreement = selected.sum(axis=0)
    raster_height = max(1, .28 * len(names))
    fig, axes = plt.subplots(
        2, 1, figsize=(15, max(7, 4.5 + raster_height)), sharex=True,
        gridspec_kw={"height_ratios": [1.5, raster_height], "hspace": .08},
    )
    interval_starts = mdates.date2num(test_dates[:-1])
    interval_ends = mdates.date2num(test_dates[1:])
    interval_widths = np.maximum(interval_ends - interval_starts, 1e-6)
    axes[0].bar(
        interval_starts,
        agreement[:-1],
        width=interval_widths,
        align="edge",
        color="#222222",
        alpha=.45,
        edgecolor="none",
        linewidth=0,
    )
    # axes[0].fill_between(
    #     test_dates, 0, agreement, where=agreement == 1, step="mid",
    #     color="#e69f00", alpha=.45, label="one method",
    # )
    # axes[0].fill_between(
    #     test_dates, 0, agreement, where=agreement >= 2, step="mid",
    #     color="#009e73", alpha=.55, label="two or more methods",
    # )
    upper = max(1, int(agreement.max(initial=0)))
    axes[0].set_ylim(0, upper + .35)
    axes[0].yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].set_ylabel("Number of agreements")
    # _legend_above(axes[0], ncol=3)
    date_edges = np.r_[mdates.date2num(test_dates),
                       mdates.date2num(test_dates[-1] + pd.Timedelta(days=1))]
    axes[1].pcolormesh(
        date_edges, np.arange(len(names) + 1) - .5, selected.astype(int),
        cmap=ListedColormap(["#f3f3f3", "#b2182b"]), shading="flat",
        vmin=0, vmax=1,
    )
    axes[1].hlines(np.arange(len(names) + 1) - .5, date_edges[0],
                   date_edges[-1], color="white", lw=.8)
    axes[1].invert_yaxis()
    axes[1].set_yticks(range(len(names)), names)
    axes[1].set_xlabel("evaluation date")
    # axes[1].set_xlabel("date")
    axes[1].xaxis_date()
    axes[1].grid(False)
    _concise_dates(axes[1])
    return fig


def plot_agreement_matrix(jaccard, names):
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(jaccard, vmin=0, vmax=1, cmap="YlGnBu")
    ax.grid(False)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    fig.colorbar(image, ax=ax, label="Jaccard agreement")
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{jaccard[i, j]:.2f}", ha="center", va="center",
                    color="white" if jaccard[i, j] > .55 else "black")
    return fig


def plot_event_month_histogram(histogram):
    """Plot grouped monthly event-count bins for every detector."""
    fig, ax = plt.subplots(figsize=(15, 7))
    month_positions = np.arange(12)
    month_labels = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    n_methods = max(1, len(histogram))
    width = .82 / n_methods
    colors = plt.get_cmap("tab10")
    for number, (name, counts) in enumerate(histogram.iterrows()):
        offset = (number - (n_methods - 1) / 2) * width
        ax.bar(month_positions + offset, counts.to_numpy(), width=width,
               color=colors(number % 10), edgecolor="white", linewidth=.5,
               label=name)
    ax.set_xticks(month_positions, month_labels)
    ax.set_xlabel("month")
    ax.set_ylabel("declustered events")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(-.55, 11.55)
    if len(histogram):
        _legend_above(ax, ncol=min(7, len(histogram)))
    return fig


def folded_window(ax, data, i, length, folded_train, folded_test,
                  average_train, pad=20, color="tab:red", focus="anomaly",
                  cohort="train", normalize=False):
    def transform(values):
        values = np.asarray(values, dtype=float)
        if not normalize:
            return values
        scale = values.std()
        if scale <= np.finfo(float).eps:
            return np.zeros_like(values)
        return (values - values.mean()) / scale

    offsets = np.arange(-pad, length + pad)
    x = i % 365 + offsets
    days = x % 365
    years = folded_train if cohort == "train" else folded_test
    shade = "#c9dff0" if cohort == "train" else "#dddddd"
    period_name = "reference" if cohort == "train" else "evaluation"
    for n, year in enumerate(years):
        ax.plot(x, transform(year[days]), color=shade, lw=.7,
                label=f"{period_name} years" if n == 0 else None)
    ax.plot(x, transform(average_train[days]), color="black", lw=1.5,
            label="reference mean")
    source = i + offsets
    valid = (source >= 0) & (source < len(data))
    ax.plot(x[valid], transform(data[source[valid]]), color="#888888", lw=2,
            label=f"{focus} year")
    n = min(length, len(data) - i)
    if n == 1:
        ax.scatter([i % 365], transform(data[i:i + 1]), color=color, s=45,
                   zorder=4,
                   label=focus)
    else:
        ax.plot(i % 365 + np.arange(n), transform(data[i:i + n]),
                color=color, lw=2.5,
                label=focus)
    ax.set_xlabel("day of year (unwrapped)")


def folded_pair(axes, data, i, length, folded_train, folded_test,
                average_train, color="tab:red", focus="anomaly"):
    folded_window(axes[0], data, i, length, folded_train, folded_test,
                  average_train, color=color, focus=focus, cohort="train")
    folded_window(axes[1], data, i, length, folded_train, folded_test,
                  average_train, color=color, focus=focus, cohort="test")


def mark_cluster_companions(axes, data, representative, cluster_points):
    """Mark non-representative raw points from the selected runs cluster."""
    companions = np.asarray(cluster_points, dtype=int)
    companions = companions[companions != representative]
    if not len(companions):
        return
    x = representative % 365 + companions - representative
    for ax in axes:
        ax.scatter(x, data[companions], color="tab:orange", marker="x", s=50,
                   zorder=5, label="same-cluster abnormal point")


def plot_folded_overview(method, test_ts, folded_train, folded_test,
                         average_train, split, points=None, other_points=None,
                         intervals=None, value_label="value"):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    windows = intervals if intervals is not None else []
    for ax, years, cohort, shade in [
        (axes[0], folded_train, "reference", "#c9dff0"),
        (axes[1], folded_test, "evaluation", "#dddddd"),
    ]:
        for n, year in enumerate(years):
            ax.plot(range(365), year, color=shade, lw=.7,
                    label=f"{cohort} years" if n == 0 else None)
        ax.plot(range(365), average_train, color="black", lw=2,
                label="reference mean")
        for start, end in windows:
            i = start - split if intervals is not None else start
            length = end - start + 1
            segment = test_ts[i:min(i + length, len(test_ts))]
            ax.plot(i % 365 + np.arange(len(segment)), segment, lw=2)
        if points is not None and len(points):
            points = np.asarray(points, dtype=int)
            ax.scatter(points % 365, test_ts[points], color="tab:red", s=24,
                       zorder=4, label="cluster representatives")
        if other_points is not None and len(other_points):
            other_points = np.asarray(other_points, dtype=int)
            ax.scatter(other_points % 365, test_ts[other_points],
                       color="tab:orange", marker="x", s=30, zorder=4,
                       label="same-cluster abnormal points")
        ax.set_xlabel("day of year")
    axes[0].set_ylabel(value_label)
    _figure_legend(fig, axes, ncol=4)
    return fig


def plot_test_anomalies(method, test_ts, test_dates, focus_index=None,
                        anomaly_length=1, reconstruction=None,
                        context_days=30, value_label="value",
                        reference_years=None,
                        reference_label="other evaluation years"):
    """Show the top-1 test anomaly in a zoomed test-only folded view."""
    values = np.asarray(test_ts, dtype=float)
    dates = pd.DatetimeIndex(test_dates)
    if len(values) != len(dates):
        raise ValueError("test_ts and test_dates must have the same length")
    if len(values) % 365:
        raise ValueError("the folded test view requires complete 365-day years")
    if anomaly_length < 1:
        raise ValueError("anomaly_length must be positive")
    if context_days < 0:
        raise ValueError("context_days cannot be negative")

    folded = values.reshape(-1, 365)
    reference = (folded if reference_years is None else
                 np.asarray(reference_years, dtype=float))
    if reference.ndim != 2 or reference.shape[1] != 365:
        raise ValueError("reference_years must contain complete 365-day years")
    # A compact aspect ratio keeps text legible when three top-1 panels are
    # placed side by side at roughly one third of a paper's text width.
    fig, ax = plt.subplots(figsize=(5.4, 4.6))

    if focus_index is not None:
        focus_index = int(focus_index)
        if not 0 <= focus_index < len(values):
            raise ValueError("focus_index lies outside the test series")
        focus_year, focus_day = divmod(focus_index, 365)
        offsets = np.arange(-context_days, anomaly_length + context_days)
        folded_days = focus_day + offsets
        calendar_days = folded_days % 365
        for year_number, year in enumerate(reference):
            ax.plot(folded_days, year[calendar_days], color="#d0d0d0", lw=1,
                    label=reference_label if year_number == 0 else None)

        year_label = str(dates[focus_year * 365].year)
        source = focus_index + offsets
        valid = (source >= 0) & (source < len(values))
        ax.plot(folded_days[valid], values[source[valid]], color="#4d4d4d",
                lw=2.2, label=year_label)

        if reconstruction is not None:
            reconstructed = np.asarray(reconstruction, dtype=float)
            if len(reconstructed) != len(values):
                raise ValueError(
                    "reconstruction and test_ts must have the same length")
            ax.plot(folded_days[valid], reconstructed[source[valid]],
                    color="#2166ac", lw=2.2, ls="--",
                    label="reconstruction")

        stop = min(focus_index + anomaly_length, len(values))
        anomaly = values[focus_index:stop]
        anomaly_days = focus_day + np.arange(len(anomaly))
        event_date = dates[focus_index].strftime("%Y-%m-%d")
        if method == "Burst":
            anomaly_label = "Burst"
        elif method in {"IMP", "MP", "AAMP"}:
            anomaly_label = (
                f"Discord")
        else:
            anomaly_label = f"Point anomaly"

        if len(anomaly) == 1:
            ax.scatter(anomaly_days, anomaly, color="#b2182b",
                       edgecolor="white", linewidth=.8, s=85, zorder=5,
                       label=anomaly_label)
        else:
            ax.plot(anomaly_days, anomaly, color="#b2182b", lw=3.2,
                    marker="o", ms=4, zorder=5, label=anomaly_label)

        ax.set_xlim(folded_days[0], folded_days[-1])
    else:
        days = np.arange(365)
        for year_number, year in enumerate(reference):
            ax.plot(days, year, color="#d0d0d0", lw=1,
                    label=reference_label if year_number == 0 else None)
        ax.set_xlim(0, 364)

    reference_year = pd.Timestamp("2001-01-01")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda day, _: (
            reference_year + pd.Timedelta(days=int(round(day)) % 365)
        ).strftime("%d/%m")
    ))
    ax.set_xlabel("Date", fontsize=18)
    ax.set_ylabel(value_label, fontsize=18)
    ax.tick_params(axis="both", labelsize=15)
    _legend_above(
        ax, y=.94, ncol=2, fontsize=14, handlelength=1.2, columnspacing=.9,
        labelspacing=.3, borderaxespad=.1,
    )
    fig.tight_layout()
    return fig


def plot_method_score(method, score, test_dates, peak_result):
    """Plot one detector's native-scale score over its
      available test dates."""
    values = np.asarray(score, dtype=float)
    dates = pd.DatetimeIndex(test_dates)
    if len(dates) < len(values):
        raise ValueError("test_dates cannot be shorter than the score series")
    dates = dates[:len(values)]

    threshold, flags, representatives = peak_result
    threshold = np.asarray(threshold, dtype=float)
    flags = np.asarray(flags, dtype=bool)
    if len(threshold) != len(values) or len(flags) != len(values):
        raise ValueError("score and peak-result arrays must have the same length")
    representatives = np.asarray(representatives, dtype=int)
    representatives = representatives[
        (representatives >= 0) & (representatives < len(values))
    ]
    raw = np.flatnonzero(flags)

    # Compact dimensions and larger local typography remain legible when two
    # score panels share one paper row.
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    ax.plot(dates, values, color="#3f3f3f", lw=1.15, alpha=.95,
            label=f"{method} score")
    finite_threshold = np.isfinite(threshold)
    if finite_threshold.any():
        ax.plot(dates[finite_threshold], threshold[finite_threshold],
                color="#2166ac", lw=2, ls="--",
                label="99th percentile")
    if len(raw):
        ax.scatter(dates[raw], values[raw], color="#e08214", marker="x",
                   s=38, linewidths=1.4, zorder=4,
                   label="anomalies")
    if len(representatives):
        ax.scatter(dates[representatives], values[representatives],
                   color="#b2182b", edgecolor="white", linewidth=.5,
                   s=58, zorder=5,
                   label="events")

    # ax.set_xlabel("evaluation date", fontsize=18)
    ax.set_ylabel("Anomaly score", fontsize=18)
    ax.tick_params(axis="both", labelsize=14)
    _concise_dates(ax, min_ticks=3, max_ticks=5)
    _legend_above(
        ax, y=.94, ncol=4, fontsize=12, handlelength=1.2, columnspacing=.7,
        labelspacing=.25, borderaxespad=.1,
    )
    fig.tight_layout(rect=(.01, .01, .99, .76))
    return fig


def plot_burst_interpretation(test_ts, test_dates, folded_train, folded_test,
                              average_train, i, length, sign, strength,
                              train_aggregates, observed,
                              value_label="value"):
    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    folded_pair(axes[:2], test_ts, i, length, folded_train, folded_test,
                average_train)
    axes[2].hist(train_aggregates, bins=15, color="lightgray")
    axes[2].hist([observed], bins=1, color="tab:red",
                 label=f"observed = {observed:.2f}")
    axes[2].set_xlabel("subsequence aggregate")
    axes[2].set_ylabel("count")
    axes[0].set_ylabel(value_label)
    _figure_legend(fig, axes, ncol=5)
    return fig


def plot_discord_interpretation(name, test_ts, train_ts, test_dates,
                                train_dates, folded_train, folded_test,
                                average_train, i, j, m, score,
                                value_label="value"):
    """Plot a cluster-peak discord and its nearest training neighbour."""
    discord = np.asarray(test_ts[i:i + m], dtype=float)
    neighbour = np.asarray(train_ts[j:j + m], dtype=float)
    if len(discord) != m or len(neighbour) != m:
        raise ValueError("discord and neighbour must both contain m values")

    is_mp = "z-normalized" in name.lower()

    def z_normalize(values):
        scale = values.std()
        return ((values - values.mean()) / scale
                if scale > np.finfo(float).eps else np.zeros_like(values))

    comparison_discord = z_normalize(discord) if is_mp else discord
    comparison_neighbour = z_normalize(neighbour) if is_mp else neighbour

    # MP is displayed in physical units in the folded panels. Only its direct
    # shape comparison is z-normalized, so that panel needs an independent
    # vertical scale. Other discord methods continue to share all three axes.
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=not is_mp)
    folded_window(
        axes[0], test_ts, i, m, folded_train, folded_test, average_train,
        color="#b2182b", focus="evaluation discord", cohort="test",
    )
    folded_window(
        axes[1], train_ts, j, m, folded_train, folded_test, average_train,
        color="#2166ac", focus="reference neighbour", cohort="train",
    )
    if is_mp:
        folded_limits = [axes[k].get_ylim() for k in (0, 1)]
        shared_limits = (
            min(limits[0] for limits in folded_limits),
            max(limits[1] for limits in folded_limits),
        )
        axes[0].set_ylim(shared_limits)
        axes[1].set_ylim(shared_limits)

    days = np.arange(m)
    axes[2].plot(days, comparison_discord, color="#b2182b", lw=2,
                 marker="o", ms=3.5, label="evaluation discord")
    axes[2].plot(days, comparison_neighbour, color="#2166ac", lw=2, ls="--",
                 marker="o", ms=3.5, label="reference neighbour")

    panel_labels = [
        f"{name}: discord within evaluation years",
        "nearest neighbour within reference years",
        (f"z-normalized comparison (score={score:.2f})" if is_mp else
         f"side-by-side comparison (score={score:.2f})"),
    ]
    for ax, label in zip(axes, panel_labels):
        ax.text(0, 1.025, label, transform=ax.transAxes, ha="left",
                va="bottom", fontweight="bold")
    axes[0].set_ylabel(value_label)
    axes[0].set_xlabel("day of year (unwrapped)")
    axes[1].set_xlabel("day of year (unwrapped)")
    axes[2].set_xlabel("day within subsequence")
    if is_mp:
        axes[2].set_ylabel("z-normalized value")
    _figure_legend(fig, axes, ncol=4)
    return fig


def plot_iforest_interpretation(test_ts, test_dates, folded_train, folded_test,
                                average_train, i, score, cluster_points):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    folded_pair(axes, test_ts, i, 1, folded_train, folded_test, average_train)
    mark_cluster_companions(axes, test_ts, i, cluster_points)
    _figure_legend(fig, axes, ncol=4)
    return fig


def plot_deep_interpretation(name, test_ts, test_dates, reconstruction,
                             folded_train, folded_test, average_train,
                             i, score, cluster_points, value_label="value"):
    cluster_points = np.asarray(cluster_points, dtype=int)
    lo = max(0, min(i, cluster_points.min()) - 20)
    hi = min(len(test_ts), max(i, cluster_points.max()) + 21)
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    folded_pair(axes[:2], test_ts, i, 1, folded_train, folded_test,
                average_train)
    mark_cluster_companions(axes[:2], test_ts, i, cluster_points)
    axes[2].plot(test_dates[lo:hi], test_ts[lo:hi], color="tab:red",
                 label="original")
    axes[2].plot(test_dates[lo:hi], reconstruction[lo:hi], color="tab:blue",
                 ls="--", label="reconstruction")
    axes[2].scatter([test_dates[i]], [test_ts[i]], color="tab:red", s=50,
                    zorder=4, label="abnormal point")
    companions = cluster_points[cluster_points != i]
    if len(companions):
        axes[2].scatter(test_dates[companions], test_ts[companions],
                        color="tab:orange", marker="x", s=50, zorder=4,
                        label="same-cluster abnormal point")
    axes[0].set_ylabel(value_label)
    axes[2].set_ylabel(value_label)
    _concise_dates(axes[2])
    _figure_legend(fig, axes, ncol=5)
    return fig
