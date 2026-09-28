"""
plots.py -- interactive (zoomable) views of what each detector flagged.

Only the test period is drawn. The training period contributes just one thing:
the seasonal average, shown as a reference line in the folded view.

From the command line, reading the CSV that detect.py wrote:

    python detect.py --input data/med.npy --start 1960-01-01 --end 2023-12-31 \
                     --frequency D --split_year 2000 --output outputs/scores.csv
    python plots.py  --scores outputs/scores.csv --input data/med.npy \
                     --start 1960-01-01 --end 2023-12-31 --frequency D --outdir figs

Writes standalone .html files: drag to zoom, double-click to reset, click a
legend entry to hide a method.

The command-line interface supports a quantile ``q`` for backwards
compatibility. Callers can also supply a criterion per detector, such as a
burst threshold or top-K events with a minimum separation.
"""

import argparse
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .detect import _doy, climatology, load_series

PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
           "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f"]
METHOD_LABELS = {
    "burst": "Seasonal burst",
    "imp": "Interval Matrix Profile",
    "mp_znorm": "Z-normalized Matrix Profile",
    "mp_raw": "Raw Matrix Profile (AAMP)",
    "iforest": "Isolation Forest",
    "deep_autoencoder": "Deep AutoEncoder",
    "deep_dlinear": "Deep DLinear",
    "deep_timesnet": "Deep TimesNet",
}

MONTH_TICKS = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# --------------------------------------------------------------------------


def load_scores(path):
    """Read the score table written by `detect.py --output`."""
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index)
    return df


def flag(scores, q=0.99):
    """Return the timestamps selected by each method's criterion.

    ``q`` remains supported as a scalar quantile for the command-line tools.
    A mapping can specify one explicit criterion per method:
    ``positive`` for detector-produced flags, ``threshold`` for an absolute
    score cutoff, or ``top_k_separated`` for the example.py event selection.
    """
    if not isinstance(q, dict):
        return scores.ge(scores.quantile(q), axis=1)

    out = pd.DataFrame(False, index=scores.index, columns=scores.columns)
    for method in scores.columns:
        criterion = q.get(method)
        if criterion is None:
            raise ValueError(f"missing detection criterion for '{method}'")

        values = scores[method].to_numpy(dtype=float)
        kind = criterion["kind"]
        if kind == "positive":
            out[method] = np.isfinite(values) & (values > 0)
        elif kind == "threshold":
            threshold = float(criterion["value"])
            out[method] = np.isfinite(values) & (values >= threshold)
        elif kind == "top_k_separated":
            k = int(criterion["k"])
            separation = max(1, int(criterion["separation"]))
            window = max(1, int(criterion.get("window", 1)))
            selected = []
            order = np.argsort(-np.nan_to_num(values, nan=-np.inf), kind="stable")
            for i in order:
                if not np.isfinite(values[i]):
                    continue
                if all(abs(int(i) - j) >= separation for j in selected):
                    selected.append(int(i))
                    if len(selected) == k:
                        break
            mask = np.zeros(len(values), dtype=bool)
            for peak in selected:
                # Keep a full window at the right boundary. For the classical
                # profiles, stable sorting still chooses the beginning of the
                # score plateau; pointwise deep scores near the end shift back.
                start = max(0, min(peak, len(mask) - window))
                mask[start:min(start + window, len(mask))] = True
            out[method] = mask
        else:
            raise ValueError(
                f"unknown detection criterion '{kind}' for '{method}'")
    return out


def _selection_summary(q):
    if isinstance(q, dict):
        return "method-specific detection criteria"
    return f"top {100 * (1 - q):.1f}% of days per method"


def method_label(method):
    return METHOD_LABELS.get(method, method.replace("_", " ").title())


def _rgba(hex_colour, alpha):
    value = hex_colour.lstrip("#")
    red, green, blue = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({red},{green},{blue},{alpha})"


def style_figure(fig):
    """Apply the app's common visual language to a Plotly figure."""
    fig.update_layout(
        font=dict(
            family="Inter, system-ui, -apple-system, BlinkMacSystemFont, sans-serif",
            size=13,
            color="#243044",
        ),
        title=dict(font=dict(size=20, color="#172033"), x=0.01, xanchor="left"),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        hoverlabel=dict(
            bgcolor="#172033", bordercolor="#172033",
            font=dict(color="#ffffff", size=12),
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor="#d7dee8",
            borderwidth=1,
            font=dict(size=11),
        ),
    )
    fig.update_xaxes(
        showgrid=True, gridcolor="#e7ecf2", gridwidth=1,
        zeroline=False, linecolor="#b8c2cf", tickfont=dict(color="#526176"),
        title_font=dict(color="#3a485d"),
    )
    fig.update_yaxes(
        showgrid=True, gridcolor="#e7ecf2", gridwidth=1,
        zeroline=False, linecolor="#b8c2cf", tickfont=dict(color="#526176"),
        title_font=dict(color="#3a485d"),
    )
    return fig


def episodes(mask, max_gap=1):
    """Merge runs of flagged days into (start, end) timestamps.

    Daily climate series are strongly autocorrelated, so one heatwave produces
    a cluster of flagged days; counting them separately is misleading.
    """
    idx = mask.index[mask.to_numpy()]
    if len(idx) == 0:
        return []
    out, start, prev = [], idx[0], idx[0]
    for t in idx[1:]:
        if (t - prev).days > max_gap:
            out.append((start, prev))
            start = t
        prev = t
    out.append((start, prev))
    return out


def _split(series, scores):
    """The first scored day is the split; nothing before it is drawn."""
    cut = scores.index[0]
    return series[series.index < cut], series[series.index >= cut], cut


def _seasonal_average(train):
    """Mean and percentile band per calendar day over the training period.
    The only thing the pre-split period contributes to the figures."""
    df = pd.DataFrame({"doy": _doy(train.index) + 1,
                       "v": train.to_numpy(float)})
    g = df.groupby("doy")["v"]
    return g.mean(), g.quantile(0.05), g.quantile(0.95)


def _anomaly_matrix(test, train):
    """Year x day-of-year departures from the training climatology, test only."""
    mean, _ = climatology(train)
    df = pd.DataFrame({"year": test.index.year,
                       "doy": _doy(test.index) + 1,
                       "v": test.to_numpy(float) - mean[_doy(test.index)]})
    return df.pivot_table(index="year", columns="doy", values="v")


def _grid(n, ncols):
    ncols = min(n, ncols)
    return int(np.ceil(n / ncols)), ncols


def _write(fig, path):
    fig.write_html(path, include_plotlyjs="cdn")
    return path


# --------------------------------------------------------------------------
# 1. full time series
# --------------------------------------------------------------------------


def plot_timeseries(series, scores, q=0.99, path="fig_series.html",
                    smooth=365, label=None, return_fig=False):
    """Test-period series on top, one flag strip per method underneath.

    The x axis is shared, so zooming into one episode zooms every strip with
    it -- which is the point of doing this interactively rather than as a PNG.
    """
    flags = flag(scores, q)
    methods = list(flags.columns)
    _, test, cut = _split(series, scores)
    label = label or series.name or "value"

    heights = np.array([4.0] + [1.15] * len(methods))
    fig = make_subplots(
        rows=len(methods) + 1, cols=1, shared_xaxes=True,
        row_heights=(heights / heights.sum()).tolist(),
        vertical_spacing=0.025,
        subplot_titles=["Observed test series"] +
                       [f"{method_label(m)} score" for m in methods],
    )

    fig.add_trace(go.Scattergl(
        x=test.index, y=test.to_numpy(), mode="lines", name=label,
        line=dict(width=0.7, color="#9aa0a6"),
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>"), row=1, col=1)

    if smooth:
        fig.add_trace(go.Scattergl(
            x=test.index, y=test.rolling(smooth, center=True).mean().to_numpy(),
            mode="lines", name=f"{smooth}-day mean",
            line=dict(width=2, color="#202124"),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>"),
            row=1, col=1)

    for k, m in enumerate(methods):
        f = flags[m].to_numpy()
        colour = PALETTE[k % len(PALETTE)]
        fig.add_trace(go.Scattergl(
            x=test.index, y=np.where(f, test.to_numpy(), np.nan),
            mode="lines",
            name=f"{method_label(m)} subsequence",
            legendgroup=m,
            line=dict(width=3.2, color=colour),
            connectgaps=False,
            customdata=scores[m].to_numpy(),
            hovertemplate=("<b>" + method_label(m) + "</b><br>"
                           "%{x|%d %b %Y}<br>value %{y:.2f}"
                           "<br>score %{customdata:.4g}<extra></extra>")),
            row=1, col=1)
        fig.add_trace(go.Scattergl(
            x=scores.index, y=scores[m], mode="lines",
            name=f"{method_label(m)} score", legendgroup=m,
            showlegend=False,
            line=dict(width=1.5, color=colour),
            fill="tozeroy", fillcolor=_rgba(colour, 0.10),
            hovertemplate="%{x|%d %b %Y}<br>score %{y:.4g}<extra></extra>"),
            row=k + 2, col=1)
        fig.add_trace(go.Scattergl(
            x=scores.index,
            y=np.where(f, scores[m].to_numpy(), np.nan),
            mode="lines",
            name=f"{method_label(m)} selected subsequence", legendgroup=m,
            showlegend=False,
            line=dict(width=3.4, color=colour),
            connectgaps=False,
            hovertemplate="%{x|%d %b %Y}<br>selected score %{y:.4g}<extra></extra>"),
            row=k + 2, col=1)
        fig.update_yaxes(title_text="score", row=k + 2, col=1)

    fig.update_yaxes(title_text=label, row=1, col=1)
    fig.update_xaxes(title_text="date", row=len(methods) + 1, col=1)
    fig.update_layout(
        title=(f"detected anomalies, {cut.year}-{test.index[-1].year} "
               f"&mdash; {_selection_summary(q)}"),
        height=390 + 130 * len(methods), template="simple_white",
        hovermode="closest", margin=dict(l=90, r=20, t=60, b=50),
        legend=dict(orientation="h", y=-0.08 if len(methods) < 4 else -0.05))
    fig.update_annotations(font=dict(size=12, color="#526176"))
    style_figure(fig)
    if return_fig:
        return fig
    return _write(fig, path)


# --------------------------------------------------------------------------
# 2. folded on the seasonal cycle
# --------------------------------------------------------------------------


def plot_folded(series, scores, q=0.99, path="fig_folded.html", ncols=3,
                label=None, show_band=False, return_fig=False):
    """Test years on a common day-of-year axis, against the training average.

    The pre-split period appears only as its seasonal mean (and, if you enable
    it in the legend, a 5-95% band). Test-year lines rising above that
    reference is the visual definition of novelty.
    """
    flags = flag(scores, q)
    methods = list(flags.columns)
    train, test, cut = _split(series, scores)
    mu, lo, hi = _seasonal_average(train)
    doy_test = _doy(test.index) + 1
    label = label or series.name or "value"

    nrows, ncols = _grid(len(methods), ncols)
    fig = make_subplots(rows=nrows, cols=ncols, shared_xaxes=True,
                        shared_yaxes=True, vertical_spacing=0.10,
                        horizontal_spacing=0.04,
                        subplot_titles=[
                            f"{method_label(m)} · "
                            f"{len(episodes(flags[m]))} subsequences"
                                        for m in methods])

    # all test years as one trace, NaN-separated: far faster than one per year
    xs, ys = [], []
    for _, grp in test.groupby(test.index.year):
        xs.extend((_doy(grp.index) + 1).tolist() + [None])
        ys.extend(grp.to_numpy().tolist() + [None])

    for k, m in enumerate(methods):
        r, c = k // ncols + 1, k % ncols + 1
        first = k == 0

        fig.add_trace(go.Scattergl(
            x=lo.index.tolist() + hi.index[::-1].tolist(),
            y=lo.to_numpy().tolist() + hi.to_numpy()[::-1].tolist(),
            fill="toself", fillcolor="rgba(150,150,150,0.28)", mode="lines",
            line=dict(width=0), name=f"train 5-95% (&lt;{cut.year})",
            legendgroup="band", showlegend=first, hoverinfo="skip",
            visible=True if show_band else "legendonly"), row=r, col=c)

        fig.add_trace(go.Scattergl(
            x=xs, y=ys, mode="lines", line=dict(width=0.5, color="#c2c7cc"),
            name=f"test years ({cut.year}+)", legendgroup="test",
            showlegend=first, hoverinfo="skip"), row=r, col=c)

        fig.add_trace(go.Scattergl(
            x=mu.index, y=mu.to_numpy(), mode="lines",
            line=dict(width=2.4, color="#202124"),
            name=f"train average (&lt;{cut.year})", legendgroup="mu",
            showlegend=first,
            hovertemplate="day %{x}<br>train mean %{y:.2f}<extra></extra>"),
            row=r, col=c)

        f = flags[m].to_numpy()
        selected_values = np.where(f, test.to_numpy(), np.nan)
        # Never connect a subsequence across the wrapped year boundary.
        year_boundaries = np.flatnonzero(np.diff(test.index.year) != 0) + 1
        selected_values[year_boundaries] = np.nan
        fig.add_trace(go.Scattergl(
            x=doy_test, y=selected_values, mode="lines+markers", name=m,
            line=dict(width=2.6, color=PALETTE[k % len(PALETTE)]),
            marker=dict(size=4, color=PALETTE[k % len(PALETTE)],
                        line=dict(width=1.2, color="#ffffff")),
            connectgaps=False,
            customdata=np.c_[test.index.year, scores[m].to_numpy()],
            hovertemplate=("<b>" + m + "</b><br>%{customdata[0]}, day %{x}"
                           "<br>value %{y:.2f}<br>score %{customdata[1]:.4f}"
                           "<extra></extra>"), showlegend=False), row=r, col=c)

        fig.update_xaxes(tickvals=MONTH_TICKS, ticktext=MONTH_NAMES,
                         range=[1, 365], row=r, col=c)
        if c == 1:
            fig.update_yaxes(title_text=label, row=r, col=c)

    fig.update_annotations(font_size=12)
    fig.update_layout(
        title=(f"folded on the seasonal cycle &mdash; test years {cut.year}-"
               f"{test.index[-1].year} against the training average, "
               f"{_selection_summary(q)}"),
        height=300 * nrows + 120, template="simple_white",
        margin=dict(l=80, r=20, t=110, b=50),
        legend=dict(orientation="h", y=1.06, x=0))
    style_figure(fig)
    if return_fig:
        return fig
    return _write(fig, path)


# --------------------------------------------------------------------------
# 3. year x day-of-year raster
# --------------------------------------------------------------------------


def plot_calendar(series, scores, q=0.99, path="fig_calendar.html", ncols=3,
                  return_fig=False):
    """Test years as rows, calendar days as columns, coloured by departure
    from the training climatology, with flagged days marked.

    Reads two things at once: horizontal structure is a method's seasonal
    preference, vertical structure is whether it fires more in recent years.
    """
    flags = flag(scores, q)
    methods = list(flags.columns)
    train, test, cut = _split(series, scores)
    mat = _anomaly_matrix(test, train)
    lim = float(np.nanpercentile(np.abs(mat.to_numpy()), 99))

    nrows, ncols = _grid(len(methods), ncols)
    fig = make_subplots(rows=nrows, cols=ncols, shared_xaxes=True,
                        shared_yaxes=True, vertical_spacing=0.10,
                        horizontal_spacing=0.04,
                        subplot_titles=[
                            f"{method_label(m)} · "
                            f"{len(episodes(flags[m]))} subsequences"
                                        for m in methods])

    for k, m in enumerate(methods):
        r, c = k // ncols + 1, k % ncols + 1
        fig.add_trace(go.Heatmap(
            z=mat.to_numpy(), x=mat.columns.to_numpy(), y=mat.index.to_numpy(),
            coloraxis="coloraxis", name="",
            hovertemplate="%{y}, day %{x}<br>%{z:+.2f} vs climatology<extra></extra>"),
            row=r, col=c)

        f = flags[m].to_numpy()
        fig.add_trace(go.Scattergl(
            x=_doy(flags.index[f]) + 1, y=flags.index[f].year, mode="markers",
            marker=dict(size=6, color="#172033", symbol="circle-open",
                        line=dict(width=1.4, color="#172033")),
            name=m, showlegend=False,
            customdata=flags.index[f].strftime("%Y-%m-%d"),
            hovertemplate="<b>" + m + "</b><br>%{customdata}<extra></extra>"),
            row=r, col=c)

        fig.update_xaxes(tickvals=MONTH_TICKS, ticktext=MONTH_NAMES,
                         range=[1, 365], row=r, col=c)
        if c == 1:
            fig.update_yaxes(title_text="year", row=r, col=c)

    fig.update_annotations(font_size=12)
    fig.update_layout(
        title=(f"calendar view &mdash; test years {cut.year}-"
               f"{test.index[-1].year}, {_selection_summary(q)}"),
        height=300 * nrows + 120, template="simple_white",
        margin=dict(l=80, r=20, t=110, b=50),
        coloraxis=dict(colorscale="RdBu", reversescale=True,
                       cmin=-lim, cmax=lim, cmid=0,
                       colorbar=dict(title="vs training<br>climatology")))
    style_figure(fig)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    if return_fig:
        return fig
    return _write(fig, path)


# --------------------------------------------------------------------------


def plot_all(series, scores, q=0.99, prefix="fig", label=None):
    return [plot_timeseries(series, scores, q, f"{prefix}_series.html", label=label),
            plot_folded(series, scores, q, f"{prefix}_folded.html", label=label),
            plot_calendar(series, scores, q, f"{prefix}_calendar.html")]


def episode_table(scores, q=0.99, max_gap=1):
    """One row per flagged episode: method, start, end, length in days.
    The tabular counterpart of the figures, for the results section."""
    flags = flag(scores, q)
    rows = [{"method": m, "start": a, "end": b, "days": (b - a).days + 1}
            for m in flags.columns for a, b in episodes(flags[m], max_gap)]
    return pd.DataFrame(rows).sort_values(["method", "start"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Plot detector scores (interactive).")
    p.add_argument("--scores", required=True, help="CSV written by detect.py --output")
    p.add_argument("--input", required=True, help="the same .npy series that was scored")
    p.add_argument("--start", required=True, help="start date of the series (YYYY-MM-DD)")
    p.add_argument("--end", help="end date of the series (YYYY-MM-DD)")
    p.add_argument("--frequency", default="D")
    p.add_argument("--label", default="value", help="y-axis label")
    p.add_argument("--q", type=float, default=0.99,
                   help="alert budget: q=0.99 flags each method's top 1%% of days")
    p.add_argument("--band", action="store_true",
                   help="show the train 5-95%% band by default in the folded view")
    p.add_argument("--outdir", default="figs")
    a = p.parse_args()

    series = load_series(a.input, a.start, a.end, a.frequency)
    series.name = a.label
    scores = load_scores(a.scores)

    missing = scores.index.difference(series.index)
    if len(missing):
        raise SystemExit(f"{len(missing)} scored dates are absent from the series "
                         f"(first: {missing[0].date()}) -- check --start/--frequency")

    os.makedirs(a.outdir, exist_ok=True)
    stem = os.path.join(a.outdir, os.path.splitext(os.path.basename(a.input))[0])
    print(f"split at {scores.index[0].date()}, {len(scores)} scored days, "
          f"methods: {', '.join(scores.columns)}")

    print(f"wrote {plot_timeseries(series, scores, a.q, f'{stem}_series.html', label=a.label)}")
    print(f"wrote {plot_folded(series, scores, a.q, f'{stem}_folded.html', label=a.label, show_band=a.band)}")
    print(f"wrote {plot_calendar(series, scores, a.q, f'{stem}_calendar.html')}")

    table = episode_table(scores, q=a.q)
    table.to_csv(f"{stem}_episodes.csv", index=False)
    print(f"wrote {stem}_episodes.csv\n")
    print(table.groupby("method")["days"].agg(["count", "mean", "max"]).round(1))
