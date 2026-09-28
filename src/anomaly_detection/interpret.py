"""
interpret.py -- why was this flagged?

The three views answer the question each method actually asks:

  burst        the flagged stretch against the seasonal normal, plus where its
               aggregated excess falls in the distribution of every training
               stretch of the same length
  matrix       the flagged subsequence and its closest historical analogue,
  _profile     each in its own seasonal position, overlaid for shape
  autoencoder  the reconstruction against the original, in physical units

    python interpret.py --scores outputs/scores.csv --input data/med.npy \
        --start 1960-01-01 --end 2023-12-31 --frequency D \
        --method burst --top 4 --outdir figs

Note: this module refits what it needs (nearest-neighbour indices, the trained
network) because detect.py returns scores only. That costs one extra pass; it
keeps the detector contract to a single array.
"""

import argparse
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .detect import _doy, _windows, climatology, load_series, samples_per_day, zscore
from .plots import PALETTE, episodes, flag, load_scores

BAND = "rgba(150,150,150,0.30)"
CLIM = "#202124"
OBS = "#d62728"
NN = "#1f77b4"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _doy_axis(index):
    """Day-of-year, unwrapped past 365 so an episode spanning New Year plots
    as a continuous run instead of jumping back to the left edge."""
    d = (_doy(index) + 1).astype(float)
    if len(d) > 1:
        d = d + np.concatenate([[0], np.cumsum(np.diff(d) < -300) * 365.0])
    return d


def _ticks(index, axis, n=6):
    step = max(1, len(index) // n)
    return axis[::step], [t.strftime("%d %b") for t in index[::step]]


def _clim_at(clim, axis):
    """Climatological mean and sd sampled along an unwrapped day-of-year axis."""
    mean, std = clim
    k = (np.asarray(axis).astype(int) - 1) % 365
    return mean[k], std[k]


def _seasonal_peers(train, target_index):
    """Historical subsequences at the target's calendar position.

    One NaN-separated trace renders every sufficiently complete training year
    efficiently while preserving each year as a distinct line.
    """
    target_doy = np.asarray(_doy(target_index))
    target_axis = _doy_axis(target_index)
    xs, ys, count = [], [], 0
    for _, group in train.groupby(train.index.year):
        by_day = pd.Series(
            group.to_numpy(float), index=_doy(group.index)
        ).groupby(level=0).mean()
        peer = by_day.reindex(target_doy).to_numpy(float)
        if np.isfinite(peer).sum() < max(2, int(0.8 * len(peer))):
            continue
        xs.extend(target_axis.tolist() + [None])
        ys.extend(peer.tolist() + [None])
        count += 1
    return xs, ys, count


def top_episodes(scores, method, q=0.99, k=4, max_gap=1):
    """The k highest-scoring flagged episodes for one method."""
    f = flag(scores, q)[method]
    eps = episodes(f, max_gap)
    peak = [scores[method].loc[a:b].max() for a, b in eps]
    return [eps[i] for i in np.argsort(peak)[::-1][:k]]


def _context(series, a, b, pad):
    lo = a - pd.Timedelta(days=pad)
    hi = b + pd.Timedelta(days=pad)
    return series.loc[lo:hi]


def _write(fig, path):
    fig.write_html(path, include_plotlyjs="cdn")
    return path


# --------------------------------------------------------------------------
# 1. burst
# --------------------------------------------------------------------------


def burst_figure(train, test, scores, q=0.99, k=4, path="fig_burst.html",
                 method="burst", pad=20, label="value", return_fig=False):
    """One row per episode: the flagged stretch folded onto its seasonal
    position, and the distribution of the aggregated excess.

    The aggregate is the detector's own statistic, sum(z)/sqrt(L). The null is
    every training window of the same length L, so the marked line reads
    directly as "this stretch is more extreme than X% of anything the training
    period produced at this duration".
    """
    eps = top_episodes(scores, method, q, k)
    if not eps:
        raise SystemExit(f"no episodes for '{method}' at q={q}")
    clim = climatology(train)
    ztr = np.nan_to_num(zscore(train, clim))
    ctr = np.concatenate([[0.0], np.cumsum(ztr)])

    fig = make_subplots(
        rows=len(eps), cols=2, column_widths=[0.62, 0.38],
        vertical_spacing=0.09, horizontal_spacing=0.09,
        subplot_titles=sum(
            [[f"{a.date()} &rarr; {b.date()} ({(b - a).days + 1} d)",
              f"aggregated excess, L = {(b - a).days + 1} d"]
             for a, b in eps], []))

    for r, (a, b) in enumerate(eps, start=1):
        seg = test.loc[a:b]
        ctx = _context(test, a, b, pad)
        ax_ctx, ax_seg = _doy_axis(ctx.index), _doy_axis(seg.index)
        mu, sd = _clim_at(clim, ax_ctx)

        fig.add_trace(go.Scatter(
            x=np.r_[ax_ctx, ax_ctx[::-1]],
            y=np.r_[mu - 1.64 * sd, (mu + 1.64 * sd)[::-1]], fill="toself",
            fillcolor=BAND, line=dict(width=0), name="train 5-95%",
            legendgroup="band", showlegend=r == 1, hoverinfo="skip"),
            row=r, col=1)
        peer_x, peer_y, peer_count = _seasonal_peers(train, ctx.index)
        fig.add_trace(go.Scatter(
            x=peer_x, y=peer_y, mode="lines",
            line=dict(color="rgba(71,85,105,0.20)", width=0.9),
            name=f"historical same-season subsequences ({peer_count})",
            legendgroup="peers", showlegend=r == 1,
            hoverinfo="skip"), row=r, col=1)
        fig.add_trace(go.Scatter(
            x=ax_ctx, y=mu, mode="lines", line=dict(color=CLIM, width=2),
            name="train average", legendgroup="mu", showlegend=r == 1,
            hovertemplate="train mean %{y:.2f}<extra></extra>"), row=r, col=1)
        fig.add_trace(go.Scatter(
            x=ax_ctx, y=ctx.to_numpy(), mode="lines",
            line=dict(color="#9aa0a6", width=1), name="observed (context)",
            legendgroup="ctx", showlegend=r == 1,
            customdata=ctx.index.strftime("%Y-%m-%d"),
            hovertemplate="%{customdata}<br>%{y:.2f}<extra></extra>"),
            row=r, col=1)
        fig.add_trace(go.Scatter(
            x=ax_seg, y=seg.to_numpy(), mode="lines",
            line=dict(color=OBS, width=2.6), name="flagged stretch",
            legendgroup="seg", showlegend=r == 1,
            customdata=seg.index.strftime("%Y-%m-%d"),
            hovertemplate="%{customdata}<br>%{y:.2f}<extra></extra>"),
            row=r, col=1)

        # null distribution of the aggregate at this duration
        L = len(seg)
        null = (ctr[L:] - ctr[:-L]) / np.sqrt(L)
        obs = float(np.nansum(zscore(seg, clim)) / np.sqrt(L))
        pct = float((null < obs).mean() * 100)

        fig.add_trace(go.Histogram(
            x=null, nbinsx=60, marker_color="#c2c7cc", name="train windows",
            legendgroup="null", showlegend=r == 1,
            hovertemplate="aggregate %{x:.2f}<br>%{y} windows<extra></extra>"),
            row=r, col=2)
        fig.add_vline(x=obs, line=dict(color=OBS, width=2.5), row=r, col=2,
                      annotation_text=f"{obs:.2f} &mdash; above {pct:.2f}% of train",
                      annotation_position="top left",
                      annotation_font=dict(size=10, color=OBS))

        tv, tt = _ticks(ctx.index, ax_ctx)
        fig.update_xaxes(tickvals=tv, ticktext=tt, row=r, col=1)
        fig.update_yaxes(title_text=label, row=r, col=1)
        fig.update_xaxes(title_text="sum(z) / sqrt(L)", row=r, col=2)
        fig.update_yaxes(title_text="count", row=r, col=2)

    fig.update_annotations(font_size=12)
    fig.update_layout(
        title=f"burst &mdash; top {len(eps)} episodes and their aggregated excess",
        height=300 * len(eps) + 130, template="simple_white", bargap=0.02,
        margin=dict(l=80, r=30, t=110, b=50),
        legend=dict(orientation="h", y=1.05, x=0))
    if return_fig:
        return fig
    return _write(fig, path)


# --------------------------------------------------------------------------
# 2. matrix profile
# --------------------------------------------------------------------------


def mp_figure(train, test, scores, q=0.99, k=4, m=15, normalize=False,
              path="fig_mp.html", method="mp_raw", label="value",
              return_fig=False):
    """One row per episode: the flagged subsequence in its seasonal position,
    its nearest training neighbour in *its* seasonal position, and the two
    overlaid on a common within-window axis.

    The third panel is the one that carries the argument. If the neighbour
    tracks the query closely, the novelty is a level or timing shift; if the
    shapes diverge, it is genuinely new behaviour.
    """
    import stumpy

    eps = top_episodes(scores, method, q, k)
    if not eps:
        raise SystemExit(f"no episodes for '{method}' at q={q}")
    clim = climatology(train)
    a_arr = np.ascontiguousarray(test.to_numpy(float))
    b_arr = np.ascontiguousarray(train.to_numpy(float))
    prof = stumpy.stump(a_arr, m, b_arr, ignore_trivial=False, normalize=normalize)
    dist, nn_idx = prof[:, 0].astype(float), prof[:, 1].astype(int)

    fig = make_subplots(
        rows=len(eps), cols=3, vertical_spacing=0.09, horizontal_spacing=0.06,
        subplot_titles=sum([["query (test)", "nearest neighbour (train)",
                             "overlaid"] for _ in eps], []))

    for r, (a, b) in enumerate(eps, start=1):
        # the window inside this episode with the largest distance
        lo = max(0, test.index.get_loc(a) - m + 1)
        hi = min(len(dist), test.index.get_loc(b) + 1)
        i = lo + int(np.nanargmax(dist[lo:hi]))
        j = int(nn_idx[i])

        qs = test.iloc[i:i + m]
        ns = train.iloc[j:j + m]
        d = float(dist[i])

        for col, (seg, colour, tag) in enumerate(
                [(qs, OBS, "query"), (ns, NN, "neighbour")], start=1):
            ax = _doy_axis(seg.index)
            mu, sd = _clim_at(clim, ax)
            fig.add_trace(go.Scatter(
                x=np.r_[ax, ax[::-1]],
                y=np.r_[mu - 1.64 * sd, (mu + 1.64 * sd)[::-1]], fill="toself",
                fillcolor=BAND, line=dict(width=0), name="train 5-95%",
                legendgroup="band", showlegend=(r == 1 and col == 1),
                hoverinfo="skip"), row=r, col=col)
            peer_x, peer_y, peer_count = _seasonal_peers(train, seg.index)
            fig.add_trace(go.Scatter(
                x=peer_x, y=peer_y, mode="lines",
                line=dict(color="rgba(71,85,105,0.20)", width=0.9),
                name=f"historical same-season subsequences ({peer_count})",
                legendgroup="peers",
                showlegend=(r == 1 and col == 1),
                hoverinfo="skip"), row=r, col=col)
            fig.add_trace(go.Scatter(
                x=ax, y=mu, mode="lines", line=dict(color=CLIM, width=1.8),
                name="train average", legendgroup="mu",
                showlegend=(r == 1 and col == 1), hoverinfo="skip"),
                row=r, col=col)
            fig.add_trace(go.Scatter(
                x=ax, y=seg.to_numpy(), mode="lines+markers",
                line=dict(color=colour, width=2.4), marker=dict(size=4),
                name=tag, legendgroup=tag, showlegend=r == 1,
                customdata=seg.index.strftime("%Y-%m-%d"),
                hovertemplate="%{customdata}<br>%{y:.2f}<extra></extra>"),
                row=r, col=col)
            tv, tt = _ticks(seg.index, ax, n=4)
            fig.update_xaxes(tickvals=tv, ticktext=tt, row=r, col=col)
            if col == 1:
                fig.update_yaxes(title_text=label, row=r, col=col)

        off = np.arange(m)
        for seg, colour, tag in [(qs, OBS, "query"), (ns, NN, "neighbour")]:
            fig.add_trace(go.Scatter(
                x=off, y=seg.to_numpy(), mode="lines",
                line=dict(color=colour, width=2.4,
                          dash="solid" if tag == "query" else "dash"),
                name=tag, legendgroup=tag, showlegend=False,
                hovertemplate=tag + " +%{x}d<br>%{y:.2f}<extra></extra>"),
                row=r, col=3)
        fig.update_xaxes(title_text="day within window", row=r, col=3)

        fig.layout.annotations[3 * (r - 1)].text = (
            f"query {qs.index[0].date()}")
        fig.layout.annotations[3 * (r - 1) + 1].text = (
            f"nearest neighbour {ns.index[0].date()}")
        fig.layout.annotations[3 * (r - 1) + 2].text = (
            f"distance {d:.2f}" + ("  (z-normalized)" if normalize else ""))

    fig.update_annotations(font_size=12)
    fig.update_layout(
        title=(f"{method} &mdash; top {len(eps)} episodes, m = {m}, "
               f"each against its closest match before the split"),
        height=290 * len(eps) + 130, template="simple_white",
        margin=dict(l=80, r=30, t=110, b=50),
        legend=dict(orientation="h", y=1.05, x=0))
    if return_fig:
        return fig
    return _write(fig, path)


# --------------------------------------------------------------------------
# 3. autoencoder
# --------------------------------------------------------------------------


def ae_figure(train, test, scores, q=0.99, k=4, m=32, hidden=8, seed=0,
              path="fig_ae.html", method="autoencoder", pad=20, label="value",
              return_fig=False):
    """One row per episode: original against reconstruction, with the residual.

    The network is refitted on the training period and its output mapped back
    to physical units, so the gap between the two curves is readable in
    degrees rather than in z-space.
    """
    from sklearn.neural_network import MLPRegressor

    eps = top_episodes(scores, method, q, k)
    if not eps:
        raise SystemExit(f"no episodes for '{method}' at q={q}")
    clim = climatology(train)
    mean, std = clim

    Xtr = _windows(zscore(train, clim), m)
    net = MLPRegressor(hidden_layer_sizes=(32, hidden, 32), max_iter=300,
                       random_state=seed, early_stopping=True).fit(Xtr, Xtr)

    zte = zscore(test, clim)
    Xte = _windows(zte, m)
    Rte = net.predict(Xte)

    # average the overlapping window reconstructions, then back to units
    acc = np.zeros(len(test))
    cnt = np.zeros(len(test))
    for i in range(len(Rte)):
        acc[i:i + m] += Rte[i]
        cnt[i:i + m] += 1
    zrec = np.divide(acc, np.maximum(cnt, 1))
    d = _doy(test.index)
    recon = pd.Series(mean[d] + zrec * std[d], index=test.index)
    recon[cnt == 0] = np.nan

    fig = make_subplots(
        rows=len(eps), cols=1, vertical_spacing=0.10,
        subplot_titles=[f"{a.date()} &rarr; {b.date()} "
                        f"({(b - a).days + 1} d)" for a, b in eps])

    for r, (a, b) in enumerate(eps, start=1):
        ctx = _context(test, a, b, pad)
        rec = recon.loc[ctx.index]
        res = ctx.to_numpy() - rec.to_numpy()

        fig.add_trace(go.Scatter(
            x=ctx.index, y=ctx.to_numpy(), mode="lines", name="original",
            line=dict(color=OBS, width=2.2), legendgroup="obs",
            showlegend=r == 1,
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>"), row=r, col=1)
        fig.add_trace(go.Scatter(
            x=ctx.index, y=rec.to_numpy(), mode="lines", name="reconstruction",
            line=dict(color=NN, width=2.2, dash="dash"), legendgroup="rec",
            showlegend=r == 1,
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>"), row=r, col=1)
        fig.add_trace(go.Scatter(
            x=ctx.index, y=res, mode="lines", name="residual",
            line=dict(color="#9aa0a6", width=0), fill="tozeroy",
            fillcolor="rgba(214,39,40,0.18)", legendgroup="res",
            showlegend=r == 1, yaxis="y",
            hovertemplate="%{x|%Y-%m-%d}<br>residual %{y:+.2f}<extra></extra>"),
            row=r, col=1)

        fig.add_vrect(x0=a, x1=b, fillcolor="rgba(214,39,40,0.07)",
                      line_width=0, row=r, col=1)
        rmse = float(np.sqrt(np.nanmean(
            (test.loc[a:b].to_numpy() - recon.loc[a:b].to_numpy()) ** 2)))
        fig.layout.annotations[r - 1].text += f"  &mdash; RMSE {rmse:.2f}"
        fig.update_yaxes(title_text=label, row=r, col=1)

    fig.update_annotations(font_size=12)
    fig.update_layout(
        title=(f"autoencoder &mdash; top {len(eps)} episodes, m = {m}, "
               f"reconstruction in physical units"),
        height=280 * len(eps) + 130, template="simple_white",
        margin=dict(l=80, r=30, t=110, b=50),
        legend=dict(orientation="h", y=1.05, x=0))
    if return_fig:
        return fig
    return _write(fig, path)


# --------------------------------------------------------------------------


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Interpret flagged anomalies.")
    p.add_argument("--scores", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end")
    p.add_argument("--frequency", default="D")
    p.add_argument("--label", default="value")
    p.add_argument("--method", nargs="+", default=["burst", "mp_raw", "autoencoder"],
                   help="which score columns to interpret")
    p.add_argument("--q", type=float, default=0.99)
    p.add_argument("--top", type=int, default=4, help="episodes per method")
    p.add_argument("--m", type=int, default=15, help="matrix profile window, in days")
    p.add_argument("--ae_m", type=int, default=32, help="autoencoder window, in days")
    p.add_argument("--outdir", default="figs")
    a = p.parse_args()

    series = load_series(a.input, a.start, a.end, a.frequency)
    series.name = a.label
    scores = load_scores(a.scores)
    spd = samples_per_day(a.frequency)
    cut = scores.index[0]
    train, test = series[series.index < cut], series[series.index >= cut]

    os.makedirs(a.outdir, exist_ok=True)
    stem = os.path.join(a.outdir, os.path.splitext(os.path.basename(a.input))[0])

    for name in a.method:
        if name not in scores.columns:
            print(f"[skip] '{name}' is not a column in {a.scores}")
            continue
        out = f"{stem}_{name}_detail.html"
        if name.startswith("mp") or name == "imp":
            print("wrote", mp_figure(train, test, scores, a.q, a.top,
                                     m=a.m * spd, normalize=name.endswith("znorm"),
                                     path=out, method=name, label=a.label))
        elif name == "autoencoder":
            print("wrote", ae_figure(train, test, scores, a.q, a.top,
                                     m=a.ae_m * spd, path=out, method=name,
                                     label=a.label))
        else:
            print("wrote", burst_figure(train, test, scores, a.q, a.top,
                                        path=out, method=name, label=a.label))
