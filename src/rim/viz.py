"""Static figures. One visual system: thin marks, recessive axes, direct labels.

Colours are a fixed categorical order checked for colour-vision-deficiency
separation; series are told apart by direct labels as well as colour.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e6e5e1"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_topic_trends(
    share: np.ndarray, years: np.ndarray, table: pd.DataFrame, path: str | Path, n_each: int = 4
) -> Path:
    """Small multiples: the topics with the strongest rising and falling trends."""
    ranked = table.sort_values("mk_z")
    falling = ranked[ranked["trend"] == "falling"].head(n_each)
    rising = ranked[ranked["trend"] == "rising"].tail(n_each).iloc[::-1]
    rows = [("Rising", rising, SERIES[0]), ("Falling", falling, SERIES[1])]
    fig, axes = plt.subplots(2, n_each, figsize=(3.1 * n_each, 5.2), sharex=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for r, (label, part, color) in enumerate(rows):
        for c in range(n_each):
            ax = axes[r, c]
            _style(ax)
            if c >= len(part):
                ax.axis("off")
                continue
            row = part.iloc[c]
            series = 100 * share[int(row["topic"])]
            ax.plot(years, series, color=color, linewidth=2)
            ax.scatter(
                years[-1:],
                series[-1:],
                color=color,
                s=22,
                zorder=3,
                edgecolor=SURFACE,
                linewidth=1.2,
            )
            terms = ", ".join(str(row.get("terms", "")).split(", ")[:2])
            ax.set_title(f"{label}: {terms}", fontsize=8, color=INK, loc="left", wrap=True)
            ax.set_ylim(bottom=0)
            if c == 0:
                ax.set_ylabel("share of documents (%)", fontsize=8, color=INK_2)
    return _save(fig, path)


def plot_backtest(scores: pd.DataFrame, path: str | Path, metric: str = "spearman") -> Path:
    """Dot plot: mean score per predictor and horizon, with the range over cut-offs."""
    data = scores.dropna(subset=[metric])
    predictors = [
        p for p in ("last_growth", "linear_trend", "ridge") if p in set(data["predictor"])
    ]
    horizons = sorted(data["horizon"].unique())
    fig, ax = plt.subplots(figsize=(6.4, 0.9 + 0.55 * len(predictors) * len(horizons)))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    labels, position = [], 0
    for horizon in horizons:
        for i, predictor in enumerate(predictors):
            values = data[(data["horizon"] == horizon) & (data["predictor"] == predictor)][metric]
            if values.empty:
                continue
            ax.plot(
                [values.min(), values.max()],
                [position, position],
                color=SERIES[i],
                linewidth=2,
                alpha=0.45,
                solid_capstyle="round",
            )
            ax.scatter(
                [values.mean()],
                [position],
                color=SERIES[i],
                s=64,
                zorder=3,
                edgecolor=SURFACE,
                linewidth=1.5,
            )
            ax.text(
                values.max() + 0.015,
                position,
                f"{values.mean():.2f}",
                va="center",
                fontsize=8,
                color=INK_2,
            )
            labels.append((position, f"{predictor.replace('_', ' ')}, {horizon}-year"))
            position += 1
        position += 0.6
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_yticks([p for p, _ in labels], [t for _, t in labels], fontsize=8, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel(
        f"{metric}, forecast vs realised growth (dot: mean, bar: range over cut-offs)",
        fontsize=8,
        color=INK_2,
    )
    return _save(fig, path)


def plot_quality_latency(points: list[dict], path: str | Path, metric: str = "nDCG@10") -> Path:
    """Quality against median latency per query. ``points``: name, latency_ms, quality, group."""
    groups = list(dict.fromkeys(p["group"] for p in points))
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    for p in points:
        color = SERIES[groups.index(p["group"]) % len(SERIES)]
        ax.scatter(
            p["latency_ms"],
            p["quality"],
            color=color,
            s=64,
            zorder=3,
            edgecolor=SURFACE,
            linewidth=1.5,
        )
        ax.annotate(
            p["name"],
            (p["latency_ms"], p["quality"]),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=8,
            color=INK_2,
        )
    ax.set_xscale("log")
    ax.set_xlabel("median latency per query (ms, log scale)", fontsize=8, color=INK_2)
    ax.set_ylabel(f"{metric} (test)", fontsize=8, color=INK_2)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=SERIES[i % len(SERIES)], label=g)
        for i, g in enumerate(groups)
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8, labelcolor=INK_2, loc="lower right")
    return _save(fig, path)
