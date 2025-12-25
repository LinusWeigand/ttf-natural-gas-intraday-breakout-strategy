from __future__ import annotations
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

def bucket_to_minutes(bucket_idx: np.ndarray, bucket_minutes: int) -> np.ndarray:
    return np.asarray(bucket_idx, dtype=float) * float(bucket_minutes)

def add_first_hour_marker(
        ax: Optional[plt.Axes] = None,
        *,
        minute: int = 60,
        label: str = "First Hour End",
        linestyle: str = "--",
        linewidth: float = 1.0,
        text_dx: int = 2,
        text_y_frac: float = 0.95,
        fontsize: int = 9,
) -> None:
    ax = ax or plt.gca()
    ax.axvline(minute, linestyle=linestyle, linewidth=linewidth)

    y0, y1 = ax.get_ylim()
    y = y0 + (y1 - y0) * float(text_y_frac)
    ax.text(minute + text_dx, y, label, fontsize=fontsize)

def add_mean_legend_entry(
        ax: Optional[plt.Axes] = None,
        *,
        label: str = "Mean (dotted)",
        linestyle: str = ":",
        linewidth: float = 1.5,
        color: str = "black",
) -> None:
    ax = ax or plt.gca()
    ax.plot([], [], linestyle=linestyle, linewidth=linewidth, color=color, label=label)

def style_axes(
        ax: Optional[plt.Axes] = None,
        *,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        grid_alpha: float = 0.3,
        tight: bool = False,
) -> None:
    ax = ax or plt.gca()
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(alpha=grid_alpha)
    if tight:
        plt.tight_layout()

def plot_ci_vlines(
        x: np.ndarray,
        y: np.ndarray,
        ylo: np.ndarray,
        yhi: np.ndarray,
        *,
        ax: Optional[plt.Axes] = None,
        alpha: float = 0.4,
) -> None:
    ax = ax or plt.gca()
    x = np.asarray(x)
    y = np.asarray(y)
    ylo = np.asarray(ylo)
    yhi = np.asarray(yhi)

    for i in range(len(x)):
        if np.isfinite(y[i]) and np.isfinite(ylo[i]) and np.isfinite(yhi[i]):
            ax.vlines(x[i], ylo[i], yhi[i], alpha=alpha)