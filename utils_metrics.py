# utils_metrics.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_metrics(df: pd.DataFrame, alpha=4, beta=1):
    """
    Compute basic ABR-style metrics from a per-chunk log.
    df must have: bitrate_kbps, rebuffer_s
    """
    avg_bitrate = df["bitrate_kbps"].mean()
    total_rebuffer = df["rebuffer_s"].sum()
    # count bitrate switches
    switches = max(int((df["bitrate_kbps"].diff() != 0).sum() - 1), 0)
    # QoE like Pensieve (simple)
    quality_score = np.log2(df["bitrate_kbps"]).mean() * 10
    qoe = quality_score - alpha * total_rebuffer - beta * switches
    return avg_bitrate, total_rebuffer, switches, qoe


def add_normalized_qoe(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize QoE PER TRACE, so every trace goes 0–100 across the 4 players.
    This is useful when some traces are very hard.
    """
    out = []
    group_cols = ["trace"] if "trace" in metrics_df.columns else []

    if not group_cols:
        # no trace column, just normalize whole df
        qmin = metrics_df["QoE"].min()
        qmax = metrics_df["QoE"].max()
        if qmax - qmin < 1e-6:
            metrics_df["QoE_norm"] = 100.0
        else:
            metrics_df["QoE_norm"] = 100.0 * (metrics_df["QoE"] - qmin) / (qmax - qmin)
        return metrics_df

    for _, g in metrics_df.groupby(group_cols):
        qmin = g["QoE"].min()
        qmax = g["QoE"].max()
        if qmax - qmin < 1e-6:
            g["QoE_norm"] = 100.0
        else:
            g["QoE_norm"] = 100.0 * (g["QoE"] - qmin) / (qmax - qmin)
        out.append(g)

    return pd.concat(out, ignore_index=True)


def add_viewer_happiness(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Viewer happiness (0–100) PER TRACE.
    Higher bitrate -> good
    More rebuffer -> bad
    More switches -> bad
    """
    out = []
    group_cols = ["trace"] if "trace" in metrics_df.columns else []

    if not group_cols:
        # single group
        max_b = metrics_df["avg_bitrate_kbps"].max()
        max_r = metrics_df["total_rebuffer_s"].max()
        max_s = metrics_df["switches"].max()

        def sd(a, b):
            return a / b if b and b > 0 else 0.0

        metrics_df["Viewer_Happiness"] = (
            0.4 * sd(metrics_df["avg_bitrate_kbps"], max_b)
            + 0.4 * (1 - sd(metrics_df["total_rebuffer_s"], max_r))
            + 0.2 * (1 - sd(metrics_df["switches"], max_s))
        ) * 100
        return metrics_df

    # per trace
    for _, g in metrics_df.groupby(group_cols):
        max_b = g["avg_bitrate_kbps"].max()
        max_r = g["total_rebuffer_s"].max()
        max_s = g["switches"].max()

        def sd(a, b):
            return a / b if b and b > 0 else 0.0

        g["Viewer_Happiness"] = (
            0.4 * sd(g["avg_bitrate_kbps"], max_b)
            + 0.4 * (1 - sd(g["total_rebuffer_s"], max_r))
            + 0.2 * (1 - sd(g["switches"], max_s))
        ) * 100
        out.append(g)
    return pd.concat(out, ignore_index=True)


def add_viewer_score_simple(df: pd.DataFrame) -> pd.DataFrame:
    """
    This is for the FINAL table (already averaged per player).
    So we just normalize ONCE across the 4 players.
    """
    max_b = df["avg_bitrate_kbps"].max()
    max_r = df["total_rebuffer_s"].max()
    max_s = df["switches"].max()

    def sd(a, b):
        return a / b if b and b > 0 else 0.0

    df["ViewerScore"] = (
        0.45 * sd(df["avg_bitrate_kbps"], max_b)
        + 0.45 * (1 - sd(df["total_rebuffer_s"], max_r))
        + 0.10 * (1 - sd(df["switches"], max_s))
    ) * 100
    return df


def plot_metric_bars(metrics_df: pd.DataFrame, metric: str, save_path: str):
    """
    General plotting helper (works for old per-trace style).
    """
    plt.figure(figsize=(8, 4))

    if "trace" in metrics_df.columns:
        x_labels = metrics_df["trace"].unique().tolist()
    else:
        x_labels = metrics_df["player"].tolist()

    players = metrics_df["player"].unique()
    x = np.arange(len(x_labels))
    width = 0.15

    for i, player in enumerate(players):
        sub = metrics_df[metrics_df["player"] == player]
        # pad if needed
        vals = sub[metric].tolist()
        plt.bar(x + i * width, vals, width, label=player)

    plt.xticks(x + width * (len(players) - 1) / 2, x_labels, rotation=25, ha="right")
    plt.ylabel(metric)
    plt.title(metric)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
