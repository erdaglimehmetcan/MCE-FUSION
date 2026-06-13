# ============================================================
# LazyPredict Result Visualization  (professional redesign)
# run_visualize_lazypredict_results.py
# ============================================================
#
# Purpose:
#   Visualises LazyPredict CSV outputs for WSI-only, clinical-only,
#   and fusion embeddings in a publication-quality style.
#
# Visual style notes (derived from BVA-508 homework style):
#   - Muted off-white background  (#F5F0EB)
#   - No axis box / spines removed
#   - Subtle horizontal grid lines
#   - Accent left-border on chart titles  (crimson strip)
#   - Bold main-title line + lighter subtitle
#   - Metric annotations written directly inside / beside bars
#   - Consistent dark-grey text  (#2A2A2A / #6A6A6A)
#   - Single cohesive colour-per-metric palette
#
# Input:  one or more LazyPredict CSV files
# Output: output_dir/
#             per_run_plots/
#             comparison_plots/
#             summaries/
#
# Example (six embeddings):
#   python run_visualize_lazypredict_results.py ^
#     --input_csvs lazy_WSI.csv lazy_clinical.csv lazy_concat.csv ^
#       lazy_gated.csv lazy_cross.csv lazy_gated_cross.csv ^
#     --run_labels "WSI-only" "Clinical-only" "Concat" ^
#       "Gated Attention" "Cross-Attention" "Gated Cross-Attention" ^
#     --output_dir results/visualizations
# ============================================================

import argparse
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator

matplotlib.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.spines.left":  False,
    "axes.spines.bottom":False,
    "xtick.bottom":      False,
    "ytick.left":        False,
    "figure.dpi":        150,
})

# ── Colour palette ──────────────────────────────────────────────────────────
PALETTE = {
    "bg":       "#FFFFFF",   # white background (thesis/print safe)
    "text":     "#2A2A2A",   # near-black for primary text
    "muted":    "#6A6A6A",   # secondary / caption text
    "accent":   "#D95F5F",   # accent strip / highlight
    "grid":     "#E5E5E5",   # subtle gridlines
    "bar":      "#4A7FB5",   # default bar fill (single metric)
    "best":     "#2E5FA3",   # top bar highlighted darker
}

# Each metric gets its own bar colour for per-run plots
METRIC_COLORS = {
    "AUC":       "#4A7FB5",
    "F1":        "#E07B54",
    "Recall":    "#6BAE8A",
    "Precision": "#9B6BB5",
    "Accuracy":  "#D4A843",
}

# For comparison / matrix plots, one colour per embedding
EMBED_COLORS = [
    "#4A7FB5", "#E07B54", "#6BAE8A",
    "#9B6BB5", "#D4A843", "#D95F5F",
    "#5BAECC", "#A0A0A0",
]


# ── I/O helpers ──────────────────────────────────────────────────────────────

def read_params_json(path):
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_lazypredict_csv(csv_path, model_col):
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    if model_col not in df.columns:
        df = df.rename(columns={df.columns[0]: model_col})
    return df


def clean_metric_column(df, col):
    if col not in df.columns:
        return None
    return pd.to_numeric(df[col], errors="coerce")


def safe_metric_columns(args):
    return {
        "AUC":       args.auc_col,
        "F1":        args.f1_col,
        "Recall":    args.recall_col,
        "Precision": args.precision_col,
        "Accuracy":  args.accuracy_col,
    }


def params_to_text(params_data):
    if not params_data:
        return ""
    parameters = params_data.get("parameters", params_data)
    useful_keys = [
        "wsi_dim", "clinical_dim", "fused_dim", "hidden_dim",
        "num_tokens", "num_heads", "epochs", "batch_size", "lr",
        "dropout", "seed",
    ]
    items = [f"{k}={parameters[k]}" for k in useful_keys if k in parameters]
    return "  |  ".join(items)


# ── Figure helpers ────────────────────────────────────────────────────────────

def _apply_bg(fig, ax_list):
    """Apply background colour to figure and all axes."""
    fig.patch.set_facecolor(PALETTE["bg"])
    for ax in ax_list:
        ax.set_facecolor(PALETTE["bg"])


def _draw_title_block(fig, main_title, subtitle="", x=0.055, y_main=0.96, y_sub=0.915):
    """Draw bold headline + muted subtitle + red accent strip (Economist style)."""
    fig.add_artist(
        plt.Rectangle(
            (x - 0.012, y_main - 0.01), 0.007, 0.06,
            color=PALETTE["accent"],
            transform=fig.transFigure,
            clip_on=False,
        )
    )
    fig.text(x, y_main, main_title,
             fontsize=14, fontweight="bold", color=PALETTE["text"],
             va="top", transform=fig.transFigure)
    if subtitle:
        fig.text(x, y_sub, subtitle,
                 fontsize=10, color=PALETTE["muted"],
                 va="top", transform=fig.transFigure)


def _style_axes(ax, x_label="", show_xgrid=False):
    """Remove all spines, add subtle y-grid, style ticks."""
    ax.grid(axis="x" if show_xgrid else "y",
            color=PALETTE["grid"], linewidth=0.8, zorder=0)
    ax.tick_params(axis="both", length=0,
                   labelsize=9, colors=PALETTE["muted"])
    if x_label:
        ax.set_xlabel(x_label, fontsize=9, color=PALETTE["muted"], labelpad=6)


def _bar_labels(ax, bars, fmt=".3f", offset_frac=0.01, inside_thresh=0.60):
    """Write value labels beside / inside bars."""
    x_max = ax.get_xlim()[1]
    for bar in bars:
        w = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        label = f"{w:{fmt}}"
        # put label inside bar if bar is long enough, else outside
        if w / x_max >= inside_thresh:
            ax.text(w - offset_frac * x_max * 2, y, label,
                    va="center", ha="right", fontsize=8,
                    color="white", fontweight="bold")
        else:
            ax.text(w + offset_frac * x_max, y, label,
                    va="center", ha="left", fontsize=8,
                    color=PALETTE["muted"])


# ── Per-run plots ─────────────────────────────────────────────────────────────

def make_metric_plot(df, metric_name, metric_col, model_col,
                     run_label, params_text, output_path, top_n):
    if metric_col not in df.columns:
        return False

    plot_df = df.copy()
    plot_df[metric_col] = clean_metric_column(plot_df, metric_col)
    plot_df = plot_df.dropna(subset=[metric_col])
    if plot_df.empty:
        return False

    plot_df = plot_df.sort_values(metric_col, ascending=False).head(top_n)
    plot_df = plot_df.sort_values(metric_col, ascending=True).reset_index(drop=True)

    n = len(plot_df)
    fig_h = max(4.5, 0.42 * n + 1.8)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    _apply_bg(fig, [ax])

    color = METRIC_COLORS.get(metric_name, PALETTE["bar"])
    colors = [PALETTE["best"] if i == n - 1 else color for i in range(n)]

    bars = ax.barh(
        plot_df[model_col], plot_df[metric_col],
        color=colors, edgecolor="none", height=0.65, zorder=3,
    )

    ax.set_xlim(0, min(1.0, plot_df[metric_col].max() + 0.10))
    _style_axes(ax, x_label=metric_name)
    _bar_labels(ax, bars)

    # Highlight best bar label in bold
    ytick_labels = ax.get_yticklabels()
    if ytick_labels:
        ytick_labels[-1].set_color(PALETTE["text"])
        ytick_labels[-1].set_fontweight("bold")

    # Title block
    _draw_title_block(
        fig,
        main_title=f"{run_label}  ·  {metric_name}",
        subtitle=f"Top {top_n} classifiers ranked by {metric_name}  |  best classifier highlighted",
    )

    # Footer params
    if params_text:
        fig.text(0.055, 0.022, params_text,
                 fontsize=7.5, color=PALETTE["muted"],
                 va="bottom", transform=fig.transFigure)

    fig.subplots_adjust(left=0.25, right=0.92, top=0.83, bottom=0.10)
    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return True


# ── Comparison plots ──────────────────────────────────────────────────────────

def make_best_model_comparison(summary_df, metric_name, output_path):
    metric_df = summary_df[summary_df["metric"] == metric_name].copy()
    if metric_df.empty:
        return False

    metric_df = metric_df.sort_values("best_score", ascending=True).reset_index(drop=True)
    n = len(metric_df)

    fig_h = max(4.5, 0.52 * n + 1.8)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    _apply_bg(fig, [ax])

    colors = [EMBED_COLORS[i % len(EMBED_COLORS)] for i in range(n)]
    labels = [
        f"{row['run_label']}   ({row['best_model']})"
        for _, row in metric_df.iterrows()
    ]

    bars = ax.barh(
        labels, metric_df["best_score"],
        color=colors, edgecolor="none", height=0.62, zorder=3,
    )

    ax.set_xlim(0, min(1.0, metric_df["best_score"].max() + 0.10))
    _style_axes(ax, x_label=metric_name)
    _bar_labels(ax, bars)

    _draw_title_block(
        fig,
        main_title=f"Best classifier per embedding  ·  {metric_name}",
        subtitle="Each bar shows the top-performing classifier from that fusion strategy",
    )

    fig.subplots_adjust(left=0.33, right=0.92, top=0.83, bottom=0.10)
    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return True


def make_metric_matrix(summary_df, output_path):
    pivot = summary_df.pivot_table(
        index="run_label", columns="metric",
        values="best_score", aggfunc="max",
    )
    preferred = ["AUC", "F1", "Recall", "Precision", "Accuracy"]
    cols = [c for c in preferred if c in pivot.columns]
    pivot = pivot[cols]

    nrows, ncols = pivot.shape
    fig_w = max(7, 1.6 * ncols)
    fig_h = max(4, 0.7 * nrows + 2.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    _apply_bg(fig, [ax])

    # Draw heatmap manually so we control colours
    data = pivot.values
    # Dynamic vmin/vmax: stretch the gradient across the actual score spread
    # so even a tight 0.6-0.8 band uses the full white-to-navy palette.
    import numpy as np
    _finite = data[~np.isnan(data)]
    _spread = _finite.max() - _finite.min()
    _pad = max(0.04, _spread * 0.15)
    vmin = max(0.0, float(_finite.min()) - _pad)
    vmax = min(1.0, float(_finite.max()) + _pad)

    # 4-stop gradient: white -> light blue -> steel blue -> deep navy
    # Gives maximum perceptual separation across any realistic score range.
    STOPS = [
        (0.00, (0xFF, 0xFF, 0xFF)),   # white
        (0.30, (0xC6, 0xDB, 0xEF)),   # light blue
        (0.65, (0x4A, 0x7F, 0xB5)),   # steel blue
        (1.00, (0x08, 0x30, 0x6B)),   # deep navy
    ]

    def _cell_colour(norm):
        for idx in range(len(STOPS) - 1):
            t0, c0 = STOPS[idx]
            t1, c1 = STOPS[idx + 1]
            if norm <= t1 or idx == len(STOPS) - 2:
                t = (norm - t0) / (t1 - t0) if t1 > t0 else 1.0
                t = max(0.0, min(1.0, t))
                r = int(c0[0] + t * (c1[0] - c0[0]))
                g = int(c0[1] + t * (c1[1] - c0[1]))
                b = int(c0[2] + t * (c1[2] - c0[2]))
                return f"#{r:02X}{g:02X}{b:02X}"
        return "#FFFFFF"

    for i in range(nrows):
        for j in range(ncols):
            val = data[i, j]
            norm = max(0.0, min(1.0, (val - vmin) / (vmax - vmin)))
            cell_col = _cell_colour(norm)
            ax.add_patch(mpatches.FancyBboxPatch(
                (j + 0.05, i + 0.05), 0.90, 0.90,
                boxstyle="round,pad=0.02",
                facecolor=cell_col, edgecolor="none", zorder=2,
            ))
            text_col = "white" if norm > 0.35 else PALETTE["text"]
            ax.text(j + 0.5, i + 0.5, f"{val:.3f}",
                    ha="center", va="center",
                    fontsize=10, fontweight="bold", color=text_col, zorder=3)

    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nrows)
    ax.set_xticks([j + 0.5 for j in range(ncols)])
    ax.set_xticklabels(cols, fontsize=10, color=PALETTE["text"], fontweight="bold")
    ax.set_yticks([i + 0.5 for i in range(nrows)])
    ax.set_yticklabels(pivot.index.tolist(), fontsize=9, color=PALETTE["text"])
    ax.tick_params(length=0)

    _draw_title_block(
        fig,
        main_title="Best-score matrix  ·  all embeddings × all metrics",
        subtitle="Each cell = best classifier score for that (embedding, metric) pair",
    )

    # Colour-scale legend strip
    import numpy as np
    grad_ax = fig.add_axes([0.88, 0.18, 0.018, 0.55])
    grad_ax.set_facecolor(PALETTE["bg"])
    grad_data = np.linspace(0, 1, 256).reshape(256, 1)
    grad_ax.imshow(grad_data[::-1], aspect="auto",
                   extent=[0, 1, vmin, vmax],
                   cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
                       "custom", ["#FFFFFF", "#C6DBEF", "#4A7FB5", "#08306B"]))
    grad_ax.set_xticks([])
    grad_ax.yaxis.tick_right()
    grad_ax.tick_params(labelsize=7, colors=PALETTE["muted"], length=0)
    for spine in grad_ax.spines.values():
        spine.set_visible(False)

    fig.subplots_adjust(left=0.22, right=0.86, top=0.83, bottom=0.10)
    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def make_grouped_bar_comparison(summary_df, output_path):
    """
    One grouped-bar chart: x-axis = metric, groups = embeddings.
    Gives an overview of how fusion strategies compare across all metrics.
    """
    import numpy as np

    preferred = ["AUC", "F1", "Recall", "Precision", "Accuracy"]
    run_labels = summary_df["run_label"].unique().tolist()
    metrics = [m for m in preferred if m in summary_df["metric"].values]

    n_metrics = len(metrics)
    n_runs = len(run_labels)
    bar_w = 0.75 / n_runs
    x = np.arange(n_metrics)

    fig_w = max(10, 1.5 * n_metrics + 2)
    fig, ax = plt.subplots(figsize=(fig_w, 5.5))
    _apply_bg(fig, [ax])

    for ri, run in enumerate(run_labels):
        run_data = summary_df[summary_df["run_label"] == run].set_index("metric")
        scores = [run_data.loc[m, "best_score"] if m in run_data.index else 0.0
                  for m in metrics]
        offset = (ri - n_runs / 2 + 0.5) * bar_w
        ax.bar(x + offset, scores, width=bar_w * 0.92,
               color=EMBED_COLORS[ri % len(EMBED_COLORS)],
               edgecolor="none", label=run, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10, color=PALETTE["text"], fontweight="bold")
    ax.set_ylim(0, 1.08)
    _style_axes(ax, show_xgrid=False)
    ax.yaxis.set_major_locator(MaxNLocator(6))

    # Legend
    legend = ax.legend(
        loc="lower right", fontsize=8.5,
        framealpha=0.85, facecolor=PALETTE["bg"],
        edgecolor=PALETTE["grid"],
    )
    for text in legend.get_texts():
        text.set_color(PALETTE["text"])

    _draw_title_block(
        fig,
        main_title="Fusion strategy comparison  ·  all metrics",
        subtitle="Best classifier score per (embedding, metric) pair",
    )

    fig.subplots_adjust(left=0.08, right=0.97, top=0.83, bottom=0.12)
    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


# ── Single-run analysis ───────────────────────────────────────────────────────

def analyze_single_run(csv_path, run_label, params_json, args, per_run_dir, summary_rows):
    df = load_lazypredict_csv(csv_path, args.model_col)
    params_data = read_params_json(params_json)
    params_text = params_to_text(params_data)
    metric_map = safe_metric_columns(args)

    run_plot_dir = per_run_dir / run_label.replace(" ", "_").replace("-", "_")
    run_plot_dir.mkdir(parents=True, exist_ok=True)

    for metric_name, metric_col in metric_map.items():
        if metric_col not in df.columns:
            continue
        output_path = run_plot_dir / f"{metric_name.lower()}_bar.png"
        make_metric_plot(
            df=df, metric_name=metric_name, metric_col=metric_col,
            model_col=args.model_col, run_label=run_label,
            params_text=params_text, output_path=output_path, top_n=args.top_n,
        )

    for metric_name, metric_col in metric_map.items():
        if metric_col not in df.columns:
            continue
        values = clean_metric_column(df, metric_col)
        temp_df = df.copy()
        temp_df[metric_col] = values
        temp_df = temp_df.dropna(subset=[metric_col])
        if temp_df.empty:
            continue
        best_row = temp_df.sort_values(metric_col, ascending=False).iloc[0]
        summary_rows.append({
            "run_label":   run_label,
            "metric":      metric_name,
            "best_model":  best_row[args.model_col],
            "best_score":  float(best_row[metric_col]),
            "csv_path":    str(csv_path),
            "params_json": str(params_json) if params_json else "",
            "params_text": params_text,
        })


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csvs",  nargs="+", required=True)
    parser.add_argument("--run_labels",  nargs="+", required=True)
    parser.add_argument("--output_dir",  required=True)
    parser.add_argument("--params_jsons", nargs="*", default=None)
    parser.add_argument("--model_col",   default="Model")
    parser.add_argument("--auc_col",     default="ROC AUC")
    parser.add_argument("--f1_col",      default="F1 Score")
    parser.add_argument("--recall_col",  default="Recall")
    parser.add_argument("--precision_col", default="Precision")
    parser.add_argument("--accuracy_col", default="Accuracy")
    parser.add_argument("--top_n",       type=int, default=15)
    parser.add_argument("--main_metric", default="AUC")
    args = parser.parse_args()

    if len(args.input_csvs) != len(args.run_labels):
        raise ValueError("--input_csvs and --run_labels must have the same length.")

    params_jsons = (
        [None] * len(args.input_csvs)
        if not args.params_jsons
        else args.params_jsons
    )
    if len(params_jsons) != len(args.input_csvs):
        raise ValueError("--params_jsons length must match --input_csvs.")

    output_dir    = Path(args.output_dir)
    per_run_dir   = output_dir / "per_run_plots"
    comparison_dir = output_dir / "comparison_plots"
    summary_dir   = output_dir / "summaries"
    for d in (per_run_dir, comparison_dir, summary_dir):
        d.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for csv_path, run_label, params_json in zip(args.input_csvs, args.run_labels, params_jsons):
        analyze_single_run(
            csv_path=csv_path, run_label=run_label, params_json=params_json,
            args=args, per_run_dir=per_run_dir, summary_rows=summary_rows,
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = summary_dir / "best_model_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    for metric_name in ["AUC", "F1", "Recall", "Precision", "Accuracy"]:
        make_best_model_comparison(
            summary_df, metric_name,
            comparison_dir / f"best_model_comparison_{metric_name.lower()}.png",
        )

    make_metric_matrix(summary_df, comparison_dir / "best_score_matrix.png")
    make_grouped_bar_comparison(summary_df, comparison_dir / "grouped_bar_all_metrics.png")

    run_info = {
        "created_at":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_csvs":   args.input_csvs,
        "run_labels":   args.run_labels,
        "params_jsons": params_jsons,
        "model_col":    args.model_col,
        "metric_columns": {
            "AUC":       args.auc_col,
            "F1":        args.f1_col,
            "Recall":    args.recall_col,
            "Precision": args.precision_col,
            "Accuracy":  args.accuracy_col,
        },
        "top_n":        args.top_n,
        "main_metric":  args.main_metric,
    }
    with open(summary_dir / "visualization_params.json", "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=4)

    print("Visualization finished.")
    print("Output directory:", output_dir)
    print("Summary CSV:     ", summary_csv)


if __name__ == "__main__":
    main()
