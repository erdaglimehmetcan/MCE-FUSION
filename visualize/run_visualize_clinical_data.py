# ============================================================
# TCGA Clinical Data Visualization
# run_visualize_clinical_data.py
# ============================================================
#
# Visualises the clinical Excel file used in the HER2 thesis pipeline.
# Same visual style as run_visualize_lazypredict_results.py:
#   - White background (thesis/LaTeX safe)
#   - Economist-style title block (red accent strip)
#   - No spines, subtle grid
#   - 300 DPI output
#
# Plots generated:
#   1. er_pr_status_bar.png         - ER / PR status side-by-side bar
#   2. pathologic_stage_bar.png     - AJCC pathologic stage distribution
#   3. t_category_bar.png           - Tumour T category distribution
#   4. n_category_bar.png           - Nodal N category distribution
#   5. age_histogram.png            - Age at index distribution + KDE
#   6. age_by_er_status.png         - Age distribution split by ER status
#   7. receptor_stage_heatmap.png   - ER status × pathologic stage count matrix
#   8. summary_table.png            - Key cohort statistics table figure
#
# Usage:
#   python run_visualize_clinical_data.py \
#     --input_xlsx tcga_clinical_data_main.xlsx \
#     --output_dir results/clinical_viz \
#     --caption "TCGA-BRCA cohort" \
#     --study_title "Multimodal Fusion – HER2 Classification"
# ============================================================

import argparse
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator

warnings.filterwarnings("ignore")

matplotlib.rcParams.update({
    "font.family":        "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.left":   False,
    "axes.spines.bottom": False,
    "xtick.bottom":       False,
    "ytick.left":         False,
    "figure.dpi":         150,
})

# ── Palette ───────────────────────────────────────────────────────────────────
PAL = {
    "bg":      "#FFFFFF",
    "text":    "#2A2A2A",
    "muted":   "#6A6A6A",
    "accent":  "#D95F5F",
    "grid":    "#E5E5E5",
    "pos":     "#4A7FB5",   # Positive
    "neg":     "#E07B54",   # Negative
    "indet":   "#A0A0A0",   # Indeterminate
    "bar":     "#4A7FB5",
    "kde":     "#D95F5F",
}

STAGE_COLORS = [
    "#08306B", "#2171B5", "#4A7FB5", "#6BAED6",
    "#9ECAE1", "#C6DBEF", "#DEEBF7", "#F7FBFF",
    "#A0A0A0", "#C8C8C8", "#E8E8E8", "#FFFFFF",
]


# ── Figure helpers ─────────────────────────────────────────────────────────────
def _bg(fig, axes):
    fig.patch.set_facecolor(PAL["bg"])
    for ax in (axes if hasattr(axes, "__iter__") else [axes]):
        ax.set_facecolor(PAL["bg"])


def _title(fig, main, sub="", x=0.055, ym=0.96, ys=0.915):
    fig.add_artist(plt.Rectangle(
        (x - 0.012, ym - 0.01), 0.007, 0.06,
        color=PAL["accent"], transform=fig.transFigure, clip_on=False,
    ))
    fig.text(x, ym, main, fontsize=14, fontweight="bold",
             color=PAL["text"], va="top", transform=fig.transFigure)
    if sub:
        fig.text(x, ys, sub, fontsize=10, color=PAL["muted"],
                 va="top", transform=fig.transFigure)


def _caption(fig, caption, study_title, x=0.98, y=0.022):
    parts = []
    if study_title:
        parts.append(study_title)
    if caption:
        parts.append(caption)
    parts.append(f"Generated {datetime.now().strftime('%Y-%m')}")
    fig.text(x, y, "  ·  ".join(parts),
             fontsize=7, color=PAL["muted"], va="bottom", ha="right",
             transform=fig.transFigure, style="italic")


def _style(ax, xlabel="", show_xgrid=False):
    ax.grid(axis="x" if show_xgrid else "y",
            color=PAL["grid"], linewidth=0.8, zorder=0)
    ax.tick_params(axis="both", length=0, labelsize=9, colors=PAL["muted"])
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color=PAL["muted"], labelpad=6)


def _bar_labels_h(ax, bars):
    xlim = ax.get_xlim()[1]
    for bar in bars:
        w = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        if w / xlim >= 0.55:
            ax.text(w - xlim * 0.015, y, f"{int(w)}",
                    va="center", ha="right", fontsize=8.5,
                    color="white", fontweight="bold")
        else:
            ax.text(w + xlim * 0.008, y, f"{int(w)}",
                    va="center", ha="left", fontsize=8.5,
                    color=PAL["muted"])


def _bar_labels_v(ax, bars):
    ylim = ax.get_ylim()[1]
    for bar in bars:
        h = bar.get_height()
        x = bar.get_x() + bar.get_width() / 2
        ax.text(x, h + ylim * 0.01, f"{int(h)}",
                va="bottom", ha="center", fontsize=8, color=PAL["muted"])


def _save(fig, path, dpi):
    fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


# ── Data loading ──────────────────────────────────────────────────────────────
def load(xlsx_path, sheet):
    df = pd.read_excel(xlsx_path, sheet_name=sheet)
    # Clean '--' and 'Stage X' as unknown
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].replace({"'--": np.nan, "Stage X": np.nan})
    return df


# ── Plot 1: ER / PR status ────────────────────────────────────────────────────
def plot_er_pr(df, out, caption, study_title, dpi):
    status_cols = {"ER Status": "er_status", "PR Status": "pr_status"}
    cats = ["Positive", "Negative", "Indeterminate"]
    colors = [PAL["pos"], PAL["neg"], PAL["indet"]]

    x = np.arange(len(status_cols))
    bar_w = 0.22
    n_cats = len(cats)

    fig, ax = plt.subplots(figsize=(8, 5))
    _bg(fig, [ax])

    for ci, (cat, col) in enumerate(zip(cats, colors)):
        counts = [
            (df[c].value_counts().get(cat, 0))
            for c in status_cols.values()
        ]
        offset = (ci - n_cats / 2 + 0.5) * bar_w
        bars = ax.bar(x + offset, counts, width=bar_w * 0.88,
                      color=col, edgecolor="none", label=cat, zorder=3)
        _bar_labels_v(ax, bars)

    ax.set_xticks(x)
    ax.set_xticklabels(list(status_cols.keys()),
                       fontsize=11, color=PAL["text"], fontweight="bold")
    ax.set_ylim(0, df.shape[0] * 0.95)
    _style(ax)
    ax.yaxis.set_major_locator(MaxNLocator(6))

    legend = ax.legend(fontsize=9, framealpha=0.9,
                       facecolor=PAL["bg"], edgecolor=PAL["grid"])
    for t in legend.get_texts():
        t.set_color(PAL["text"])

    _title(fig,
           main="ER and PR receptor status distribution",
           sub=f"n = {df.shape[0]} samples  |  missing values excluded from counts")
    _caption(fig, caption, study_title)

    fig.subplots_adjust(left=0.08, right=0.97, top=0.83, bottom=0.12)
    _save(fig, out, dpi)


# ── Plot 2: AJCC pathologic stage ─────────────────────────────────────────────
def plot_stage(df, out, caption, study_title, dpi):
    vc = df["ajcc_pathologic_stage"].dropna().value_counts()
    # sort by stage order heuristic
    stage_order = ["Stage I", "Stage IA", "Stage IB", "Stage II",
                   "Stage IIA", "Stage IIB", "Stage III", "Stage IIIA",
                   "Stage IIIB", "Stage IIIC"]
    ordered = [s for s in stage_order if s in vc.index]
    remaining = [s for s in vc.index if s not in ordered]
    labels = ordered + remaining
    counts = [vc[l] for l in labels]

    n = len(labels)
    colors = [STAGE_COLORS[i % len(STAGE_COLORS)] for i in range(n)]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    _bg(fig, [ax])

    bars = ax.barh(labels[::-1], counts[::-1],
                   color=colors[::-1], edgecolor="none", height=0.65, zorder=3)
    _bar_labels_h(ax, bars)

    _style(ax, show_xgrid=True)
    ax.set_xlabel("Number of samples", fontsize=9, color=PAL["muted"], labelpad=6)

    _title(fig,
           main="AJCC pathologic stage distribution",
           sub=f"n = {sum(counts)} samples with stage annotation")
    _caption(fig, caption, study_title)

    fig.subplots_adjust(left=0.18, right=0.92, top=0.83, bottom=0.10)
    _save(fig, out, dpi)


# ── Plot 3: T category ────────────────────────────────────────────────────────
def plot_t_category(df, out, caption, study_title, dpi):
    t_order = ["T1a", "T1b", "T1c", "T1", "T2", "T2b", "T3", "T3a",
               "T4", "T4b", "T4d", "TX"]
    vc = df["ajcc_pathologic_t"].dropna().value_counts()
    ordered = [t for t in t_order if t in vc.index]
    remaining = [t for t in vc.index if t not in ordered]
    labels = ordered + remaining
    counts = [vc[l] for l in labels]

    n = len(labels)
    # gradient from light to dark blue
    blues = [STAGE_COLORS[min(i, len(STAGE_COLORS) - 1)] for i in range(n)]

    fig, ax = plt.subplots(figsize=(9, 5))
    _bg(fig, [ax])

    bars = ax.bar(labels, counts,
                  color=blues, edgecolor="none", zorder=3)
    _bar_labels_v(ax, bars)

    _style(ax)
    ax.set_ylim(0, max(counts) * 1.18)
    ax.tick_params(axis="x", labelsize=9, colors=PAL["text"])

    _title(fig,
           main="Tumour T-category distribution",
           sub=f"n = {sum(counts)} samples with T-category annotation")
    _caption(fig, caption, study_title)

    fig.subplots_adjust(left=0.08, right=0.97, top=0.83, bottom=0.12)
    _save(fig, out, dpi)


# ── Plot 4: N category ────────────────────────────────────────────────────────
def plot_n_category(df, out, caption, study_title, dpi):
    vc = df["ajcc_pathologic_n"].dropna().value_counts().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    _bg(fig, [ax])

    colors = [PAL["pos"] if "N0" in l else PAL["neg"] if "N3" in l else PAL["bar"]
              for l in vc.index]
    bars = ax.barh(vc.index, vc.values,
                   color=colors, edgecolor="none", height=0.65, zorder=3)
    _bar_labels_h(ax, bars)
    _style(ax, show_xgrid=True)
    ax.set_xlabel("Number of samples", fontsize=9, color=PAL["muted"], labelpad=6)

    _title(fig,
           main="Nodal N-category distribution",
           sub=f"n = {vc.sum()} samples  |  N0 = node-negative (blue)  |  N3 = advanced nodal (orange)")
    _caption(fig, caption, study_title)

    fig.subplots_adjust(left=0.18, right=0.92, top=0.83, bottom=0.10)
    _save(fig, out, dpi)


# ── Plot 5: Age histogram ─────────────────────────────────────────────────────
def plot_age(df, out, caption, study_title, dpi):
    ages = df["age_at_index"].dropna()

    fig, ax = plt.subplots(figsize=(9, 5))
    _bg(fig, [ax])

    bins = np.arange(20, 96, 5)
    ax.hist(ages, bins=bins, color=PAL["bar"], edgecolor="white",
            linewidth=0.6, zorder=3, alpha=0.9)

    # KDE overlay
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(ages, bw_method=0.25)
    xs = np.linspace(ages.min(), ages.max(), 300)
    kde_vals = kde(xs) * len(ages) * 5   # scale to histogram height
    ax2 = ax.twinx()
    ax2.set_facecolor(PAL["bg"])
    ax2.plot(xs, kde_vals, color=PAL["kde"], linewidth=2.0, zorder=4)
    ax2.set_yticks([])
    for sp in ax2.spines.values():
        sp.set_visible(False)

    # Median line
    med = ages.median()
    ax.axvline(med, color=PAL["muted"], linewidth=1.2, linestyle="--", zorder=5)
    ax.text(med + 0.5, ax.get_ylim()[1] * 0.92,
            f"median = {med:.0f}", fontsize=8, color=PAL["muted"])

    _style(ax, xlabel="Age at index (years)")
    ax.set_ylabel("Count", fontsize=9, color=PAL["muted"])
    ax.yaxis.set_major_locator(MaxNLocator(6))

    _title(fig,
           main="Age at index distribution",
           sub=f"n = {len(ages)} samples  |  mean = {ages.mean():.1f}  |  std = {ages.std():.1f}  |  red curve = KDE")
    _caption(fig, caption, study_title)

    fig.subplots_adjust(left=0.09, right=0.97, top=0.83, bottom=0.12)
    _save(fig, out, dpi)


# ── Plot 6: Age by ER status ──────────────────────────────────────────────────
def plot_age_by_er(df, out, caption, study_title, dpi):
    groups = {
        "Positive": df[df["er_status"] == "Positive"]["age_at_index"].dropna(),
        "Negative": df[df["er_status"] == "Negative"]["age_at_index"].dropna(),
    }
    cols = [PAL["pos"], PAL["neg"]]
    bins = np.arange(20, 96, 5)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    _bg(fig, axes)

    for ax, (label, ages), col in zip(axes, groups.items(), cols):
        ax.hist(ages, bins=bins, color=col, edgecolor="white",
                linewidth=0.5, zorder=3, alpha=0.90)
        med = ages.median()
        ax.axvline(med, color=PAL["muted"], linewidth=1.2, linestyle="--", zorder=5)
        ax.text(med + 0.5, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 10,
                f"med={med:.0f}", fontsize=8, color=PAL["muted"])
        _style(ax, xlabel="Age at index (years)")
        ax.set_title(f"ER {label}  (n={len(ages)})",
                     fontsize=10, color=PAL["text"], fontweight="bold", pad=8)
        ax.yaxis.set_major_locator(MaxNLocator(5))

    axes[0].set_ylabel("Count", fontsize=9, color=PAL["muted"])

    _title(fig,
           main="Age distribution by ER receptor status",
           sub="Dashed line = group median")
    _caption(fig, caption, study_title)

    fig.subplots_adjust(left=0.07, right=0.97, top=0.83, bottom=0.12, wspace=0.08)
    _save(fig, out, dpi)


# ── Plot 7: ER status × stage heatmap ────────────────────────────────────────
def plot_receptor_stage_heatmap(df, out, caption, study_title, dpi):
    stage_order = ["Stage I", "Stage IA", "Stage IB", "Stage II",
                   "Stage IIA", "Stage IIB", "Stage III",
                   "Stage IIIA", "Stage IIIB", "Stage IIIC"]
    er_order = ["Positive", "Negative"]

    ct = pd.crosstab(df["er_status"], df["ajcc_pathologic_stage"])
    ct = ct.reindex(index=[e for e in er_order if e in ct.index],
                    columns=[s for s in stage_order if s in ct.columns],
                    fill_value=0)

    nrows, ncols = ct.shape
    C0 = (0xFF, 0xFF, 0xFF)
    C1 = (0xC6, 0xDB, 0xEF)
    C2 = (0x4A, 0x7F, 0xB5)
    C3 = (0x08, 0x30, 0x6B)
    STOPS = [(0.00, C0), (0.30, C1), (0.65, C2), (1.00, C3)]

    vmax = ct.values.max()
    vmin = 0

    def _cell_col(val):
        norm = (val - vmin) / (vmax - vmin) if vmax > vmin else 0
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

    fig_w = max(10, 1.3 * ncols)
    fig, ax = plt.subplots(figsize=(fig_w, 3.5))
    _bg(fig, [ax])

    for i, er in enumerate(ct.index):
        for j, stage in enumerate(ct.columns):
            val = ct.loc[er, stage]
            norm = (val - vmin) / (vmax - vmin) if vmax > vmin else 0
            cell_col = _cell_col(val)
            ax.add_patch(mpatches.FancyBboxPatch(
                (j + 0.05, i + 0.05), 0.90, 0.90,
                boxstyle="round,pad=0.02",
                facecolor=cell_col, edgecolor="none", zorder=2,
            ))
            text_col = "white" if norm > 0.40 else PAL["text"]
            ax.text(j + 0.5, i + 0.5, str(val),
                    ha="center", va="center",
                    fontsize=10, fontweight="bold", color=text_col, zorder=3)

    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nrows)
    ax.set_xticks([j + 0.5 for j in range(ncols)])
    ax.set_xticklabels(ct.columns, fontsize=8.5, color=PAL["text"],
                       rotation=30, ha="right")
    ax.set_yticks([i + 0.5 for i in range(nrows)])
    ax.set_yticklabels(ct.index, fontsize=10, color=PAL["text"], fontweight="bold")
    ax.tick_params(length=0)

    # Colour scale strip
    grad_ax = fig.add_axes([0.91, 0.18, 0.018, 0.55])
    grad_ax.set_facecolor(PAL["bg"])
    grad_data = np.linspace(0, 1, 256).reshape(256, 1)
    grad_ax.imshow(grad_data[::-1], aspect="auto",
                   extent=[0, 1, vmin, vmax],
                   cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
                       "custom", ["#FFFFFF", "#C6DBEF", "#4A7FB5", "#08306B"]))
    grad_ax.set_xticks([])
    grad_ax.yaxis.tick_right()
    grad_ax.tick_params(labelsize=7, colors=PAL["muted"], length=0)
    for sp in grad_ax.spines.values():
        sp.set_visible(False)

    _title(fig,
           main="ER status × AJCC pathologic stage  ·  sample counts",
           sub="Cell = number of samples in that (ER status, stage) group")
    _caption(fig, caption, study_title)

    fig.subplots_adjust(left=0.10, right=0.89, top=0.78, bottom=0.22)
    _save(fig, out, dpi)


# ── Plot 8: Summary table ─────────────────────────────────────────────────────
def plot_summary_table(df, out, caption, study_title, dpi):
    ages = df["age_at_index"].dropna()
    rows = [
        ["Total samples",          f"{df.shape[0]:,}"],
        ["Unique patients",         f"{df['patient_id'].nunique():,}"],
        ["Age – mean ± std",        f"{ages.mean():.1f} ± {ages.std():.1f}"],
        ["Age – median [IQR]",
         f"{ages.median():.0f}  [{ages.quantile(0.25):.0f} – {ages.quantile(0.75):.0f}]"],
        ["ER Positive",
         f"{(df['er_status']=='Positive').sum():,}  ({(df['er_status']=='Positive').mean()*100:.1f}%)"],
        ["ER Negative",
         f"{(df['er_status']=='Negative').sum():,}  ({(df['er_status']=='Negative').mean()*100:.1f}%)"],
        ["PR Positive",
         f"{(df['pr_status']=='Positive').sum():,}  ({(df['pr_status']=='Positive').mean()*100:.1f}%)"],
        ["PR Negative",
         f"{(df['pr_status']=='Negative').sum():,}  ({(df['pr_status']=='Negative').mean()*100:.1f}%)"],
        ["Missing age",             f"{df['age_at_index'].isna().sum():,}"],
        ["Missing ER status",       f"{df['er_status'].isna().sum():,}"],
        ["Missing stage",           f"{df['ajcc_pathologic_stage'].isna().sum():,}"],
    ]

    fig, ax = plt.subplots(figsize=(8, 6))
    _bg(fig, [ax])
    ax.axis("off")

    col_widths = [0.60, 0.40]
    row_h = 0.072
    start_y = 0.88

    # Header
    for j, (hdr, w) in enumerate(zip(["Variable", "Value"], col_widths)):
        x = sum(col_widths[:j])
        ax.text(x + 0.01, start_y + row_h * 0.3, hdr,
                fontsize=10, fontweight="bold", color=PAL["text"],
                transform=ax.transAxes)
    # accent bar under header
    ax.add_patch(plt.Rectangle((0, start_y), 1, 0.005,
                 transform=ax.transAxes, facecolor=PAL["accent"],
                 edgecolor="none", zorder=5, clip_on=False))
    ax.add_patch(plt.Rectangle((0, start_y + row_h), 1, 0.002,
                 transform=ax.transAxes, facecolor=PAL["grid"],
                 edgecolor="none", zorder=5, clip_on=False))

    for ri, (var, val) in enumerate(rows):
        y = start_y - (ri + 1) * row_h
        bg_col = "#F7FBFF" if ri % 2 == 0 else PAL["bg"]
        ax.add_patch(plt.Rectangle(
            (0, y), 1, row_h,
            transform=ax.transAxes,
            facecolor=bg_col, edgecolor="none", zorder=1,
        ))
        ax.text(0.01, y + row_h * 0.3, var,
                fontsize=9, color=PAL["text"],
                transform=ax.transAxes)
        ax.text(col_widths[0] + 0.01, y + row_h * 0.3, val,
                fontsize=9, color=PAL["text"], fontweight="bold",
                transform=ax.transAxes)
        ax.add_patch(plt.Rectangle((0, y), 1, 0.001,
                     transform=ax.transAxes, facecolor=PAL["grid"],
                     edgecolor="none", zorder=2, clip_on=False))

    _title(fig,
           main="Cohort summary statistics",
           sub="Key descriptive statistics for the TCGA clinical dataset")
    _caption(fig, caption, study_title)

    fig.subplots_adjust(left=0.05, right=0.97, top=0.83, bottom=0.05)
    _save(fig, out, dpi)


# ── Plot 9: HER2 label distribution ──────────────────────────────────────────
def plot_her2_distribution(labels_csv, out, caption, study_title, dpi):
    ldf = pd.read_csv(labels_csv)
    vc  = ldf["label"].value_counts().sort_index()

    n_neg = vc.get(0, 0)
    n_pos = vc.get(1, 0)
    total = n_neg + n_pos
    labels  = ["HER2-Negative (0)", "HER2-Positive (1)"]
    counts  = [n_neg, n_pos]
    colors  = [PAL["neg"], PAL["pos"]]
    pcts    = [n_neg / total * 100, n_pos / total * 100]

    # ── left: bar chart ───────────────────────────────────────────────────────
    fig, (ax_bar, ax_pie) = plt.subplots(1, 2, figsize=(12, 5),
                                          gridspec_kw={"width_ratios": [1.4, 1]})
    _bg(fig, [ax_bar, ax_pie])

    bars = ax_bar.barh(labels[::-1], counts[::-1],
                       color=colors[::-1], edgecolor="none", height=0.50, zorder=3)

    xlim = max(counts) * 1.22
    ax_bar.set_xlim(0, xlim)
    _style(ax_bar, show_xgrid=True)
    ax_bar.set_xlabel("Number of samples", fontsize=9, color=PAL["muted"], labelpad=6)

    for bar, cnt, pct in zip(bars, counts[::-1], pcts[::-1]):
        w = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        # count inside bar
        ax_bar.text(w - xlim * 0.015, y, f"{cnt:,}",
                    va="center", ha="right", fontsize=11,
                    color="white", fontweight="bold")
        # percentage outside
        ax_bar.text(w + xlim * 0.015, y, f"{pct:.1f}%",
                    va="center", ha="left", fontsize=10, color=PAL["muted"])

    # imbalance ratio annotation
    ratio = n_neg / n_pos if n_pos > 0 else float("inf")
    ax_bar.text(0.98, 0.06,
                f"Imbalance ratio  {ratio:.2f} : 1",
                transform=ax_bar.transAxes,
                fontsize=9, color=PAL["muted"], ha="right", style="italic")

    # ── right: donut chart ────────────────────────────────────────────────────
    wedge_props = dict(width=0.45, edgecolor="white", linewidth=2.5)
    wedges, _ = ax_pie.pie(
        counts, colors=colors,
        startangle=90, counterclock=False,
        wedgeprops=wedge_props,
    )
    # centre text
    ax_pie.text(0, 0, f"n={total:,}", ha="center", va="center",
                fontsize=12, fontweight="bold", color=PAL["text"])

    # legend inside pie panel
    legend_patches = [
        mpatches.Patch(color=colors[i], label=f"{labels[i]}  ({pcts[i]:.1f}%)")
        for i in range(2)
    ]
    ax_pie.legend(handles=legend_patches, loc="lower center",
                  bbox_to_anchor=(0.5, -0.12), fontsize=9,
                  framealpha=0.9, facecolor=PAL["bg"], edgecolor=PAL["grid"])

    _title(fig,
           main="HER2 label distribution",
           sub=f"Total = {total:,} samples  |  label 0 = HER2-negative  |  label 1 = HER2-positive")
    _caption(fig, caption, study_title)

    fig.subplots_adjust(left=0.08, right=0.97, top=0.83, bottom=0.12, wspace=0.30)
    _save(fig, out, dpi)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_xlsx",  required=True)
    parser.add_argument("--sheet",       default=0,
                        help="Sheet name or index (default: first sheet)")
    parser.add_argument("--labels_csv",  default=None,
                        help="Labels CSV (patient_id, label). If omitted, auto-detected from same folder as --input_xlsx")
    parser.add_argument("--output_dir",  required=True)
    parser.add_argument("--caption",     default="",
                        help="Dataset / source line bottom-right of every figure")
    parser.add_argument("--study_title", default="",
                        help="Short study name embedded in captions")
    parser.add_argument("--dpi",         type=int, default=300)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sheet = int(args.sheet) if str(args.sheet).isdigit() else args.sheet
    df = load(args.input_xlsx, sheet)

    kw = dict(caption=args.caption, study_title=args.study_title, dpi=args.dpi)

    plot_er_pr(df,                  out / "er_pr_status_bar.png",          **kw)
    plot_stage(df,                  out / "pathologic_stage_bar.png",       **kw)
    plot_t_category(df,             out / "t_category_bar.png",             **kw)
    plot_n_category(df,             out / "n_category_bar.png",             **kw)
    plot_age(df,                    out / "age_histogram.png",              **kw)
    plot_age_by_er(df,              out / "age_by_er_status.png",           **kw)
    plot_receptor_stage_heatmap(df, out / "receptor_stage_heatmap.png",     **kw)
    plot_summary_table(df,          out / "summary_table.png",              **kw)

    # Auto-detect labels CSV if not explicitly provided
    labels_csv = args.labels_csv
    if not labels_csv:
        xlsx_dir = Path(args.input_xlsx).parent
        candidates = list(xlsx_dir.glob("*.csv"))
        for c in candidates:
            try:
                cols = pd.read_csv(c, nrows=1).columns.tolist()
                if "label" in cols:
                    labels_csv = str(c)
                    print(f"Auto-detected labels CSV: {labels_csv}")
                    break
            except Exception:
                continue

    if labels_csv:
        plot_her2_distribution(labels_csv, out / "her2_label_distribution.png", **kw)
    else:
        print("No labels CSV found — skipping HER2 distribution plot.")

    print("Done. Output:", out)


if __name__ == "__main__":
    main()
