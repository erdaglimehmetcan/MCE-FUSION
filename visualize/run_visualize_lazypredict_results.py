# ============================================================
# LazyPredict Result Visualization
# run_visualize_lazypredict_results.py
# ============================================================
#
# Purpose:
#   Visualizes LazyPredict CSV outputs for WSI-only, clinical-only,
#   and fusion embeddings.
#
# Main functions:
#   1. Creates per-run bar plots for:
#        - AUC
#        - F1
#        - Recall
#        - Precision
#        - Accuracy
#
#   2. Writes exact metric values on the plots.
#
#   3. Stores tracking information using optional params.json files.
#
#   4. Compares multiple embedding results by selecting the best
#      classifier from each embedding according to a chosen metric.
#
# Input:
#   One or more LazyPredict CSV files.
#
# Output:
#   output_dir/
#       per_run_plots/
#       comparison_plots/
#       summaries/
#
# Example single CSV:
#   python visualize_lazypredict_results.py ^
#     --input_csvs "D:\results\lazy_wsi_only\lazy_val_results.csv" ^
#     --run_labels "WSI-only" ^
#     --output_dir "D:\results\visualizations"
#
# Example six embeddings:
#   python visualize_lazypredict_results.py ^
#     --input_csvs ^
#       "D:\results\lazy_wsi_only\lazy_val_results.csv" ^
#       "D:\results\lazy_clinical_only\lazy_val_results.csv" ^
#       "D:\results\lazy_concat\lazy_val_results.csv" ^
#       "D:\results\lazy_gated_attention\lazy_val_results.csv" ^
#       "D:\results\lazy_cross_attention\lazy_val_results.csv" ^
#       "D:\results\lazy_gated_cross_attention\lazy_val_results.csv" ^
#     --run_labels ^
#       "WSI-only" ^
#       "Clinical-only" ^
#       "Concat" ^
#       "Gated Attention" ^
#       "Cross-Attention" ^
#       "Gated Cross-Attention" ^
#     --output_dir "D:\results\visualizations"
#
# Important:
#   Change column names using argparse if your CSV columns differ.
# ============================================================

import argparse
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt


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

    if model_col in df.columns:
        pass
    elif df.columns[0].lower().startswith("unnamed"):
        df = df.rename(columns={df.columns[0]: model_col})
    else:
        df = df.rename(columns={df.columns[0]: model_col})

    return df


def safe_metric_columns(args):
    metric_map = {
        "AUC": args.auc_col,
        "F1": args.f1_col,
        "Recall": args.recall_col,
        "Precision": args.precision_col,
        "Accuracy": args.accuracy_col,
    }

    return metric_map


def clean_metric_column(df, col):
    if col not in df.columns:
        return None

    values = pd.to_numeric(df[col], errors="coerce")
    return values


def make_metric_plot(
    df,
    metric_name,
    metric_col,
    model_col,
    run_label,
    params_text,
    output_path,
    top_n,
):
    if metric_col not in df.columns:
        return False

    plot_df = df.copy()
    plot_df[metric_col] = clean_metric_column(plot_df, metric_col)
    plot_df = plot_df.dropna(subset=[metric_col])

    if plot_df.empty:
        return False

    plot_df = plot_df.sort_values(metric_col, ascending=False).head(top_n)
    plot_df = plot_df.sort_values(metric_col, ascending=True)

    fig_height = max(5, 0.45 * len(plot_df))
    fig, ax = plt.subplots(figsize=(12, fig_height))

    bars = ax.barh(plot_df[model_col], plot_df[metric_col])

    ax.set_title(f"{run_label} - {metric_name}", fontsize=14, fontweight="bold")
    ax.set_xlabel(metric_name)
    ax.set_ylabel("Classifier")

    max_value = plot_df[metric_col].max()
    ax.set_xlim(0, min(1.05, max(1.0, max_value + 0.08)))

    for bar in bars:
        width = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        ax.text(
            width + 0.01,
            y,
            f"{width:.4f}",
            va="center",
            fontsize=9,
        )

    if params_text:
        fig.text(
            0.01,
            0.01,
            params_text,
            ha="left",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return True


def params_to_text(params_data):
    if not params_data:
        return ""

    parameters = params_data.get("parameters", params_data)

    useful_keys = [
        "fused_dir",
        "output_dir",
        "wsi_dim",
        "clinical_dim",
        "fused_dim",
        "hidden_dim",
        "num_tokens",
        "num_heads",
        "epochs",
        "batch_size",
        "lr",
        "dropout",
        "seed",
        "split_dir",
    ]

    items = []

    for key in useful_keys:
        if key in parameters:
            items.append(f"{key}={parameters[key]}")

    return " | ".join(items)


def analyze_single_run(
    csv_path,
    run_label,
    params_json,
    args,
    per_run_dir,
    summary_rows,
):
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
            df=df,
            metric_name=metric_name,
            metric_col=metric_col,
            model_col=args.model_col,
            run_label=run_label,
            params_text=params_text,
            output_path=output_path,
            top_n=args.top_n,
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

        summary_rows.append(
            {
                "run_label": run_label,
                "metric": metric_name,
                "best_model": best_row[args.model_col],
                "best_score": float(best_row[metric_col]),
                "csv_path": str(csv_path),
                "params_json": str(params_json) if params_json else "",
                "params_text": params_text,
            }
        )


def make_best_model_comparison(summary_df, metric_name, output_path):
    metric_df = summary_df[summary_df["metric"] == metric_name].copy()

    if metric_df.empty:
        return False

    metric_df = metric_df.sort_values("best_score", ascending=True)

    fig_height = max(5, 0.55 * len(metric_df))
    fig, ax = plt.subplots(figsize=(13, fig_height))

    labels = [
        f"{row['run_label']}\n({row['best_model']})"
        for _, row in metric_df.iterrows()
    ]

    bars = ax.barh(labels, metric_df["best_score"])

    ax.set_title(
        f"Best classifier comparison by {metric_name}",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel(metric_name)
    ax.set_ylabel("Embedding / best classifier")

    max_value = metric_df["best_score"].max()
    ax.set_xlim(0, min(1.05, max(1.0, max_value + 0.08)))

    for bar in bars:
        width = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        ax.text(
            width + 0.01,
            y,
            f"{width:.4f}",
            va="center",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return True


def make_metric_matrix(summary_df, output_path):
    pivot = summary_df.pivot_table(
        index="run_label",
        columns="metric",
        values="best_score",
        aggfunc="max",
    )

    preferred_cols = ["AUC", "F1", "Recall", "Precision", "Accuracy"]
    existing_cols = [c for c in preferred_cols if c in pivot.columns]
    pivot = pivot[existing_cols]

    fig_width = max(8, 1.5 * len(existing_cols))
    fig_height = max(4, 0.6 * len(pivot))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    im = ax.imshow(pivot.values, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticklabels(pivot.index)

    ax.set_title(
        "Best-score matrix across embeddings",
        fontsize=14,
        fontweight="bold",
    )

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.values[i, j]
            ax.text(
                j,
                i,
                f"{value:.4f}",
                ha="center",
                va="center",
                fontsize=9,
            )

    fig.colorbar(im, ax=ax, label="Score")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_csvs", nargs="+", required=True)
    parser.add_argument("--run_labels", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--params_jsons", nargs="*", default=None)

    parser.add_argument("--model_col", default="Model")
    parser.add_argument("--auc_col", default="ROC AUC")
    parser.add_argument("--f1_col", default="F1 Score")
    parser.add_argument("--recall_col", default="Recall")
    parser.add_argument("--precision_col", default="Precision")
    parser.add_argument("--accuracy_col", default="Accuracy")

    parser.add_argument("--top_n", type=int, default=15)
    parser.add_argument("--main_metric", default="AUC")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    per_run_dir = output_dir / "per_run_plots"
    comparison_dir = output_dir / "comparison_plots"
    summary_dir = output_dir / "summaries"

    per_run_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    if len(args.input_csvs) != len(args.run_labels):
        raise ValueError("Number of input_csvs must match number of run_labels.")

    if args.params_jsons is None or len(args.params_jsons) == 0:
        params_jsons = [None] * len(args.input_csvs)
    else:
        if len(args.params_jsons) != len(args.input_csvs):
            raise ValueError("Number of params_jsons must match number of input_csvs.")
        params_jsons = args.params_jsons

    summary_rows = []

    for csv_path, run_label, params_json in zip(
        args.input_csvs,
        args.run_labels,
        params_jsons,
    ):
        analyze_single_run(
            csv_path=csv_path,
            run_label=run_label,
            params_json=params_json,
            args=args,
            per_run_dir=per_run_dir,
            summary_rows=summary_rows,
        )

    summary_df = pd.DataFrame(summary_rows)

    summary_csv = summary_dir / "best_model_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    for metric_name in ["AUC", "F1", "Recall", "Precision", "Accuracy"]:
        output_path = comparison_dir / f"best_model_comparison_{metric_name.lower()}.png"
        make_best_model_comparison(summary_df, metric_name, output_path)

    matrix_path = comparison_dir / "best_score_matrix.png"
    make_metric_matrix(summary_df, matrix_path)

    run_info = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_csvs": args.input_csvs,
        "run_labels": args.run_labels,
        "params_jsons": params_jsons,
        "model_col": args.model_col,
        "metric_columns": {
            "AUC": args.auc_col,
            "F1": args.f1_col,
            "Recall": args.recall_col,
            "Precision": args.precision_col,
            "Accuracy": args.accuracy_col,
        },
        "top_n": args.top_n,
        "main_metric": args.main_metric,
    }

    with open(summary_dir / "visualization_params.json", "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=4)

    print("Visualization finished.")
    print("Output directory:", output_dir)
    print("Summary CSV:", summary_csv)


if __name__ == "__main__":
    main()