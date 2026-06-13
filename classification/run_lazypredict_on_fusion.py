#pip install lazypredict

# ============================================================
# LazyPredict Classifier Evaluation for Fused Embeddings
# ============================================================
#
# Purpose:
#   Run LazyClassifier on fused .npy embeddings produced by any
#   fusion method: concat, gated attention, cross-attention,
#   gated cross-attention, etc.
#
# Expected fused_dir structure:
#   fused_dir/
#       fused_train/
#           patient001.npy
#       fused_val/
#           patient101.npy
#       fused_test/
#           patient201.npy
#       train_patients.csv
#       val_patients.csv
#       test_patients.csv
#
# Example:
#   python run_lazypredict_on_fusion.py ^
#     --fused_dir "D:\embeddings\fused_concat" ^
#     --output_dir "D:\results\lazy_fused_concat"
#
# Example with final test evaluation:
#   python run_lazypredict_on_fusion.py ^
#     --fused_dir "D:\embeddings\fused_gated_attention" ^
#     --output_dir "D:\results\lazy_gated_attention" ^
#     --evaluate_test
#
# Arguments:
#   --fused_dir       Folder containing fused_train, fused_val, fused_test and split CSV files.
#   --output_dir      Folder where LazyPredict results will be saved.
#   --patient_col     Patient ID column name. Default: patient_id
#   --label_col       Label column name. Default: label
#   --standardize     Apply StandardScaler fitted on train only. Default: True
#   --no_standardize  Disable standardization.
#   --evaluate_test   Also run final LazyPredict evaluation on train+val vs test.
#   --random_state    Random seed. Default: 42
#   --verbose         LazyPredict verbosity. Default: 0
#
# Outputs:
#   output_dir/
#       lazy_val_results.csv
#       lazy_val_predictions.csv
#       run_summary.txt
#
#   If --evaluate_test
#       lazy_test_results.csv
#       lazy_test_predictions.csv
#
# ============================================================

#pip install lazypredict

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from lazypredict.Supervised import LazyClassifier


# ============================================================
# Data loading
# ============================================================

def load_split_embeddings(
    fused_split_dir,
    split_csv,
    patient_col="patient_id",
    label_col="label",
):
    df = pd.read_csv(split_csv)

    if patient_col not in df.columns:
        raise ValueError(f"Missing patient column: {patient_col}")

    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")

    X_list = []
    y_list = []
    patient_ids = []

    for _, row in df.iterrows():
        patient_id = str(row[patient_col])
        label = int(row[label_col])

        emb_path = Path(fused_split_dir) / f"{patient_id}.npy"

        if not emb_path.exists():
            print(f"[WARN] Missing fused embedding, skipping: {patient_id}")
            continue

        emb = np.load(emb_path).astype(np.float32).reshape(-1)

        X_list.append(emb)
        y_list.append(label)
        patient_ids.append(patient_id)

    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.int64)

    return X, y, patient_ids


# ============================================================
# LazyPredict runner
# ============================================================

def run_lazypredict(
    X_train,
    X_eval,
    y_train,
    y_eval,
    output_results_csv,
    output_predictions_csv,
    random_state=42,
    verbose=0,
):
    clf = LazyClassifier(
        verbose=verbose,
        ignore_warnings=True,
        custom_metric=None,
        predictions=True,
        random_state=random_state,
    )

    models, predictions = clf.fit(
        X_train,
        X_eval,
        y_train,
        y_eval,
    )

    models.to_csv(output_results_csv)
    predictions.to_csv(output_predictions_csv, index=False)

    return models, predictions


# ============================================================
# Main
# ============================================================

def run_single(fused_dir, output_dir, args):
    # --------------------------------------------------------
    # Run LazyPredict evaluation for one fused_dir.
    # --------------------------------------------------------

    tag = fused_dir.name

    print(f"\n{'='*60}")
    print(f"Processing: {tag}")
    print(f"{'='*60}")

    # CSVs are normally copied into fused_dir by the fusion scripts.
    # If missing (e.g. wsi_only / clinical_only didn't copy them),
    # fall back to --split_dir if provided.
    def resolve_csv(filename):
        p = fused_dir / filename
        if p.exists():
            return p
        if args.split_dir:
            p2 = Path(args.split_dir) / filename
            if p2.exists():
                print(f"[INFO] {filename} not in fused_dir, using split_dir fallback.")
                return p2
        raise FileNotFoundError(
            f"Missing {filename}: not in {fused_dir} and no --split_dir fallback provided."
        )

    train_csv = resolve_csv("train_patients.csv")
    val_csv   = resolve_csv("val_patients.csv")
    test_csv  = resolve_csv("test_patients.csv")

    fused_train_dir = fused_dir / "fused_train"
    fused_val_dir   = fused_dir / "fused_val"
    fused_test_dir  = fused_dir / "fused_test"

    for path in [fused_train_dir, fused_val_dir, fused_test_dir]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required path: {path}")

    print("Loading fused embeddings...")

    X_train, y_train, train_ids = load_split_embeddings(
        fused_train_dir, train_csv, args.patient_col, args.label_col,
    )
    X_val, y_val, val_ids = load_split_embeddings(
        fused_val_dir, val_csv, args.patient_col, args.label_col,
    )
    X_test, y_test, test_ids = load_split_embeddings(
        fused_test_dir, test_csv, args.patient_col, args.label_col,
    )

    print("Train:", X_train.shape, y_train.shape)
    print("Val:  ", X_val.shape,   y_val.shape)
    print("Test: ", X_test.shape,  y_test.shape)

    # Standardization must be fitted only on training data.
    if args.standardize:
        scaler = StandardScaler()
        X_train_used = scaler.fit_transform(X_train)
        X_val_used   = scaler.transform(X_val)
        X_test_used  = scaler.transform(X_test)
    else:
        X_train_used = X_train
        X_val_used   = X_val
        X_test_used  = X_test

    # --------------------------------------------------------
    # Main LazyPredict screening:
    # Train on train split, evaluate on validation split.
    # --------------------------------------------------------

    print("\nRunning LazyPredict on validation split...")

    val_results, val_predictions = run_lazypredict(
        X_train=X_train_used,
        X_eval=X_val_used,
        y_train=y_train,
        y_eval=y_val,
        output_results_csv=output_dir / f"lazy_val_results_{tag}.csv",
        output_predictions_csv=output_dir / f"lazy_val_predictions_{tag}.csv",
        random_state=args.random_state,
        verbose=args.verbose,
    )

    print("\nValidation results:")
    print(val_results)

    # --------------------------------------------------------
    # Optional final test evaluation:
    # Train on train+val, evaluate on test.
    # Use this only when you are ready to check final performance.
    # --------------------------------------------------------

    if args.evaluate_test:
        print("\nRunning LazyPredict on final test split...")

        X_trainval = np.concatenate([X_train, X_val], axis=0)
        y_trainval = np.concatenate([y_train, y_val], axis=0)

        if args.standardize:
            scaler_test = StandardScaler()
            X_trainval_used    = scaler_test.fit_transform(X_trainval)
            X_test_final_used  = scaler_test.transform(X_test)
        else:
            X_trainval_used   = X_trainval
            X_test_final_used = X_test

        test_results, test_predictions = run_lazypredict(
            X_train=X_trainval_used,
            X_eval=X_test_final_used,
            y_train=y_trainval,
            y_eval=y_test,
            output_results_csv=output_dir / f"lazy_test_results_{tag}.csv",
            output_predictions_csv=output_dir / f"lazy_test_predictions_{tag}.csv",
            random_state=args.random_state,
            verbose=args.verbose,
        )

        print("\nTest results:")
        print(test_results)

    # --------------------------------------------------------
    # Save run summary.
    # --------------------------------------------------------

    with open(output_dir / f"run_summary_{tag}.txt", "w", encoding="utf-8") as f:
        f.write("LazyPredict Fusion Evaluation Summary\n")
        f.write("=====================================\n\n")
        f.write(f"Fused directory: {fused_dir}\n")
        f.write(f"Output directory: {output_dir}\n\n")
        f.write(f"Train shape: {X_train.shape}\n")
        f.write(f"Val shape:   {X_val.shape}\n")
        f.write(f"Test shape:  {X_test.shape}\n\n")
        f.write(f"Standardize:   {args.standardize}\n")
        f.write(f"Evaluate test: {args.evaluate_test}\n")
        f.write(f"Random state:  {args.random_state}\n\n")
        f.write("Train class distribution:\n")
        f.write(str(pd.Series(y_train).value_counts()))
        f.write("\n\nVal class distribution:\n")
        f.write(str(pd.Series(y_val).value_counts()))
        f.write("\n\nTest class distribution:\n")
        f.write(str(pd.Series(y_test).value_counts()))
        f.write("\n")

    print(f"\nFinished: {tag}")
    print("Results saved to:", output_dir)


def main():
    parser = argparse.ArgumentParser()

    # Accept one or more fused dirs in a single run.
    parser.add_argument("--fused_dirs", nargs="+", required=True, type=str,
                        help="One or more fused embedding directories.")
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--split_dir", default=None, type=str,
                        help="Fallback split dir if train/val/test CSVs are missing from fused_dir.")

    parser.add_argument("--patient_col", default="patient_id", type=str)
    parser.add_argument("--label_col",   default="label",      type=str)

    parser.add_argument("--standardize",    action="store_true",  default=True)
    parser.add_argument("--no_standardize", action="store_false", dest="standardize")

    parser.add_argument("--evaluate_test", action="store_true")

    parser.add_argument("--random_state", default=42, type=int)
    parser.add_argument("--verbose",      default=0,  type=int)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fused_dirs = [Path(d) for d in args.fused_dirs]

    print(f"Running LazyPredict on {len(fused_dirs)} fused dir(s)...")

    for fused_dir in fused_dirs:
        run_single(fused_dir, output_dir, args)

    print(f"\nAll done. {len(fused_dirs)} run(s) completed.")
    print("Output directory:", output_dir)


if __name__ == "__main__":
    main()
