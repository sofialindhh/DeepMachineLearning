
import os
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

RESULTS_DIR = "results"

# Load existing prediction files


full = pd.read_csv(f"{RESULTS_DIR}/predictions.csv")
icl = pd.read_csv(
    f"{RESULTS_DIR}/icl_restricted_context_predictions.csv"
)
pfn = pd.read_csv(
    f"{RESULTS_DIR}/pfn_restricted_context_predictions.csv"
)

keys = ["dataset", "test_index"]

# Check that there are no duplicate predictions per test point.
for name, df in [
    ("full", full),
    ("ICL restricted", icl),
    ("PFN restricted", pfn),
]:
    if df.duplicated(keys).any():
        raise ValueError(
            f"Duplicate dataset/test_index rows in {name} predictions."
        )


# Keep only test points predicted by BOTH ICL and PFN
# in the restricted-context experiment


common = icl.merge(
    pfn,
    on=keys,
    how="inner",
    suffixes=("_icl", "_pfn"),
    validate="one_to_one",
)

assert (
    common["y_true_icl"] == common["y_true_pfn"]
).all()


# Match full-context predictions to those same points


full_common = full.merge(
    common[keys + ["y_true_icl"]],
    on=keys,
    how="inner",
    validate="one_to_one",
)

assert (full_common["y_true"] == full_common["y_true_icl"]).all()

# Ensure all expected common points were found in full predictions.
for dataset, group in common.groupby("dataset"):
    n_full = (full_common["dataset"] == dataset).sum()
    if n_full != len(group):
        raise ValueError(
            f"{dataset}: found {n_full} full-context rows, "
            f"but expected {len(group)} common test points."
        )

# Calculate metrics for full and restricted context


def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, y_pred
        ),
        "macro_f1": f1_score(
            y_true, y_pred,
            average="macro",
            zero_division=0,
        ),
    }


rows = []

for dataset, group in common.groupby("dataset"):
    full_group = full_common[
        full_common["dataset"] == dataset
    ]

    # Full-context models, evaluated on common test points.
    full_models = {
        "KNN": "knn_pred",
        "PFN": "tabpfn_pred",
        "ICL": "tabicl_pred",
    }

    for model, column in full_models.items():
        metrics = calculate_metrics(
            full_group["y_true"],
            full_group[column],
        )

        rows.append({
            "dataset": dataset,
            "model": model,
            "context": "Full",
            "n_test": len(full_group),
            **metrics,
        })

    # Restricted-context models, same test points.
    for model, group_model in [
        ("PFN", group),
        ("ICL", group),
    ]:
        pred_col = f"y_pred_{model.lower()}"
        true_col = f"y_true_{model.lower()}"

        metrics = calculate_metrics(
            group_model[true_col],
            group_model[pred_col],
        )

        rows.append({
            "dataset": dataset,
            "model": model,
            "context": "K=5",
            "n_test": len(group_model),
            **metrics,
        })

results = pd.DataFrame(rows)


# Calculate changes from Full to K=5 for PFN and ICL

comparison_rows = []

for dataset in results["dataset"].unique():
    for model in ["PFN", "ICL"]:
        subset = results[
            (results["dataset"] == dataset)
            & (results["model"] == model)
        ]

        full_row = subset[subset["context"] == "Full"]
        k5_row = subset[subset["context"] == "K=5"]

        if full_row.empty or k5_row.empty:
            continue

        full_row = full_row.iloc[0]
        k5_row = k5_row.iloc[0]

        comparison_rows.append({
            "dataset": dataset,
            "model": model,
            "n_test": int(k5_row["n_test"]),
            "full_accuracy": full_row["accuracy"],
            "k5_accuracy": k5_row["accuracy"],
            "delta_accuracy": (
                k5_row["accuracy"] - full_row["accuracy"]
            ),
            "full_balanced_accuracy": full_row["balanced_accuracy"],
            "k5_balanced_accuracy": k5_row["balanced_accuracy"],
            "delta_balanced_accuracy": (
                k5_row["balanced_accuracy"]
                - full_row["balanced_accuracy"]
            ),
            "full_macro_f1": full_row["macro_f1"],
            "k5_macro_f1": k5_row["macro_f1"],
            "delta_macro_f1": (
                k5_row["macro_f1"] - full_row["macro_f1"]
            ),
        })

comparison = pd.DataFrame(comparison_rows)


# Save and print results

results.to_csv(
    f"{RESULTS_DIR}/context_metrics_matched.csv",
    index=False,
)

comparison.to_csv(
    f"{RESULTS_DIR}/context_change_matched.csv",
    index=False,
)

print("\n=== Metrics on the same test points ===")
print(results.to_string(
    index=False,
    float_format=lambda x: f"{x:.4f}",
))

print("\n=== Change from Full to K=5 ===")
print(comparison.to_string(
    index=False,
    float_format=lambda x: f"{x:.4f}",
))

print("\nSaved:")
print("results/context_metrics_matched.csv")
print("results/context_change_matched.csv")
