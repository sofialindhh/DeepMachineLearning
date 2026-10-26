
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

RESULTS_DIR = "study_B/results"

icl = pd.read_csv(
    f"{RESULTS_DIR}/icl_restricted_context_predictions.csv"
)
pfn = pd.read_csv(
    f"{RESULTS_DIR}/pfn_restricted_context_predictions.csv"
)

# Match same dataset, K och testpoint
keys = ["dataset", "context_size", "test_index"]

merged = icl.merge(
    pfn,
    on=keys,
    how="inner",
    suffixes=("_icl", "_pfn"),
    validate="one_to_one",
)

# Check that true value is correct between models
assert (merged["y_true_icl"] == merged["y_true_pfn"]).all()

rows = []

for dataset, group in merged.groupby("dataset"):
    y_true = group["y_true_icl"]

    for model in ["icl", "pfn"]:
        y_pred = group[f"y_pred_{model}"]

        rows.append({
            "dataset": dataset,
            "model": model.upper(),
            "context_size": int(group["context_size"].iloc[0]),
            "n_test_common": len(group),
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(
                y_true, y_pred
            ),
            "macro_f1": f1_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
        })

results = pd.DataFrame(rows)

print("\nMatched comparison: same test points")
print(results.to_string(index=False, float_format="%.4f"))

results.to_csv(
    f"{RESULTS_DIR}/restricted_context_matched_comparison.csv",
    index=False,
)

print("\nSaved to results/restricted_context_matched_comparison.csv")
