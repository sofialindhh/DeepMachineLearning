import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

from tabpfn import TabPFNClassifier
from tabicl import TabICLClassifier

from main import loadfile
from tabpfn.errors import TabPFNValidationError



# Configuration


MODEL_NAME = "ICL"  # "PFN" or "ICL"
K = 5

DATASETS = [
    (
        "Blood Transfusion",
        "blood_transfusion.arff",
        "DonatedBloodInMarch2007",
    ),
    (
        "Diabetes",
        "diabetes.arff",
        "TestedPositiveForDiabetes",
    ),
    (
        "Maternal Health Risk",
        "maternal_health.arff",
        "RiskLevel",
    ),
    (
        "German Credit",
        "german_credit.arff",
        "good_or_bad_customer",
    ),
    (
        "QSAR Biodegradation",
        "QSAR_biodeg.arff",
        "Biodegradable",
    ),
]

KNN_FILE = "results/knn_neighbors.csv"
RESULTS_DIR = "results"



# Create model


def create_model():
    if MODEL_NAME == "PFN":
        return TabPFNClassifier()

    if MODEL_NAME == "ICL":
        return TabICLClassifier(device="cpu")

    raise ValueError(f"Unknown model: {MODEL_NAME}")



# Evaluate one dataset


def evaluate_dataset(dataset_name, filename, target):
    print(f"\n{'=' * 60}")
    print(f"Dataset: {dataset_name}")
    print(f"Model: {MODEL_NAME}")
    print(f"Context size: {K}")
    print(f"{'=' * 60}")

    # Use exactly the same split as the existing project.
    X_train, X_test, y_train, y_test = loadfile(
        filename, target
    )

    knn_df = pd.read_csv(KNN_FILE)

    dataset_neighbors = knn_df[
        knn_df["dataset"] == dataset_name
    ].copy()

    if dataset_neighbors.empty:
        raise ValueError(
            f"No KNN neighbors found for {dataset_name}"
        )

    predictions = []
    true_labels = []
    prediction_records = []

    single_class_contexts = 0
    failed_predictions = 0
    failed_test_indices = []

    

    # Each test point gets its own K nearest training rows.
    for test_index in X_test.index:
        neighbors = dataset_neighbors[
            dataset_neighbors["test_index"] == test_index
        ].sort_values("neighbor_rank").head(K)

        if len(neighbors) != K:
            raise ValueError(
                f"Expected {K} neighbors for test index "
                f"{test_index}, found {len(neighbors)}"
            )

        neighbor_indices = neighbors["neighbor_index"].tolist()

        # Recover the corresponding training rows and labels.
        X_context = X_train.loc[neighbor_indices]
        y_context = y_train.loc[neighbor_indices]

        # If every neighbor has the same class, the context
        # contains no class ambiguity. Avoid fitting a model
        # that may reject a one-class training set.

        model = create_model()

        try:
            model.fit(X_context, y_context)

            test_row = X_test.loc[[test_index]]
            prediction = model.predict(test_row)[0]

        except TabPFNValidationError as e:
            if MODEL_NAME == "PFN":
                failed_predictions += 1
                failed_test_indices.append(test_index)
                print(
                    f"Skipping test index {test_index}: {e}"
                )
                continue
            raise

        y_true = y_test.loc[test_index]

        predictions.append(prediction)
        true_labels.append(y_true)

        prediction_records.append({
            "dataset": dataset_name,
            "model": MODEL_NAME,
            "context_size": K,
            "test_index": test_index,
            "y_true": y_true,
            "y_pred": prediction,
        })

    # Metrics
    accuracy = accuracy_score(true_labels, predictions)

    balanced_acc = balanced_accuracy_score(
        true_labels, predictions
    )

    macro_f1 = f1_score(
        true_labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    result = {
        "dataset": dataset_name,
        "model": MODEL_NAME,
        "context_size": K,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "single_class_contexts": single_class_contexts,
        "n_test": len(true_labels),
        "failed_predictions": failed_predictions,
    }

    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Balanced accuracy: {balanced_acc:.4f}")
    print(f"Macro F1:          {macro_f1:.4f}")
    print(
        "Single-class contexts: "
        f"{single_class_contexts}/{len(true_labels)}"
    )
    print(f"Failed predictions: {failed_predictions}")
    print(f"Successful predictions: {len(true_labels)}")

    # Save individual predictions
    predictions_df = pd.DataFrame(prediction_records)

    predictions_file = os.path.join(
        RESULTS_DIR,
        f"{MODEL_NAME.lower()}_restricted_context_predictions.csv",
    )

    if os.path.exists(predictions_file):
        existing_df = pd.read_csv(predictions_file)

        # Replace previous results for this dataset and K
        existing_df = existing_df[
            ~(
                (existing_df["dataset"] == dataset_name)
                & (existing_df["context_size"] == K)
        )
        ]

        predictions_df = pd.concat(
            [existing_df, predictions_df],
            ignore_index=True,
        )

    predictions_df.to_csv(predictions_file, index=False)

    # Save failed test indices separately
    failures_file = os.path.join(
        RESULTS_DIR,
        f"{MODEL_NAME.lower()}_restricted_context_failures.csv",
    )

    failure_records = pd.DataFrame({
        "dataset": [dataset_name] * len(failed_test_indices),
        "model": [MODEL_NAME] * len(failed_test_indices),
        "context_size": [K] * len(failed_test_indices),
        "test_index": failed_test_indices,
    })

    if os.path.exists(failures_file):
        existing_failures = pd.read_csv(failures_file)

        existing_failures = existing_failures[
            ~(
                (existing_failures["dataset"] == dataset_name)
                & (existing_failures["context_size"] == K)
            )
        ]

        failure_records = pd.concat(
            [existing_failures, failure_records],
            ignore_index=True,
        )

    failure_records.to_csv(failures_file, index=False)

    print(f"Individual predictions saved to: {predictions_file}")
    print(f"Failed indices saved to: {failures_file}")

    return result



# Run experiment


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = []

    for dataset_name, filename, target in DATASETS:
        result = evaluate_dataset(
            dataset_name,
            filename,
            target,
        )
        results.append(result)

    results_df = pd.DataFrame(results)

    output_file = os.path.join(
        RESULTS_DIR,
        f"{MODEL_NAME.lower()}_restricted_context_results.csv",
    )

    results_df.to_csv(output_file, index=False)

    print("\nFinal results:")
    print(results_df.to_string(index=False))
    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    main()
