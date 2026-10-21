
import os
import numpy as np
import pandas as pd

from tabpfn import TabPFNClassifier
from tabpfn_extensions.interpretability import get_decoder_readout

from tabicl import TabICLClassifier

from main import loadfile

import tabicl._model.attention as attention_module
import math
import torch


MODEL_NAME = "PFN"  # Välj "PFN" eller "ICL"
K = 5

DATASETS = [
    ("Blood Transfusion", "blood_transfusion.arff",
     "DonatedBloodInMarch2007"),

    ("Diabetes", "diabetes.arff",
     "TestedPositiveForDiabetes"),

    ("Maternal Health Risk", "maternal_health.arff",
     "RiskLevel"),

    ("German Credit", "german_credit.arff",
     "good_or_bad_customer"),

    ("QSAR Biodegradation", "QSAR_biodeg.arff",
     "Biodegradable"),
]

RESULTS_DIR = "results"
KNN_FILE = os.path.join(RESULTS_DIR, "knn_neighbors.csv")

# Input model PFN or ICL

def create_model(model_name):
    """Create the chosen model"""

    name = model_name.upper()

    if name == "PFN":
        return TabPFNClassifier()

    elif name == "ICL":
        return TabICLClassifier(device="cpu")

    else:
        raise ValueError(
            f"Okänd modell: {model_name}. Välj 'PFN' eller 'ICL'."
        )


def get_tabicl_attention(model, X_test, X_train):
    """Capture TabICL attention from test rows to training rows."""

    captured = []
    original_sdpa = attention_module.sdpa_with_flattened_batch

    n_train = len(model.X_train_) if hasattr(model, "X_train_") else None

    def capture_sdpa(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0.0,
        ssmax_layer=None,
    ):
        # Run the original implementation unchanged.
        output = original_sdpa(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            ssmax_layer=ssmax_layer,
        )

        # Select ICL attention: all train+test queries attending
        # to training keys, with SSMax enabled.
        if (
            q.ndim == 4
            and k.ndim == 4
            and q.shape[-2] == len(X_train) + len(X_test)
            and k.shape[-2] == len(X_train)
            and ssmax_layer is not None
        ):
            with torch.no_grad():
                q_used = q.detach()

                # Apply TabICL's query scaling before reconstructing
                # the attention probabilities.
                q_used = ssmax_layer(q_used, k.shape[-2])

                scores = torch.matmul(
                    q_used.float(),
                    k.detach().float().transpose(-2, -1),
                ) / math.sqrt(q.shape[-1])

                if attn_mask is not None:
                    mask = attn_mask
                    if mask.dtype == torch.bool:
                        scores = scores.masked_fill(~mask, float("-inf"))
                    else:
                        scores = scores + mask.float()

                attention = torch.softmax(scores, dim=-1)

                # Keep only test queries; key positions correspond
                # to the training rows.
                attention = attention[:, :, -len(X_test):, :]

                # Average over estimator and head dimensions:
                # (4, 8, n_test, n_train) -> (n_test, n_train)
                attention = attention.mean(dim=(0, 1))

                captured.append({
                    "layer": len(captured),
                    "attention": attention.cpu().numpy(),
                })

        return output

    attention_module.sdpa_with_flattened_batch = capture_sdpa

    try:
        model.predict(X_test)
    finally:
        attention_module.sdpa_with_flattened_batch = original_sdpa

    return captured

# Get Weights/ Readout

def get_model_weights(model_name, model, X_test, X_train):
    name = model_name.upper()

    if name == "PFN":
        weights, train_indices = get_decoder_readout(model, X_test)

        weights = np.asarray(weights)
        train_indices = np.asarray(train_indices, dtype=int)

        if weights.shape == (len(X_train), len(X_test)):
            weights = weights.T

        if weights.shape != (len(X_test), len(train_indices)):
            raise ValueError(
                f"Unexpected TabPFN readout shape: {weights.shape}"
            )

        if np.any(train_indices < 0) or np.any(
            train_indices >= len(X_train)
        ):
            raise ValueError("TabPFN train indices out of range.")

        return weights, train_indices

    elif name == "ICL":
        captured = get_tabicl_attention(model, X_test, X_train)

        if not captured:
            raise RuntimeError(
               "No TabICL ICL attention was captured."
            )

        # Each captured matrix has shape (n_test, n_train).
        weights = np.mean(
          [item["attention"] for item in captured],
          axis=0,
        )

        train_positions = np.arange(len(X_train))

        if weights.shape != (len(X_test), len(X_train)):
            raise ValueError(
                f"Unexpected TabICL attention shape: {weights.shape}; "
                f"expected {(len(X_test), len(X_train))}"
            )

        return weights, train_positions

    else:
        raise ValueError(f"Unknown model: {model_name}")


# Calculate overlap for dataset

def analyze_dataset(
    model_name,
    dataset_name,
    filename,
    target
):
    print(f"\n{'=' * 60}")
    print(f"Dataset: {dataset_name}")
    print(f"Modell: {model_name}")
    print(f"{'=' * 60}")

    # Load same train-test-split as in main().
    X_train, X_test, y_train, y_test = loadfile(
        filename,
        target
    )

    # Get KNN results.
    knn_df = pd.read_csv(KNN_FILE)

    required_columns = {
        "dataset",
        "test_index",
        "neighbor_rank",
        "neighbor_index",
    }

    missing = required_columns - set(knn_df.columns)

    if missing:
        raise ValueError(
            f"{KNN_FILE} missing columns: {missing}"
        )

    knn_df["test_index"] = knn_df["test_index"].astype(int)
    knn_df["neighbor_index"] = (
        knn_df["neighbor_index"].astype(int)
    )
    knn_df["neighbor_rank"] = (
        knn_df["neighbor_rank"].astype(int)
    )

    dataset_knn = knn_df[
        knn_df["dataset"] == dataset_name
    ].copy()

    if dataset_knn.empty:
        raise ValueError(
            f"No KNN-results found for {dataset_name}. "
            "Control names of datasets in KNN-file."
        )

    # Check that KNN and this uses
    # the same testpoints.
    knn_test_indices = set(
        dataset_knn["test_index"].unique()
    )
    expected_test_indices = set(
        X_test.index.astype(int)
    )

    if knn_test_indices != expected_test_indices:
        raise ValueError(
            f"Test index doesn't match for {dataset_name}. "
            "Check that main.py and this file uses "
            "same train/test-split."
        )

    # Create and train model.
    model = create_model(model_name)
    model.fit(X_train, y_train)

    # Get weights for model.
    weights, train_positions = get_model_weights(
        model_name,
        model,
        X_test,
        X_train
    )

    # Map positions in X_train to original index.
    train_original_indices = (
        X_train.index.to_numpy()[train_positions]
    )

    print("No of trainingpoints in readout:", len(train_positions))

    # Make sure that KNN has at least K neigbours per testpoint.
    neighbor_counts = dataset_knn.groupby(
        "test_index"
    ).size()

    if (neighbor_counts < K).any():
        raise ValueError(
            f"Some testpoints has fewer than {K} KNN-neighbours."
        )

    results = []

    # Analyzy each testpoint.
    for i, test_index in enumerate(X_test.index.astype(int)):

        # Model's K highest ranked trainingpoints.
        top_positions = np.argsort(
            weights[i]
        )[::-1][:K]

        model_neighbors = set(
            train_original_indices[top_positions].astype(int)
        )

        # KNN:s K nearest trainingpoints.
        test_knn = dataset_knn[
            dataset_knn["test_index"] == test_index
        ].sort_values("neighbor_rank")

        knn_neighbors = set(
            test_knn["neighbor_index"].head(K).astype(int)
        )

        shared = model_neighbors.intersection(knn_neighbors)

        overlap = len(shared) / K

        results.append({
            "dataset": dataset_name,
            "model": model_name.upper(),
            "test_index": test_index,
            "k": K,
            "n_shared": len(shared),
            "overlap_at_k": overlap,
            "model_neighbors": sorted(model_neighbors),
            "knn_neighbors": sorted(knn_neighbors),
            "shared_neighbors": sorted(shared),
        })

    result_df = pd.DataFrame(results)

    # Summarize the result for datasetet.
    summary_df = pd.DataFrame([{
        "dataset": dataset_name,
        "model": model_name.upper(),
        "k": K,
        "n_test": len(result_df),
        "mean_overlap_at_k": result_df[
            "overlap_at_k"
        ].mean(),
        "median_overlap_at_k": result_df[
            "overlap_at_k"
        ].median(),
    }])

    print("\nResultat:")
    print(summary_df.to_string(index=False))

    return result_df, summary_df


# Run all datasets

def run_analysis(model_name=MODEL_NAME):

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(KNN_FILE):
        raise FileNotFoundError(
            f"Can't find {KNN_FILE}. "
            "First run main.py to save KNN-results."
        )

    all_results = []
    all_summaries = []

    for dataset_name, filename, target in DATASETS:

        result_df, summary_df = analyze_dataset(
            model_name=model_name,
            dataset_name=dataset_name,
            filename=filename,
            target=target
        )

        all_results.append(result_df)
        all_summaries.append(summary_df)

    final_results = pd.concat(
        all_results,
        ignore_index=True
    )

    final_summary = pd.concat(
        all_summaries,
        ignore_index=True
    )

    model_tag = model_name.upper()

    results_path = os.path.join(
        RESULTS_DIR,
        f"{model_tag.lower()}_knn_overlap_all_datasets.csv"
    )

    summary_path = os.path.join(
        RESULTS_DIR,
        f"{model_tag.lower()}_knn_overlap_summary.csv"
    )

    final_results.to_csv(results_path, index=False)
    final_summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 60)
    print("ANALYSIS FINISHED")
    print("=" * 60)

    print(f"Results per testpoint: {results_path}")
    print(f"Summary:         {summary_path}")

    print("\nSummary:")
    print(final_summary.to_string(index=False))


if __name__ == "__main__":
    run_analysis()

