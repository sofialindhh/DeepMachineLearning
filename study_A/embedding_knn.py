"""Layerwise embedding kNN experiment for TabPFN v2.6.

For each dataset, the script runs TabPFN v2.6 once, captures the input
embedding and all transformer block outputs with forward hooks, and evaluates
a kNN readout at each representation. It compares those readouts with kNN on
the preprocessed input features, TabPFN's own prediction, and the
majority-class baseline.

The experiment uses the four classification datasets in ``datasets/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

import my_arch_setup

# Route the official v2.6 checkpoint to the local architecture copy.
my_arch_setup.use_as_v26()

from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

DATA_DIR = Path("study_A/datasets")
EMB_DIR = DATA_DIR / "embedding"
RESULTS_DIR = Path("study_A/results")

DATASETS = {
    "anneal": "classes",
    "blood-transfusion-service-center": "DonatedBloodInMarch2007",
    "credit-g": "good_or_bad_customer",
    "diabetes": "TestedPositiveForDiabetes",
}

SEED = 0
TEST_SIZE = 0.3
N_ESTIMATORS = 1
THINKING_ROWS = 64
N_LAYERS = 24
K_VALUES = (1, 3, 5, 10)
MAX_ROWS = 500


def extract_one(
    name: str,
    target: str,
    max_rows: int | None = MAX_ROWS,
    seed: int = SEED,
) -> dict:
    """Run TabPFN once and save the captured activations for one dataset."""
    frame = pd.read_csv(DATA_DIR / f"{name}.csv")
    X = frame.drop(columns=[target])
    y = frame[target]

    print(f"{name}: {frame.shape}")

    if max_rows is not None and len(X) > max_rows:
        X, _, y, _ = train_test_split(
            X,
            y,
            train_size=max_rows,
            random_state=seed,
            stratify=y,
        )
        print(f"  sampled {len(X)} rows")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=y,
    )

    clf = TabPFNClassifier.create_default_for_version(
        ModelVersion.V2_6,
        device="cpu",
        random_state=seed,
        n_estimators=N_ESTIMATORS,
        inference_config={"CLASS_SHIFT_METHOD": None},
    )
    clf.fit(X_train, y_train)

    arch = clf.models_[0]
    arch.eval()

    captured: dict[str, torch.Tensor] = {}

    def capture_model_input(module, args):
        captured["X_full"] = args[0].detach().cpu()

    def capture_before_tf(module, args):
        captured["before_tf"] = args[0].detach().cpu()

    def capture_layer(module, inputs, output, index):
        captured[f"after_tf_layer{index}"] = output[0].detach().cpu()

    arch.register_forward_pre_hook(capture_model_input)
    arch.add_thinking_rows.register_forward_pre_hook(capture_before_tf)

    for index, block in enumerate(arch.blocks):
        block.register_forward_hook(
            lambda module, inputs, output, index=index: capture_layer(
                module, inputs, output, index
            )
        )

    # One predict call runs the full classifier pipeline, including all
    # preprocessing, while the hooks record the tensors that enter the model.
    y_pred = clf.predict(X_test)

    X_full = captured["X_full"]
    n_train = len(X_train)
    X_train_pre = X_full[:n_train, 0, :].numpy()
    X_test_pre = X_full[n_train:, 0, :].numpy()

    out_dir = EMB_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "x_train.npy", X_train_pre)
    np.save(out_dir / "x_test.npy", X_test_pre)
    np.save(out_dir / "y_train.npy", np.asarray(y_train))
    np.save(out_dir / "y_test.npy", np.asarray(y_test))
    np.save(out_dir / "y_pred.npy", np.asarray(y_pred))
    torch.save(captured, out_dir / f"{name}_embedding.pt")

    size_mb = (out_dir / f"{name}_embedding.pt").stat().st_size / 1e6

    return {
        "dataset": name,
        "n_train": n_train,
        "n_test": len(X_test),
        "before_tf": tuple(captured["before_tf"].shape),
        "layer0": tuple(captured["after_tf_layer0"].shape),
        "n_classes": int(pd.Series(y_train).nunique()),
        "pt_MB": round(size_mb, 1),
    }


def row_embedding(hidden: torch.Tensor, has_thinking: bool) -> np.ndarray:
    """Build one feature vector per row, excluding the target column."""
    hidden = hidden[0]
    if has_thinking:
        hidden = hidden[THINKING_ROWS:]

    features = hidden[:, :-1, :]
    return features.reshape(features.shape[0], -1).cpu().numpy()


def knn_eval(
    X_context: np.ndarray,
    y_context: np.ndarray,
    X_query: np.ndarray,
    y_query: np.ndarray,
    y_pred: np.ndarray,
    k: int,
) -> dict[str, float]:
    """Evaluate one kNN readout on one representation."""
    scaler = StandardScaler().fit(X_context)
    X_context = scaler.transform(X_context)
    X_query = scaler.transform(X_query)

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_context, y_context)
    pred = model.predict(X_query)

    return {
        "k": k,
        "acc": accuracy_score(y_query, pred),
        "balanced_acc": balanced_accuracy_score(y_query, pred),
        "agreement": float((pred == y_pred).mean()),
        "majority_acc": float(
            pd.Series(y_query).value_counts(normalize=True).iloc[0]
        ),
    }


def evaluate_one(name: str) -> list[dict[str, object]]:
    """Evaluate before_tf, input-space, and all block readouts."""
    emb = EMB_DIR / name

    y_train = np.load(emb / "y_train.npy", allow_pickle=True)
    y_test = np.load(emb / "y_test.npy", allow_pickle=True)
    y_pred = np.load(emb / "y_pred.npy", allow_pickle=True)
    n_train = len(y_train)
    n_test = len(y_test)

    captured = torch.load(
        emb / f"{name}_embedding.pt",
        map_location="cpu",
        weights_only=False,
    )

    native_acc = accuracy_score(y_test, y_pred)
    native_balanced_acc = balanced_accuracy_score(y_test, y_pred)

    rows: list[dict[str, object]] = []

    def add(method: str, context: np.ndarray, query: np.ndarray) -> None:
        for k in K_VALUES:
            metrics = knn_eval(
                context,
                y_train,
                query,
                y_test,
                y_pred,
                k,
            )
            rows.append(
                {
                    "dataset": name,
                    "method": method,
                    **metrics,
                    "n_train": n_train,
                    "n_test": n_test,
                    "native_acc": native_acc,
                    "native_balanced_acc": native_balanced_acc,
                }
            )

    before_tf = row_embedding(captured["before_tf"], has_thinking=False)
    add("before_tf", before_tf[:n_train], before_tf[n_train:])

    X_train = np.load(emb / "x_train.npy", allow_pickle=True)
    X_test = np.load(emb / "x_test.npy", allow_pickle=True)
    add("input", X_train, X_test)

    for layer in range(N_LAYERS):
        layer_embedding = row_embedding(
            captured[f"after_tf_layer{layer}"],
            has_thinking=True,
        )
        add(
            f"L{layer}",
            layer_embedding[:n_train],
            layer_embedding[n_train:],
        )

    return rows


def build_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Summarise each dataset at k=5."""
    summary_rows = []

    for name in DATASETS:
        sub = results[(results["dataset"] == name) & (results["k"] == 5)]
        if sub.empty:
            continue

        input_row = sub[sub["method"] == "input"].iloc[0]
        before_row = sub[sub["method"] == "before_tf"].iloc[0]
        layers = sub[sub["method"].str.startswith("L")]

        best_acc = layers.loc[layers["acc"].idxmax()]
        best_agree = layers.loc[layers["agreement"].idxmax()]

        summary_rows.append(
            {
                "dataset": name,
                "n_train": input_row["n_train"],
                "n_test": input_row["n_test"],
                "native_acc": input_row["native_acc"],
                "native_balanced_acc": input_row["native_balanced_acc"],
                "majority_acc": input_row["majority_acc"],
                "input_acc": input_row["acc"],
                "input_balanced_acc": input_row["balanced_acc"],
                "input_agreement": input_row["agreement"],
                "before_acc": before_row["acc"],
                "before_balanced_acc": before_row["balanced_acc"],
                "before_agreement": before_row["agreement"],
                "best_acc_method": best_acc["method"],
                "best_acc": best_acc["acc"],
                "best_acc_agreement": best_acc["agreement"],
                "best_agree_method": best_agree["method"],
                "best_agree_acc": best_agree["acc"],
                "best_agree_agreement": best_agree["agreement"],
            }
        )

    return pd.DataFrame(summary_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=MAX_ROWS,
        help="cap the dataset size before splitting; 0 means no cap",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="recompute embeddings even if they already exist",
    )
    parser.add_argument(
        "--output",
        default=str(RESULTS_DIR),
        help="directory for the result CSV files",
    )
    args = parser.parse_args()

    max_rows = args.max_rows or None
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, target in DATASETS.items():
        required = [
            EMB_DIR / name / "x_train.npy",
            EMB_DIR / name / "x_test.npy",
            EMB_DIR / name / "y_train.npy",
            EMB_DIR / name / "y_test.npy",
            EMB_DIR / name / "y_pred.npy",
            EMB_DIR / name / f"{name}_embedding.pt",
        ]
        if args.force_extract or not all(path.exists() for path in required):
            info = extract_one(
                name,
                target,
                max_rows=max_rows,
                seed=args.seed,
            )
            print(
                f"  before_tf={info['before_tf']} "
                f"layer0={info['layer0']} "
                f"pt={info['pt_MB']} MB"
            )

    rows = []
    for name in DATASETS:
        rows.extend(evaluate_one(name))

    results = pd.DataFrame(rows)
    results_path = output_dir / "embedding_knn_all.csv"
    results.to_csv(results_path, index=False)

    summary = build_summary(results)
    summary_path = output_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"\nSaved {len(results)} rows to {results_path}")
    print(f"Saved summary to {summary_path}\n")
    print(
        summary[
            [
                "dataset",
                "native_acc",
                "input_acc",
                "before_acc",
                "best_acc_method",
                "best_acc",
                "best_agree_method",
                "best_agree_agreement",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
