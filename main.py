import arff
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score
)
from tabpfn import TabPFNClassifier
from tabicl import TabICLClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Load the ARFF file
def loadfile(f, target):
    with open(f"data/{f}", "r") as file:
        dataset = arff.load(file)

    # Convert to a pandas DataFrame
    df = pd.DataFrame(
        dataset["data"],
        columns=[attribute[0] for attribute in dataset["attributes"]]
    )

    print("First five rows:")
    print(df.head())

    print("\nDataset shape:")
    print(df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nTarget distribution:")
    print(df[target].value_counts())

    print("\nMissing values:")
    print(df.isnull().sum())

    # Split into features and target

    X = df.drop(columns=[target])
    y = df[target]


    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=0,
        stratify=y
    )

    print("Training samples:", len(X_train))
    print("Test samples:", len(X_test))
    return X_train, X_test, y_train, y_test


# KNN

def knn(X_train, X_test, y_train, y_test):
    # Identify numerical and categorical columns
    numerical_columns = X_train.select_dtypes(
        include=["number"]
    ).columns

    categorical_columns = X_train.select_dtypes(
        exclude=["number"]
    ).columns

    # Preprocess numerical and categorical columns separately
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numerical", numerical_pipeline, numerical_columns),
        ("categorical", categorical_pipeline, categorical_columns)
    ])

    # KNN
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", KNeighborsClassifier(n_neighbors=5))
    ])

    model.fit(X_train, y_train)

    # Find the 5 nearest training points for each test point
    distances, neighbor_positions = model.named_steps["classifier"].kneighbors(
        model.named_steps["preprocessor"].transform(X_test),
        n_neighbors=5
    )

    # Convert training positions to original row indices
    neighbor_indices = X_train.index.to_numpy()[neighbor_positions]

    # Save the neighbors in a DataFrame
    neighbors_df = pd.DataFrame([
        {
            "test_index": X_test.index[i],
            "neighbor_rank": rank + 1,
            "neighbor_index": neighbor_indices[i, rank],
            "distance": distances[i, rank],
            "neighbor_label": y_train.loc[neighbor_indices[i, rank]]
        }
        for i in range(len(X_test))
        for rank in range(5)
    ])

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print("\nKNN results")
    print("Accuracy:", accuracy)
    print("Balanced accuracy:", balanced_accuracy)
    print("Macro F1:", macro_f1)

    return accuracy, balanced_accuracy, macro_f1, y_pred, neighbors_df


# TabPFN
def tabPFN(X_train, X_test, y_train, y_test):
    tabpfn = TabPFNClassifier()

    tabpfn.fit(X_train, y_train)

    y_pred = tabpfn.predict(X_test)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print("\nTabPFN results")
    print("Accuracy:", accuracy)
    print("Balanced accuracy:", balanced_accuracy)
    print("Macro F1:", macro_f1)
    return accuracy, balanced_accuracy, macro_f1, y_pred


# TabICL
def tabICL(X_train, X_test, y_train, y_test):
    tabicl = TabICLClassifier(device="cpu")

    tabicl.fit(X_train, y_train)

    y_pred = tabicl.predict(X_test)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print("\nTabICL results")
    print("Accuracy:", accuracy)
    print("Balanced accuracy:", balanced_accuracy)
    print("Macro F1:", macro_f1)
    return accuracy, balanced_accuracy, macro_f1, y_pred


if __name__ == "__main__":
    all_predictions = []

    # Blood Transfusion
    X_train, X_test, y_train, y_test = loadfile("blood_transfusion.arff", "DonatedBloodInMarch2007")

    knn_accuracy_bt, knn_balanced_accuracy_bt, knn_macro_f1_bt, knn_pred_bt, neighbours_bt = knn(X_train, X_test, y_train, y_test)
    tabpfn_accuracy_bt, tabpfn_balanced_accuracy_bt, tabpfn_macro_f1_bt, tabpfn_pred_bt = tabPFN(X_train, X_test, y_train, y_test)
    tabicl_accuracy_bt, tabicl_balanced_accuracy_bt, tabicl_macro_f1_bt, tabicl_pred_bt = tabICL(X_train, X_test, y_train, y_test)

    predictions_bt = pd.DataFrame({
        "dataset": "Blood Transfusion",
        "test_index": X_test.index,
        "y_true": y_test.to_numpy(),
        "knn_pred": knn_pred_bt,
        "tabpfn_pred": tabpfn_pred_bt,
        "tabicl_pred": tabicl_pred_bt,
    })

    all_predictions.append(predictions_bt)

    # Diabetes
    X_train, X_test, y_train, y_test = loadfile("diabetes.arff", "TestedPositiveForDiabetes")

    knn_accuracy_d, knn_balanced_accuracy_d, knn_macro_f1_d, knn_pred_d, neighbours_d = knn(X_train, X_test, y_train, y_test)
    tabpfn_accuracy_d, tabpfn_balanced_accuracy_d, tabpfn_macro_f1_d, tabpfn_pred_d = tabPFN(X_train, X_test, y_train, y_test)
    tabicl_accuracy_d, tabicl_balanced_accuracy_d, tabicl_macro_f1_d, tabicl_pred_d = tabICL(X_train, X_test, y_train, y_test)

    predictions_d = pd.DataFrame({
        "dataset": "Diabetes",
        "test_index": X_test.index,
        "y_true": y_test.to_numpy(),
        "knn_pred": knn_pred_d,
        "tabpfn_pred": tabpfn_pred_d,
        "tabicl_pred": tabicl_pred_d,
    })

    all_predictions.append(predictions_d)

    # Maternal Health Risk
    X_train, X_test, y_train, y_test = loadfile("maternal_health.arff", "RiskLevel")

    knn_accuracy_mh, knn_balanced_accuracy_mh, knn_macro_f1_mh, knn_pred_mh, neighbours_mh = knn(X_train, X_test, y_train, y_test)
    tabpfn_accuracy_mh, tabpfn_balanced_accuracy_mh, tabpfn_macro_f1_mh, tabpfn_pred_mh = tabPFN(X_train, X_test, y_train, y_test)
    tabicl_accuracy_mh, tabicl_balanced_accuracy_mh, tabicl_macro_f1_mh, tabicl_pred_mh = tabICL(X_train, X_test, y_train, y_test)

    predictions_mh = pd.DataFrame({
        "dataset": "Maternal Health Risk",
        "test_index": X_test.index,
        "y_true": y_test.to_numpy(),
        "knn_pred": knn_pred_mh,
        "tabpfn_pred": tabpfn_pred_mh,
        "tabicl_pred": tabicl_pred_mh,
    })

    all_predictions.append(predictions_mh)

    # German Credit
    X_train, X_test, y_train, y_test = loadfile("german_credit.arff", "good_or_bad_customer")

    knn_accuracy_gc, knn_balanced_accuracy_gc, knn_macro_f1_gc, knn_pred_gc, neighbours_gc = knn(X_train, X_test, y_train, y_test)
    tabpfn_accuracy_gc, tabpfn_balanced_accuracy_gc, tabpfn_macro_f1_gc, tabpfn_pred_gc = tabPFN(X_train, X_test, y_train, y_test)
    tabicl_accuracy_gc, tabicl_balanced_accuracy_gc, tabicl_macro_f1_gc, tabicl_pred_gc = tabICL(X_train, X_test, y_train, y_test)

    predictions_gc = pd.DataFrame({
        "dataset": "German Credit",
        "test_index": X_test.index,
        "y_true": y_test.to_numpy(),
        "knn_pred": knn_pred_gc,
        "tabpfn_pred": tabpfn_pred_gc,
        "tabicl_pred": tabicl_pred_gc,
    })

    all_predictions.append(predictions_gc)

    #QSAR Biodegradation
    X_train, X_test, y_train, y_test = loadfile("QSAR_biodeg.arff", "Biodegradable")

    knn_accuracy_qb, knn_balanced_accuracy_qb, knn_macro_f1_qb, knn_pred_qb, neighbours_qb = knn(X_train, X_test, y_train, y_test)
    tabpfn_accuracy_qb, tabpfn_balanced_accuracy_qb, tabpfn_macro_f1_qb, tabpfn_pred_qb  = tabPFN(X_train, X_test, y_train, y_test)
    tabicl_accuracy_qb, tabicl_balanced_accuracy_qb, tabicl_macro_f1_qb, tabicl_pred_qb = tabICL(X_train, X_test, y_train, y_test)

    predictions_qb = pd.DataFrame({
        "dataset": "QSAR Biodegradation",
        "test_index": X_test.index,
        "y_true": y_test.to_numpy(),
        "knn_pred": knn_pred_qb,
        "tabpfn_pred": tabpfn_pred_qb,
        "tabicl_pred": tabicl_pred_qb,
    })

    all_predictions.append(predictions_qb)

    # Save results

    predictions = pd.concat(all_predictions, ignore_index=True)

    predictions.to_csv(
        "results/predictions.csv",
        index=False
    )

    print("Saved to results/predictions.csv")

    os.makedirs("results", exist_ok=True)

    results = pd.DataFrame([
        {
            "dataset": "Blood Transfusion",
            "model": "KNN",
            "accuracy": knn_accuracy_bt,
            "balanced_accuracy": knn_balanced_accuracy_bt,
            "macro_f1": knn_macro_f1_bt,
        },
        {
            "dataset": "Blood Transfusion",
            "model": "TabPFN",
            "accuracy": tabpfn_accuracy_bt,
            "balanced_accuracy": tabpfn_balanced_accuracy_bt,
            "macro_f1": tabpfn_macro_f1_bt,
        },
        {
            "dataset": "Blood Transfusion",
            "model": "TabICL",
            "accuracy": tabicl_accuracy_bt,
            "balanced_accuracy": tabicl_balanced_accuracy_bt,
            "macro_f1": tabicl_macro_f1_bt,
        },
        {
            "dataset": "Diabetes",
            "model": "KNN",
            "accuracy": knn_accuracy_d,
            "balanced_accuracy": knn_balanced_accuracy_d,
            "macro_f1": knn_macro_f1_d,
        },
        {
            "dataset": "Diabetes",
            "model": "TabPFN",
            "accuracy": tabpfn_accuracy_d,
            "balanced_accuracy": tabpfn_balanced_accuracy_d,
            "macro_f1": tabpfn_macro_f1_d,
        },
        {
            "dataset": "Diabetes",
            "model": "TabICL",
            "accuracy": tabicl_accuracy_d,
            "balanced_accuracy": tabicl_balanced_accuracy_d,
            "macro_f1": tabicl_macro_f1_d,
        },
        {
            "dataset": "Maternal Health Risk",
            "model": "KNN",
            "accuracy": knn_accuracy_mh,
            "balanced_accuracy": knn_balanced_accuracy_mh,
            "macro_f1": knn_macro_f1_mh,
        },
        {
            "dataset": "Maternal Health Risk",
            "model": "TabPFN",
            "accuracy": tabpfn_accuracy_mh,
            "balanced_accuracy": tabpfn_balanced_accuracy_mh,
            "macro_f1": tabpfn_macro_f1_mh,
        },
        {
            "dataset": "Maternal Health Risk",
            "model": "TabICL",
            "accuracy": tabicl_accuracy_mh,
            "balanced_accuracy": tabicl_balanced_accuracy_mh,
            "macro_f1": tabicl_macro_f1_mh,
        },
        {
            "dataset": "German Credit",
            "model": "KNN",
            "accuracy": knn_accuracy_gc,
            "balanced_accuracy": knn_balanced_accuracy_gc,
            "macro_f1": knn_macro_f1_gc,
        },
        {
            "dataset": "German Credit",
            "model": "TabPFN",
            "accuracy": tabpfn_accuracy_gc,
            "balanced_accuracy": tabpfn_balanced_accuracy_gc,
            "macro_f1": tabpfn_macro_f1_gc,
        },
        {
            "dataset": "German Credit",
            "model": "TabICL",
            "accuracy": tabicl_accuracy_gc,
            "balanced_accuracy": tabicl_balanced_accuracy_gc,
            "macro_f1": tabicl_macro_f1_gc,
        },
        {
            "dataset": "QSAR Biodegradation",
            "model": "KNN",
            "accuracy": knn_accuracy_qb,
            "balanced_accuracy": knn_balanced_accuracy_qb,
            "macro_f1": knn_macro_f1_qb,
        },
        {
            "dataset": "QSAR Biodegradation",
            "model": "TabPFN",
            "accuracy": tabpfn_accuracy_qb,
            "balanced_accuracy": tabpfn_balanced_accuracy_qb,
            "macro_f1": tabpfn_macro_f1_qb,
        },
        {
            "dataset": "QSAR Biodegradation",
            "model": "TabICL",
            "accuracy": tabicl_accuracy_qb,
            "balanced_accuracy": tabicl_balanced_accuracy_qb,
            "macro_f1": tabicl_macro_f1_qb,
        },
    ])

    results.to_csv("results/metrics.csv", index=False)

    print(results)
    print("Saved to results/metrics.csv")


    all_neighbors = pd.concat(
        [
            neighbours_bt.assign(dataset="Blood Transfusion"),
            neighbours_d.assign(dataset="Diabetes"),
            neighbours_mh.assign(dataset="Maternal Health Risk"),
            neighbours_gc.assign(dataset="German Credit"),
            neighbours_qb.assign(dataset="QSAR Biodegradation")
        ],
        ignore_index=True
    )

    all_neighbors.to_csv("results/knn_neighbors.csv", index=False)
    print(all_neighbors.head(10))
    print("Number of saved neighbors:", len(all_neighbors))
    print(all_neighbors.groupby("dataset").size())  