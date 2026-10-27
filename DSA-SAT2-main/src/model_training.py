"""
Model Training Module for CH15 Road Safety Insight.
Trains a Scikit-Learn Pipeline with ColumnTransformer and RandomForestClassifier,
evaluates on untouched stratified test data, computes metrics, and persists artifacts.
"""

import os
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from src.utils import get_project_root, save_json
from src.feature_engineering import prepare_features, NUMERICAL_FEATURES, CATEGORICAL_FEATURES


def train_and_evaluate_model(cleaned_csv_path=None, model_output_path=None, metrics_output_path=None):
    """
    Executes the machine learning training and evaluation pipeline:
    1. Loads cleaned data and separates features from target.
    2. Performs stratified train-test split (80/20, random_state=42).
    3. Builds unified ColumnTransformer + RandomForest pipeline.
    4. Fits pipeline strictly on training data.
    5. Evaluates model on untouched test data.
    6. Computes comprehensive classification metrics and confusion matrix.
    7. Computes feature importances with model-association disclosures.
    8. Persists joblib model artifact and JSON metrics report.
    9. Generates confusion matrix and feature importance plots.
    """
    project_root = get_project_root()
    if cleaned_csv_path is None:
        cleaned_csv_path = os.path.join(project_root, "data", "processed", "cleaned_accidents.csv")
    if model_output_path is None:
        model_output_path = os.path.join(project_root, "models", "road_safety_model.joblib")
    if metrics_output_path is None:
        metrics_output_path = os.path.join(project_root, "outputs", "model_metrics.json")

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
    charts_dir = os.path.join(project_root, "outputs", "charts")
    os.makedirs(charts_dir, exist_ok=True)

    # 1. Load data and features
    df = pd.read_csv(cleaned_csv_path)
    X, y, meta = prepare_features(df)

    classes = ["Minor", "Serious", "Fatal"]

    # 2. Stratified train-test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(f"[INFO] Train samples: {len(X_train)} | Test samples: {len(X_test)}")
    print(f"[INFO] Training class balance:\n{y_train.value_counts(normalize=True).round(4).to_dict()}")
    print(f"[INFO] Test class balance:\n{y_test.value_counts(normalize=True).round(4).to_dict()}")

    # 3. Build ColumnTransformer preprocessing pipeline
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )

    # 4. Define Random Forest Classifier Pipeline
    rf_classifier = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", rf_classifier)
    ])

    # 5. Fit pipeline on training data ONLY
    print("[INFO] Fitting preprocessing and Random Forest classifier on training partition...")
    pipeline.fit(X_train, y_train)

    # 6. Predict on untouched test data
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)

    # 7. Calculate evaluation metrics
    acc = accuracy_score(y_test, y_pred)
    macro_prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    macro_rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    weighted_prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    weighted_rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # Class-wise metrics
    cls_report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    class_metrics = {}
    for cls in classes:
        if cls in cls_report_dict:
            class_metrics[cls] = {
                "precision": round(cls_report_dict[cls]["precision"], 4),
                "recall": round(cls_report_dict[cls]["recall"], 4),
                "f1_score": round(cls_report_dict[cls]["f1-score"], 4),
                "test_support": int(cls_report_dict[cls]["support"])
            }

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    cm_dict = {
        "labels": classes,
        "matrix": cm.tolist()
    }

    # ==========================================================
    # AUTOMATED CHECKS: Validate Confusion Matrix Alignment
    # ==========================================================
    cm_row_sums = cm.sum(axis=1)
    cm_col_sums = cm.sum(axis=0)
    for idx, cls in enumerate(classes):
        # 1. Row sums must equal true class supports
        true_sup = class_metrics[cls]["test_support"]
        assert cm_row_sums[idx] == true_sup, (
            f"Automated Check FAILED: Row sum for '{cls}' ({cm_row_sums[idx]}) != true support ({true_sup})"
        )
        # 2. Column sums must equal predicted class counts
        pred_cnt = int(np.sum(y_pred == cls))
        assert cm_col_sums[idx] == pred_cnt, (
            f"Automated Check FAILED: Col sum for '{cls}' ({cm_col_sums[idx]}) != predicted count ({pred_cnt})"
        )

    # 3. Total sum must equal test set size
    assert cm.sum() == len(y_test), f"Automated Check FAILED: CM total ({cm.sum()}) != test size ({len(y_test)})"

    # 4. Accuracy must strictly match diagonal trace divided by total sum
    trace_acc = np.trace(cm) / cm.sum()
    assert np.isclose(trace_acc, acc), f"Automated Check FAILED: Trace acc ({trace_acc}) != accuracy_score ({acc})"
    print("[INFO] Automated Checks PASSED: Confusion-matrix row sums equal true supports, column sums equal predicted counts, and trace matches accuracy.")

    # 8. Extract Feature Importance
    try:
        fitted_preprocessor = pipeline.named_steps["preprocessor"]
        cat_encoder = fitted_preprocessor.named_transformers_["cat"].named_steps["onehot"]
        cat_encoded_feature_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
        all_feature_names = NUMERICAL_FEATURES + cat_encoded_feature_names

        classifier_importances = pipeline.named_steps["classifier"].feature_importances_
        feature_importance_df = pd.DataFrame({
            "feature": all_feature_names,
            "importance": classifier_importances
        }).sort_values(by="importance", ascending=False)

        top_features = feature_importance_df.head(15).to_dict(orient="records")
    except Exception as e:
        print(f"[WARNING] Feature names extraction encountered exception: {e}")
        top_features = []
        feature_importance_df = pd.DataFrame()

    # 9. Structure and save metrics JSON
    metrics_payload = {
        "model_architecture": "RandomForestClassifier",
        "parameters": {
            "n_estimators": 150,
            "max_depth": 10,
            "min_samples_leaf": 2,
            "class_weight": "balanced",
            "random_state": 42
        },
        "sample_counts": {
            "total": len(df),
            "train": len(X_train),
            "test": len(X_test),
            "train_ratio": 0.80,
            "test_ratio": 0.20
        },
        "overall_metrics": {
            "accuracy": round(acc, 4),
            "macro_precision": round(macro_prec, 4),
            "macro_recall": round(macro_rec, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_precision": round(weighted_prec, 4),
            "weighted_recall": round(weighted_rec, 4),
            "weighted_f1": round(weighted_f1, 4)
        },
        "class_wise_metrics": class_metrics,
        "confusion_matrix": cm_dict,
        "feature_importance_top15": top_features,
        "performance_assessment": {
            "class_imbalance_analysis": (
                f"The training and test sets feature heavy class imbalance: Minor incidents represent ~68.6%, "
                f"Serious incidents ~27.5%, and Fatal incidents only ~3.9% of records. "
                f"On the held-out test set (N=1,196), overall accuracy is {acc*100:.2f}%, and the macro F1-score is {macro_f1:.4f}, "
                f"reflecting the difficulty of distinguishing rare fatal-severity crashes (recall: {class_metrics['Fatal']['recall']*100:.1f}%) "
                f"from synthetic tabular attributes. Balanced class weighting was applied to prevent complete mode collapse into the majority Minor class."
            ),
            "causality_disclaimer": (
                "Feature importance scores quantify statistical associations within the trained decision tree ensemble. "
                "They do NOT establish causal mechanisms or real-world physical accident causation."
            )
        }
    }

    save_json(metrics_payload, metrics_output_path)

    # 10. Persist complete pipeline
    joblib.dump(pipeline, model_output_path)
    print(f"[INFO] Fitted model pipeline saved to {model_output_path}")

    # 11. Plot and save Confusion Matrix chart
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes,
                cbar=False, ax=ax, annot_kws={"size": 13, "weight": "bold"})
    ax.set_title("Random Forest Severity Confusion Matrix (Test Set)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Severity", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Ground-Truth Severity", fontsize=11, fontweight="bold")
    plt.tight_layout()
    cm_chart_path = os.path.join(charts_dir, "13_confusion_matrix.png")
    plt.savefig(cm_chart_path, dpi=300)
    plt.close()

    # 12. Plot and save Feature Importance chart
    if not feature_importance_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_df = feature_importance_df.head(12).sort_values(by="importance", ascending=True)
        ax.barh(plot_df["feature"], plot_df["importance"], color="#2b8cbe", edgecolor="#333333", height=0.65)
        ax.set_title("Top 12 Features by Random Forest Gini Importance (Model Associations)", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Mean Decrease in Impurity (Gini Importance)", fontsize=11, fontweight="bold")
        for idx, val in enumerate(plot_df["importance"]):
            ax.text(val + 0.001, idx, f"{val:.4f}", va="center", fontsize=9, fontweight="bold")
        ax.set_xlim(0, plot_df["importance"].max() * 1.15)
        plt.tight_layout()
        fi_chart_path = os.path.join(charts_dir, "14_feature_importance.png")
        plt.savefig(fi_chart_path, dpi=300)
        plt.close()

    # Print summary evaluation
    print("\n" + "=" * 55)
    print("MODEL EVALUATION SUMMARY (UNTOUCHED TEST SET)")
    print("=" * 55)
    print(f"Accuracy:         {acc:.4f} ({acc*100:.2f}%)")
    print(f"Macro F1-Score:   {macro_f1:.4f}")
    print(f"Weighted F1-Score:{weighted_f1:.4f}")
    print("\nClass-wise Performance:")
    for cls in classes:
        m = class_metrics[cls]
        print(f"  [{cls:7s}] Prec: {m['precision']:.4f} | Rec: {m['recall']:.4f} | F1: {m['f1_score']:.4f} | Support: {m['test_support']}")
    print("=" * 55 + "\n")

    return pipeline, metrics_payload


if __name__ == "__main__":
    train_and_evaluate_model()
