import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
LABELS = ["alert", "drowsy", "distracted"]
FEATURES = [
    "ear_mean",
    "ear_std",
    "ear_norm",
    "mar_mean",
    "perclos",
    "blink_rate",
    "max_closed_sec",
    "pitch_mean",
    "pitch_std",
    "yaw_mean",
    "yaw_std",
]
MODEL_NAMES = ["threshold", "random_forest", "svm_rbf"]
PERCLOS_THRESHOLD = 0.2
YAW_THRESHOLD = 20.0


def load_data():
    paths = sorted(DATA_DIR.glob("*.csv"))
    if not paths:
        raise SystemExit(f"No CSV files found in {DATA_DIR}")
    df = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    df = df[df["label"].isin(LABELS)].dropna(subset=FEATURES + ["subject"]).copy()
    df[FEATURES] = df[FEATURES].astype(float)
    return df


def predict_threshold(frame):
    drowsy = frame["perclos"].to_numpy() > PERCLOS_THRESHOLD
    distracted = np.abs(frame["yaw_mean"].to_numpy()) > YAW_THRESHOLD
    return np.where(drowsy, "drowsy", np.where(distracted, "distracted", "alert"))


def build(name):
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)
    if name == "svm_rbf":
        return make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced"))
    return None


def cross_validate(name, df, n_splits):
    predictions = np.empty(len(df), dtype=object)
    splitter = GroupKFold(n_splits=n_splits)
    for train_idx, test_idx in splitter.split(df, df["label"], df["subject"]):
        train, test = df.iloc[train_idx], df.iloc[test_idx]
        if name == "threshold":
            predictions[test_idx] = predict_threshold(test)
        else:
            model = build(name)
            model.fit(train[FEATURES], train["label"])
            predictions[test_idx] = model.predict(test[FEATURES])
    return predictions.astype(str)


def summarize(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "drowsy_precision": precision[1],
        "drowsy_recall": recall[1],
        "drowsy_f1": f1[1],
    }


def save_confusion(name, y_true, y_pred):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=LABELS, ax=ax, colorbar=False)
    ax.set_title(f"{name} (subject-grouped CV)")
    fig.tight_layout()
    fig.savefig(MODEL_DIR / f"cm_{name}.png", dpi=120)
    plt.close(fig)


def write_report(df, results, n_subjects, n_splits):
    lines = ["# Model comparison", ""]
    if n_subjects < 3:
        lines += [f"> WARNING: only {n_subjects} subject(s). Results are not reliable for generalization.", ""]
    lines += [
        f"- Subjects: {', '.join(sorted(df['subject'].unique()))}",
        f"- Rows: {len(df)}",
        f"- Evaluation: GroupKFold by subject ({n_splits} folds), pooled out-of-fold predictions",
        f"- Thresholds (baseline): PERCLOS > {PERCLOS_THRESHOLD}, |yaw_mean| > {YAW_THRESHOLD} deg",
        "",
        "## Summary",
        "",
        "| Model | Accuracy | Macro-F1 | Drowsy precision | Drowsy recall | Drowsy F1 |",
        "|---|---|---|---|---|---|",
    ]
    for name, (summary, _) in results.items():
        lines.append(
            f"| {name} | {summary['accuracy']:.3f} | {summary['macro_f1']:.3f} | "
            f"{summary['drowsy_precision']:.3f} | {summary['drowsy_recall']:.3f} | {summary['drowsy_f1']:.3f} |"
        )
    for name, (_, y_pred) in results.items():
        report = classification_report(df["label"], y_pred, labels=LABELS, zero_division=0)
        matrix = confusion_matrix(df["label"], y_pred, labels=LABELS)
        lines += ["", f"## {name}", "", "```", report, "Confusion matrix (rows=true, cols=pred):", str(matrix), "```"]
    (MODEL_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    df = load_data()
    subjects = df["subject"].unique()
    n_subjects = len(subjects)
    if n_subjects < 2:
        raise SystemExit("Need at least 2 subjects for subject-grouped evaluation.")
    if n_subjects < 3:
        print(f"WARNING: only {n_subjects} subjects. Results cannot be trusted for generalization; collect at least 3.")
    n_splits = min(5, n_subjects)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Subjects: {', '.join(subjects)} | rows: {len(df)} | folds: {n_splits}")
    print(df["label"].value_counts().to_string())

    results = {}
    for name in MODEL_NAMES:
        y_pred = cross_validate(name, df, n_splits)
        results[name] = (summarize(df["label"], y_pred), y_pred)
        save_confusion(name, df["label"], y_pred)

    print("\nComparison (subject-grouped CV):")
    for name, (summary, _) in results.items():
        print(
            f"{name:14s} acc={summary['accuracy']:.3f} macroF1={summary['macro_f1']:.3f} "
            f"drowsy_recall={summary['drowsy_recall']:.3f}"
        )
    write_report(df, results, n_subjects, n_splits)

    rf = build("random_forest")
    rf.fit(df[FEATURES], df["label"])
    importances = sorted(zip(FEATURES, rf.feature_importances_), key=lambda item: item[1], reverse=True)
    print("\nRandom Forest feature importances:")
    for feature, value in importances:
        print(f"  {feature:16s} {value:.4f}")

    best = max(results, key=lambda name: (results[name][0]["macro_f1"], results[name][0]["drowsy_recall"]))
    estimator = None if best == "threshold" else build(best)
    if estimator is not None:
        estimator.fit(df[FEATURES], df["label"])
    payload = {
        "model_name": best,
        "estimator": estimator,
        "features": FEATURES,
        "classes": LABELS,
        "thresholds": {"perclos": PERCLOS_THRESHOLD, "yaw_abs": YAW_THRESHOLD},
    }
    joblib.dump(payload, MODEL_DIR / "drowsy_model.pkl")
    print(f"\nBest model by macro-F1: {best}. Saved to {MODEL_DIR / 'drowsy_model.pkl'}")
    print(f"Report: {MODEL_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
