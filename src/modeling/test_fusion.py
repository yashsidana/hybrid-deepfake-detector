"""
Fusion (hybrid) branch — final, one-time test-set evaluation.

Loads the SVM + StandardScaler + DistributionMatcher bundle saved by
train_fusion.py and evaluates it on data/processed/fusion/test.npz, which
the test set never touched during training or model selection (same
"test set is one-time-use" discipline test_semantic.py / test_temporal.py
follow).
"""

import json
import os

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.features.fusion import build_fused_vector

EMBEDDINGS_ROOT = "data/processed/fusion"
MODEL_PATH = "models/fusion_classifier/fusion_model.pkl"
REPORT_PATH = "saved_models/test_fusion_evaluation_report.json"
CLASS_NAMES = ["real", "fake"]


def _fuse_with_bundle(semantic_emb, temporal_emb, bundle):
    matcher = bundle["distribution_matcher"]
    semantic_weight = bundle["semantic_weight"]
    temporal_weight = bundle["temporal_weight"]

    base = build_fused_vector(semantic_emb, temporal_emb, semantic_weight, temporal_weight)
    distribution_scores = matcher.score(base)
    return build_fused_vector(
        semantic_emb, temporal_emb, semantic_weight, temporal_weight,
        distribution_scores=distribution_scores,
    )


def main():
    test_path = os.path.join(EMBEDDINGS_ROOT, "test.npz")
    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"{test_path} not found. Run `python -m src.modeling.extract_embeddings` first."
        )
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No fusion model at {MODEL_PATH}. Run `python -m src.modeling.train_fusion` first."
        )

    data = np.load(test_path, allow_pickle=True)
    semantic_emb, temporal_emb, labels = (
        data["semantic_embeddings"], data["temporal_embeddings"], data["labels"]
    )

    bundle = joblib.load(MODEL_PATH)
    svm = bundle["svm"]
    scaler = bundle["scaler"]

    fused = _fuse_with_bundle(semantic_emb, temporal_emb, bundle)
    fused_scaled = scaler.transform(fused)

    preds = svm.predict(fused_scaled)
    probs = svm.predict_proba(fused_scaled)[:, 1]  # P(fake)

    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    report = classification_report(
        labels, preds, labels=[0, 1], target_names=CLASS_NAMES, zero_division=0
    )

    if len(np.unique(labels)) == 2:
        roc_auc = roc_auc_score(labels, probs)
    else:
        roc_auc = None
        print(
            "\n[Warning] ROC-AUC skipped: test set contains only one class. "
            "Check data/splits/test.csv."
        )

    print("\n" + "=" * 55)
    print("FUSION (HYBRID) BRANCH — FINAL TEST SET EVALUATION")
    print("=" * 55)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}" if roc_auc is not None else "ROC-AUC:   N/A")
    print(f"\nConfusion Matrix (rows=true, cols=pred) {CLASS_NAMES}:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        }, f, indent=2)

    print(f"\nSaved evaluation report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
