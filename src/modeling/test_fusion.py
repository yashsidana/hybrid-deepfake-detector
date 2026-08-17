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
    balanced_accuracy_score,
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


def _fuse_with_bundle(semantic_emb, temporal_emb, forensic_emb, bundle):
    matcher = bundle["distribution_matcher"]
    semantic_weight = bundle["semantic_weight"]
    temporal_weight = bundle["temporal_weight"]
    # forensic_weight/uses_forensic: added when the forensic branch landed.
    # .get() with defaults so a fusion_model.pkl trained before that change
    # still loads and evaluates correctly (semantic+temporal only).
    forensic_weight = bundle.get("forensic_weight", 1.0)
    uses_forensic = bundle.get("uses_forensic", False)

    forensic_arg = forensic_emb if uses_forensic else None

    base = build_fused_vector(
        semantic_emb, temporal_emb,
        semantic_weight=semantic_weight, temporal_weight=temporal_weight,
        forensic_embeddings=forensic_arg, forensic_weight=forensic_weight,
    )
    distribution_scores = matcher.score(base)
    return build_fused_vector(
        semantic_emb, temporal_emb,
        semantic_weight=semantic_weight, temporal_weight=temporal_weight,
        forensic_embeddings=forensic_arg, forensic_weight=forensic_weight,
        distribution_scores=distribution_scores,
    )


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings-root", default=EMBEDDINGS_ROOT)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--report-path", default=REPORT_PATH)
    args = parser.parse_args()

    test_path = os.path.join(args.embeddings_root, "test.npz")
    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"{test_path} not found. Run `python -m src.modeling.extract_embeddings` first."
        )
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(
            f"No fusion model at {args.model_path}. Run `python -m src.modeling.train_fusion` first."
        )

    data = np.load(test_path, allow_pickle=True)
    semantic_emb, temporal_emb, labels = (
        data["semantic_embeddings"], data["temporal_embeddings"], data["labels"]
    )
    forensic_emb = data["forensic_embeddings"] if "forensic_embeddings" in data.files else None

    bundle = joblib.load(args.model_path)
    svm = bundle["svm"]
    scaler = bundle["scaler"]

    fused = _fuse_with_bundle(semantic_emb, temporal_emb, forensic_emb, bundle)
    fused_scaled = scaler.transform(fused)

    preds = svm.predict(fused_scaled)
    probs = svm.predict_proba(fused_scaled)[:, 1]  # P(fake)

    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    # Macro-F1 and balanced accuracy, in addition to the positive-class
    # (fake) precision/recall/F1 above -- same rationale as
    # train_semantic.py/train_temporal.py's evaluate_metrics(): on an
    # imbalanced test set, positive-class F1 alone can look fine while the
    # model has quietly stopped recognizing the minority class.
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    balanced_acc = balanced_accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
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
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Precision (fake):  {precision:.4f}")
    print(f"Recall (fake):     {recall:.4f}")
    print(f"F1 (fake):         {f1:.4f}")
    print(f"Macro-F1:          {macro_f1:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"ROC-AUC:           {roc_auc:.4f}" if roc_auc is not None else "ROC-AUC:           N/A")
    print(f"\nTest set composition: {int((labels == 0).sum())} real, {int((labels == 1).sum())} fake")
    print(f"False Positives (real predicted fake): {int(fp)}")
    print(f"False Negatives (fake predicted real): {int(fn)}")
    print(f"\nConfusion Matrix (rows=true, cols=pred) {CLASS_NAMES}:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    os.makedirs(os.path.dirname(args.report_path) or ".", exist_ok=True)
    with open(args.report_path, "w") as f:
        json.dump({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "macro_f1": macro_f1,
            "balanced_accuracy": balanced_acc,
            "roc_auc": roc_auc,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "num_real_test": int((labels == 0).sum()),
            "num_fake_test": int((labels == 1).sum()),
            "false_positives": int(fp),
            "false_negatives": int(fn),
        }, f, indent=2)

    print(f"\nSaved evaluation report to {args.report_path}")


if __name__ == "__main__":
    main()
