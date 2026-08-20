"""
Ablation study: quantifies how much each branch contributes to the hybrid
classifier, per the project's Section 15 requirement.

    Model A: semantic only                          -> SVM
    Model B: semantic + temporal                     -> SVM
    Model C: semantic + temporal + forensic + dist.  -> SVM   (the full pipeline)

Rather than duplicating train_fusion.py/test_fusion.py into three near-copies,
this reuses the exact same primitives (DistributionMatcher, build_fused_vector,
StandardScaler, GridSearchCV+SVC) and simply varies which branches are
included per model. "Excluding" a branch is done by omitting it from
build_fused_vector() entirely (not zero-padding it in) so an ablated
model's fused vector has the branch's dimensions genuinely absent, not
present-but-inert -- this matches what "semantic only" should actually mean
for e.g. distribution-matching, which would otherwise be fit on a
partially-degenerate space.

Each model gets ITS OWN DistributionMatcher and StandardScaler, fit fresh
on that model's own train split -- Model A's distribution matcher lives in
semantic-only space, Model C's in the full fused space, etc. This mirrors
train_fusion.py's actual fitting discipline (train-real only, never val/test)
for every variant, not just the full model.

Requires data/processed/fusion/{train,val,test}.npz already produced by
extract_embeddings.py (with forensic_embeddings present in the .npz for
Model C to mean anything -- if forensic wasn't extracted, Model C reduces
to Model B and this script says so explicitly rather than silently
pretending forensic was used).

Usage:
    python -m src.modeling.run_ablation
    python -m src.modeling.run_ablation --embeddings-root data/processed/fusion_cross_dfdc_to_celebdf
"""

import argparse
import json
import os

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.features.fusion import DistributionMatcher, build_fused_vector

EMBEDDINGS_ROOT = "data/processed/fusion"
REPORT_PATH = "saved_models/ablation_report.json"

MODEL_DEFS = {
    "A: Semantic only": {"use_temporal": False, "use_forensic": False},
    "B: Semantic + Temporal": {"use_temporal": True, "use_forensic": False},
    "C: Semantic + Temporal + Forensic (full hybrid)": {"use_temporal": True, "use_forensic": True},
}


def _load_split(split_name, embeddings_root):
    path = os.path.join(embeddings_root, f"{split_name}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.modeling.extract_embeddings` first "
            f"(pointed at --output-root {embeddings_root} if this isn't the default)."
        )
    data = np.load(path, allow_pickle=True)
    forensic = data["forensic_embeddings"] if "forensic_embeddings" in data.files else None
    return data["semantic_embeddings"], data["temporal_embeddings"], forensic, data["labels"]


def _fuse(sem, temp, forensic, use_temporal, use_forensic, matcher=None):
    kwargs = dict(semantic_weight=1.0)
    temporal_arg = temp if use_temporal else np.zeros((sem.shape[0], 0), dtype=np.float32)
    forensic_arg = forensic if (use_forensic and forensic is not None) else None

    base = build_fused_vector(
        sem, temporal_arg, temporal_weight=1.0,
        forensic_embeddings=forensic_arg, forensic_weight=1.0,
    )
    if matcher is None:
        return base, None
    distribution_scores = matcher.score(base)
    fused = build_fused_vector(
        sem, temporal_arg, temporal_weight=1.0,
        forensic_embeddings=forensic_arg, forensic_weight=1.0,
        distribution_scores=distribution_scores,
    )
    return fused, distribution_scores


def _run_one_model(name, cfg, train, val, test, c_grid):
    train_sem, train_temp, train_forensic, train_labels = train
    val_sem, val_temp, val_forensic, val_labels = val
    test_sem, test_temp, test_forensic, test_labels = test

    use_temporal = cfg["use_temporal"]
    use_forensic = cfg["use_forensic"] and train_forensic is not None

    if cfg["use_forensic"] and train_forensic is None:
        print(
            f"[{name}] forensic_embeddings not present in the .npz files -- "
            f"this model reduces to semantic+temporal only. Run "
            f"precompute_forensic.py + re-run extract_embeddings.py to "
            f"include forensic features."
        )

    base_train, _ = _fuse(train_sem, train_temp, train_forensic, use_temporal, use_forensic)
    real_mask = train_labels == 0
    if real_mask.sum() < 2:
        raise RuntimeError(
            f"[{name}] Need at least 2 'real' samples in train to fit the "
            f"distribution matcher."
        )
    matcher = DistributionMatcher().fit(base_train[real_mask])

    train_fused, _ = _fuse(train_sem, train_temp, train_forensic, use_temporal, use_forensic, matcher)
    val_fused, _ = _fuse(val_sem, val_temp, val_forensic, use_temporal, use_forensic, matcher)
    test_fused, _ = _fuse(test_sem, test_temp, test_forensic, use_temporal, use_forensic, matcher)

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_fused)
    val_scaled = scaler.transform(val_fused)
    test_scaled = scaler.transform(test_fused)

    n_splits = max(2, min(3, int(np.bincount(train_labels).min())))
    search = GridSearchCV(
        SVC(probability=True, class_weight="balanced"),
        {"C": c_grid, "kernel": ["rbf"], "gamma": ["scale"]},
        scoring="f1_macro", cv=n_splits, n_jobs=-1,
    )
    search.fit(train_scaled, train_labels)
    best_svm = search.best_estimator_

    val_preds = best_svm.predict(val_scaled)
    val_macro_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)

    test_preds = best_svm.predict(test_scaled)
    test_probs = best_svm.predict_proba(test_scaled)[:, 1]

    accuracy = accuracy_score(test_labels, test_preds)
    macro_f1 = f1_score(test_labels, test_preds, average="macro", zero_division=0)
    balanced_acc = balanced_accuracy_score(test_labels, test_preds)
    roc_auc = roc_auc_score(test_labels, test_probs) if len(np.unique(test_labels)) == 2 else None

    return {
        "model": name,
        "fused_dim": int(train_fused.shape[1]),
        "best_C": search.best_params_["C"],
        "val_macro_f1": float(val_macro_f1),
        "test_accuracy": float(accuracy),
        "test_macro_f1": float(macro_f1),
        "test_balanced_accuracy": float(balanced_acc),
        "test_roc_auc": float(roc_auc) if roc_auc is not None else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings-root", default=EMBEDDINGS_ROOT)
    parser.add_argument("--report-path", default=REPORT_PATH)
    parser.add_argument("--c-grid", nargs="+", type=float, default=[0.1, 1, 10, 100])
    args = parser.parse_args()

    train = _load_split("train", args.embeddings_root)
    val = _load_split("val", args.embeddings_root)
    test = _load_split("test", args.embeddings_root)

    print(f"Train: {len(train[3])} | Val: {len(val[3])} | Test: {len(test[3])}\n")

    results = []
    for name, cfg in MODEL_DEFS.items():
        print(f"--- Running {name} ---")
        result = _run_one_model(name, cfg, train, val, test, args.c_grid)
        results.append(result)
        print(
            f"  fused_dim={result['fused_dim']} best_C={result['best_C']} "
            f"val_macro_f1={result['val_macro_f1']:.4f}\n"
            f"  TEST: acc={result['test_accuracy']:.4f} "
            f"macro_f1={result['test_macro_f1']:.4f} "
            f"balanced_acc={result['test_balanced_accuracy']:.4f} "
            f"roc_auc={result['test_roc_auc'] if result['test_roc_auc'] is not None else 'N/A'}\n"
        )

    print("=" * 90)
    print(f"{'Model':<50} {'Accuracy':>10} {'Macro-F1':>10} {'Bal.Acc':>10} {'ROC-AUC':>10}")
    print("-" * 90)
    for r in results:
        auc_str = f"{r['test_roc_auc']:.4f}" if r["test_roc_auc"] is not None else "N/A"
        print(
            f"{r['model']:<50} {r['test_accuracy']:>10.4f} {r['test_macro_f1']:>10.4f} "
            f"{r['test_balanced_accuracy']:>10.4f} {auc_str:>10}"
        )
    print("=" * 90)

    os.makedirs(os.path.dirname(args.report_path) or ".", exist_ok=True)
    with open(args.report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAblation report saved to {args.report_path}")


if __name__ == "__main__":
    main()
