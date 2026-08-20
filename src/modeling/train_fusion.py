"""
Phase 3 (Feature Fusion): trains the final hybrid classifier on top of the
already-trained semantic and temporal branches.

Pipeline (matches the proposal's Methodology sections 4-7):
  4. Data Preprocessing    -> StandardScaler on the fused vector
  5. Feature Fusion        -> build_fused_vector() concatenates semantic +
                               temporal embeddings (+ forensic, once available)
  6. Distribution Matching -> DistributionMatcher, fit on TRAIN "real"
                               embeddings only, appended as one extra feature
  7. ML Classification     -> SVM (config.yaml: models.fusion.classifier_type),
                               hyperparameters chosen by grid search using
                               the SAME macro-F1 selection criterion the
                               semantic/temporal branches use (see
                               train_semantic.py: raw accuracy on an
                               imbalanced dataset rewards collapsing to the
                               majority class).

Requires data/processed/fusion/{train,val}.npz — run
`python -m src.modeling.extract_embeddings` first.
"""

import json
import os

import joblib
import numpy as np
import yaml
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.features.fusion import DistributionMatcher, build_fused_vector

CONFIG_PATH = "config/config.yaml"
EMBEDDINGS_ROOT = "data/processed/fusion"
MODEL_DIR = "models/fusion_classifier"
MODEL_PATH = os.path.join(MODEL_DIR, "fusion_model.pkl")
REPORT_PATH = "saved_models/train_fusion_report.json"


def _load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _load_split(split_name, embeddings_root=EMBEDDINGS_ROOT):
    path = os.path.join(embeddings_root, f"{split_name}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.modeling.extract_embeddings` first."
        )
    data = np.load(path, allow_pickle=True)

    # forensic_embeddings is backward-compatible: .npz files written before
    # the forensic branch existed won't have this key. Fall back to None so
    # _fuse() below simply omits it, rather than requiring everyone to
    # re-run extract_embeddings.py immediately after pulling this change.
    forensic = data["forensic_embeddings"] if "forensic_embeddings" in data.files else None

    return data["semantic_embeddings"], data["temporal_embeddings"], forensic, data["labels"]


def _fuse(semantic_emb, temporal_emb, forensic_emb, matcher,
          semantic_weight, temporal_weight, forensic_weight):
    """
    Builds the fused vector INCLUDING the distribution-matching feature.
    Computed in two passes (base fusion -> distribution score -> re-fuse
    with that score appended) since the matcher scores the fused embedding
    space -- semantic + temporal + forensic together -- not any one branch
    alone.
    """
    base = build_fused_vector(
        semantic_emb, temporal_emb,
        semantic_weight=semantic_weight, temporal_weight=temporal_weight,
        forensic_embeddings=forensic_emb, forensic_weight=forensic_weight,
    )
    distribution_scores = matcher.score(base)
    return build_fused_vector(
        semantic_emb, temporal_emb,
        semantic_weight=semantic_weight, temporal_weight=temporal_weight,
        forensic_embeddings=forensic_emb, forensic_weight=forensic_weight,
        distribution_scores=distribution_scores,
    )


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings-root", default=EMBEDDINGS_ROOT,
                         help="Directory with train.npz/val.npz from extract_embeddings.py "
                              "(default: data/processed/fusion; use a distinct directory "
                              "per experiment, matching that run's --output-root).")
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--report-path", default=REPORT_PATH)
    args = parser.parse_args()

    config = _load_config()
    fusion_cfg = config.get("models", {}).get("fusion", {})
    semantic_weight = fusion_cfg.get("semantic_weight", 1.0)
    temporal_weight = fusion_cfg.get("temporal_weight", 1.0)
    forensic_weight = fusion_cfg.get("forensic_weight", 1.0)

    train_sem, train_temp, train_forensic, train_labels = _load_split("train", embeddings_root=args.embeddings_root)
    val_sem, val_temp, val_forensic, val_labels = _load_split("val", embeddings_root=args.embeddings_root)

    if train_forensic is None:
        print(
            "No forensic_embeddings found in data/processed/fusion/train.npz "
            "-- training on semantic + temporal only. Run "
            "`python -m src.preprocessing.precompute_forensic` and re-run "
            "extract_embeddings.py to include forensic features."
        )

    print(f"Train: {len(train_labels)} samples | Val: {len(val_labels)} samples")

    # Distribution matcher: fit on TRAIN "real" (label 0) embeddings only —
    # never val/test, and never the "fake" class, per the proposal's
    # description of learning the distribution of GENUINE media.
    base_train_fused = build_fused_vector(
        train_sem, train_temp,
        semantic_weight=semantic_weight, temporal_weight=temporal_weight,
        forensic_embeddings=train_forensic, forensic_weight=forensic_weight,
    )
    real_mask = train_labels == 0
    if real_mask.sum() < 2:
        raise RuntimeError(
            "Need at least 2 'real' samples in the training split to fit "
            "the distribution matcher — check data/splits/train.csv and "
            "confirm class balance."
        )
    matcher = DistributionMatcher()
    matcher.fit(base_train_fused[real_mask])

    train_fused = _fuse(train_sem, train_temp, train_forensic, matcher,
                         semantic_weight, temporal_weight, forensic_weight)
    val_fused = _fuse(val_sem, val_temp, val_forensic, matcher,
                       semantic_weight, temporal_weight, forensic_weight)

    scaler = StandardScaler()
    train_fused_scaled = scaler.fit_transform(train_fused)
    val_fused_scaled = scaler.transform(val_fused)

    svm_cfg = fusion_cfg.get("svm", {})
    param_grid = {
        "C": svm_cfg.get("C_grid", [0.1, 1, 10, 100]),
        "kernel": [svm_cfg.get("kernel", "rbf")],
        "gamma": ["scale"],
    }

    # class_weight="balanced": same imbalance rationale as the semantic and
    # temporal branches (see train_semantic.py's compute_class_weights
    # docstring) — Celeb-DF/DFDC skew heavily toward "fake", and an
    # unweighted SVM would happily ignore "real" entirely to minimize
    # hinge loss.
    base_svm = SVC(probability=True, class_weight="balanced")

    # cv=3 (not the sklearn default 5) because a fold count larger than the
    # smallest class's sample count raises inside cross-validation — Val
    # "real" counts can be small on Celeb-DF/DFDC. Lower this further if
    # GridSearchCV still errors on a very small dev run.
    n_splits = min(3, int(np.bincount(train_labels).min()))
    n_splits = max(2, n_splits)
    search = GridSearchCV(base_svm, param_grid, scoring="f1_macro", cv=n_splits, n_jobs=-1)
    search.fit(train_fused_scaled, train_labels)

    best_svm = search.best_estimator_
    val_preds = best_svm.predict(val_fused_scaled)
    val_macro_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)

    print(f"Best params: {search.best_params_}")
    print(f"Val Macro-F1: {val_macro_f1:.4f}")

    os.makedirs(os.path.dirname(args.model_path) or ".", exist_ok=True)
    joblib.dump({
        "svm": best_svm,
        "scaler": scaler,
        "distribution_matcher": matcher,
        "semantic_weight": semantic_weight,
        "temporal_weight": temporal_weight,
        "forensic_weight": forensic_weight,
        "uses_forensic": train_forensic is not None,
        "best_params": search.best_params_,
        "val_macro_f1": val_macro_f1,
    }, args.model_path)

    os.makedirs(os.path.dirname(args.report_path) or ".", exist_ok=True)
    with open(args.report_path, "w") as f:
        json.dump({
            "best_params": search.best_params_,
            "val_macro_f1": val_macro_f1,
            "train_samples": int(len(train_labels)),
            "val_samples": int(len(val_labels)),
        }, f, indent=2)

    print(f"Fusion model saved to {args.model_path}")
    print("Run `python -m src.modeling.test_fusion` for the final test-set evaluation.")


if __name__ == "__main__":
    main()
