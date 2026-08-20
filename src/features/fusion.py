"""
Feature fusion for the hybrid classifier (Phase 3 of the pipeline).

Combines the semantic branch's embedding (EfficientNet-B0, 256-d) and the
temporal branch's embedding (CNN+LSTM, 256-d) into a single fused feature
vector, per the proposal's methodology:
  4. Data Preprocessing   -> normalize/scale (handled by StandardScaler in
                              train_fusion.py / test_fusion.py, not here)
  5. Feature Fusion       -> concatenate branch embeddings with adaptive
                              weighting (build_fused_vector below)
  6. Distribution Matching -> statistical distance from the learned "real"
                              media distribution, appended as one extra
                              engineered feature (DistributionMatcher below)

Handcrafted forensic features (SRM, landmark motion, statistical texture,
rPPG) are appended into this same fused vector once
src/features/forensic_extractor.py lands — build_fused_vector() already
accepts an optional forensic_embeddings array for exactly that reason, so
train_fusion.py / test_fusion.py won't need structural changes when that
branch is ready.
"""

import numpy as np
from sklearn.covariance import LedoitWolf


class DistributionMatcher:
    """
    Learns the distribution of "real" media in fused-embedding space from
    the training set, then scores any new sample by its statistical
    distance from that learned distribution (Mahalanobis distance).

    Per the proposal: "During training, the system learns the distribution
    of the features of genuine media content... If there are significant
    differences [from a query sample], it indicates AI-generated media
    content." That distance is exposed here as one additional feature for
    the downstream classifier to weigh, rather than a hard-coded threshold
    — the SVM learns how much to trust it during train_fusion.py's fit.

    Covariance is estimated with Ledoit-Wolf shrinkage rather than the raw
    sample covariance. This matters a lot here specifically: "real" videos
    are the minority class (Celeb-DF v2 is ~86% fake / ~14% real, and DFDC's
    sample set isn't much better), so the number of real-class TRAINING
    examples this fits on can easily be smaller than the fused vector's
    dimensionality (256 semantic + 256 temporal = 512-d, before forensic
    features are even added). A plain sample covariance is singular in that
    regime and a small fixed ridge term (tried initially) doesn't reliably
    fix it — a synthetic sanity test with n=40 real samples in 512-d
    produced wildly unstable distances (two orders of magnitude apart for a
    mild, controlled mean shift). Ledoit-Wolf shrinkage is the standard,
    well-conditioned fix for exactly this n << d regime and shrinks toward
    a scaled identity by a data-driven amount rather than a hand-picked
    constant.
    """

    def __init__(self):
        self.mean_ = None
        self.inv_cov_ = None

    def fit(self, real_embeddings):
        real_embeddings = np.asarray(real_embeddings, dtype=np.float64)
        if len(real_embeddings) < 2:
            raise ValueError(
                "DistributionMatcher.fit needs at least 2 real-class "
                f"samples, got {len(real_embeddings)}."
            )

        self.mean_ = real_embeddings.mean(axis=0)
        estimator = LedoitWolf().fit(real_embeddings)
        # LedoitWolf already exposes the (well-conditioned) inverse of its
        # shrunk covariance estimate directly -- no separate np.linalg.inv
        # call needed.
        self.inv_cov_ = estimator.precision_
        return self

    def score(self, embeddings):
        """
        Returns the Mahalanobis distance of each row in `embeddings` from
        the learned real-media distribution. Higher = less like genuine
        media = more suspicious.
        """
        if self.mean_ is None:
            raise RuntimeError("DistributionMatcher.fit() must be called before score().")

        embeddings = np.asarray(embeddings, dtype=np.float64)
        diff = embeddings - self.mean_
        left = diff @ self.inv_cov_
        dist_sq = np.einsum("ij,ij->i", left, diff)
        dist_sq = np.clip(dist_sq, a_min=0, a_max=None)  # guard tiny negative values from float error
        return np.sqrt(dist_sq)


def build_fused_vector(semantic_embeddings, temporal_embeddings,
                        semantic_weight=1.0, temporal_weight=1.0,
                        forensic_embeddings=None, forensic_weight=1.0,
                        distribution_scores=None):
    """
    Concatenates branch embeddings into one fused vector per sample, with
    each branch scaled by its configured weight before concatenation
    (config.yaml's models.fusion.semantic_weight / temporal_weight /
    forensic_weight).

    Weights may be scalars (global, static weighting — what's used today)
    or per-sample arrays of shape [N, 1] (adaptive weighting — what the
    proposal describes: "if compression is detected... the weight of
    temporal and rPPG features is higher"). Both broadcast correctly
    against [N, D] embeddings, so train_fusion.py won't need to change
    when forensic_extractor.py's compression detector starts producing
    per-sample weights.

    distribution_scores, if given, is appended as one extra scalar column.
    """
    semantic_embeddings = np.asarray(semantic_embeddings, dtype=np.float32)
    temporal_embeddings = np.asarray(temporal_embeddings, dtype=np.float32)

    parts = [
        semantic_embeddings * semantic_weight,
        temporal_embeddings * temporal_weight,
    ]

    if forensic_embeddings is not None:
        forensic_embeddings = np.asarray(forensic_embeddings, dtype=np.float32)
        parts.append(forensic_embeddings * forensic_weight)

    fused = np.concatenate(parts, axis=1)

    if distribution_scores is not None:
        distribution_scores = np.asarray(distribution_scores, dtype=np.float32).reshape(-1, 1)
        fused = np.concatenate([fused, distribution_scores], axis=1)

    return fused
