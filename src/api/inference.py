"""
Live inference: runs the full hybrid pipeline (semantic + temporal +
forensic -> fusion classifier) over a single uploaded video and returns a
real/fake prediction with a probability score. This is what
src/api/app.py's /predict endpoint calls.

Mirrors extract_embeddings.py's preprocessing exactly, since that's what
the fusion classifier was trained on -- the difference is
extract_embeddings.py reads PRECOMPUTED face crops/sequences cached to
disk by precompute_faces.py / precompute_temporal.py for videos already
in the dataset pipeline, while this module computes them fresh, once,
from an arbitrary uploaded video. _prepare_semantic_input() /
_prepare_temporal_input() below must stay identical to
extract_embeddings.py's _semantic_transform / _temporal_normalize
handling, or embeddings computed here won't be comparable to what the
fusion classifier actually learned from.
"""

import os

import cv2
import joblib
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.features.forensic_extractor import extract_forensic_vector
from src.features.fusion import build_fused_vector
from src.features.semantic_extractor import SemanticClassifier
from src.features.temporal_extractor import TemporalClassifier
from src.preprocessing.face_detection import detect_face
from src.preprocessing.frame_sampler import sample_frames

SEMANTIC_CHECKPOINT = "saved_models/semantic_checkpoint.pth"
TEMPORAL_CHECKPOINT = "saved_models/temporal_checkpoint.pth"
FUSION_MODEL_PATH = "models/fusion_classifier/fusion_model.pkl"

# Must match extract_embeddings.py's _semantic_transform exactly.
_semantic_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Must match extract_embeddings.py's _temporal_normalize exactly.
_temporal_normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


class ModelNotReadyError(RuntimeError):
    """
    Raised when a required checkpoint/model file is missing. app.py turns
    this into a clear 503 HTTP response instead of a raw traceback.
    """


class _PipelineBundle:
    """
    Loads every checkpoint/model exactly once (on the first request) and
    reuses them for every subsequent prediction -- these are process-
    lifetime singletons, not reloaded per request.
    """

    def __init__(self):
        self.device = None
        self.semantic_model = None
        self.temporal_model = None
        self.fusion_bundle = None

    def ensure_loaded(self):
        if self.fusion_bundle is not None:
            return  # already loaded

        missing = [
            p for p in (SEMANTIC_CHECKPOINT, TEMPORAL_CHECKPOINT, FUSION_MODEL_PATH)
            if not os.path.exists(p)
        ]
        if missing:
            raise ModelNotReadyError(
                "Missing required model file(s): " + ", ".join(missing) +
                ". Run train_semantic.py, train_temporal.py, "
                "extract_embeddings.py, and train_fusion.py first."
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.semantic_model = SemanticClassifier().to(self.device)
        checkpoint = torch.load(SEMANTIC_CHECKPOINT, map_location=self.device)
        self.semantic_model.load_state_dict(checkpoint["model_state_dict"])
        self.semantic_model.eval()

        self.temporal_model = TemporalClassifier().to(self.device)
        checkpoint = torch.load(TEMPORAL_CHECKPOINT, map_location=self.device)
        self.temporal_model.load_state_dict(checkpoint["model_state_dict"])
        self.temporal_model.eval()

        self.fusion_bundle = joblib.load(FUSION_MODEL_PATH)


_bundle = _PipelineBundle()


def _prepare_semantic_input(face_image_bgr):
    rgb = cv2.cvtColor(face_image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    return _semantic_transform(pil_image)


def _prepare_temporal_input(face_sequence_bgr):
    sequence = face_sequence_bgr[..., ::-1].copy()  # BGR -> RGB, matches temporal_dataset.py
    sequence = torch.from_numpy(sequence).permute(0, 3, 1, 2).float() / 255.0
    sequence = torch.stack([_temporal_normalize(frame) for frame in sequence])
    return sequence


import time

@torch.no_grad()
def predict_video(video_path):
    """
    Runs the full pipeline over `video_path` and returns comprehensive
    multi-modal diagnostics for the web UI and REST API.
    """
    start_time = time.time()
    _bundle.ensure_loaded()
    device = _bundle.device

    # 1. Semantic input: representative face crop
    single_frame = sample_frames(video_path, num_frames=1)
    face_image = detect_face(single_frame[0])  # [224, 224, 3] uint8 BGR

    # 2. Temporal input: 16-frame face sequence
    sequence_frames = sample_frames(video_path, num_frames=16, size=(224, 224))
    face_sequence = np.zeros((16, 224, 224, 3), dtype=np.uint8)
    for i in range(sequence_frames.shape[0]):
        face_sequence[i] = detect_face(sequence_frames[i])

    # 3. Forensic handcrafted features
    forensic_vector = extract_forensic_vector(video_path, face_image, face_sequence)

    # 4. Neural Branch Embeddings
    semantic_tensor = _prepare_semantic_input(face_image).unsqueeze(0).to(device)
    temporal_tensor = _prepare_temporal_input(face_sequence).unsqueeze(0).to(device)

    semantic_embedding, semantic_logits = _bundle.semantic_model(semantic_tensor)
    temporal_embedding, temporal_logits = _bundle.temporal_model(temporal_tensor)

    semantic_embedding = semantic_embedding.cpu().numpy()
    temporal_embedding = temporal_embedding.cpu().numpy()
    forensic_embedding = forensic_vector[np.newaxis, :]

    # Branch-specific probabilities from neural heads
    sem_prob_fake = float(torch.softmax(semantic_logits, dim=1)[0, 1].item())
    temp_prob_fake = float(torch.softmax(temporal_logits, dim=1)[0, 1].item())

    # 5. Hybrid Feature Fusion & Distribution Matching
    bundle = _bundle.fusion_bundle
    matcher = bundle["distribution_matcher"]
    semantic_weight = bundle["semantic_weight"]
    temporal_weight = bundle["temporal_weight"]
    forensic_weight = bundle.get("forensic_weight", 1.0)
    uses_forensic = bundle.get("uses_forensic", False)

    forensic_arg = forensic_embedding if uses_forensic else None

    base_fused = build_fused_vector(
        semantic_embedding, temporal_embedding,
        semantic_weight=semantic_weight, temporal_weight=temporal_weight,
        forensic_embeddings=forensic_arg, forensic_weight=forensic_weight,
    )
    distribution_score = float(matcher.score(base_fused)[0])
    fused = build_fused_vector(
        semantic_embedding, temporal_embedding,
        semantic_weight=semantic_weight, temporal_weight=temporal_weight,
        forensic_embeddings=forensic_arg, forensic_weight=forensic_weight,
        distribution_scores=np.array([distribution_score]),
    )

    fused_scaled = bundle["scaler"].transform(fused)
    probs = bundle["svm"].predict_proba(fused_scaled)[0]  # [P(real), P(fake)]

    fake_probability = float(probs[1])
    real_probability = float(probs[0])
    prediction = "fake" if fake_probability >= 0.5 else "real"
    confidence = float(max(fake_probability, real_probability) * 100.0)
    elapsed_ms = round((time.time() - start_time) * 1000, 1)

    # Handcrafted feature telemetry
    # SRM energy (first 24 dims mean), texture variance, landmark jitter, rPPG variance
    srm_energy = float(np.mean(np.abs(forensic_vector[:24])))
    texture_score = float(np.mean(forensic_vector[24:47]))
    landmark_stability = float(np.mean(forensic_vector[47:59]))
    rppg_pulse_var = float(forensic_vector[59] if len(forensic_vector) > 59 else 0.0)

    # Reasons list for UI breakdown
    reasons = [
        f"Semantic Spatial Model: {sem_prob_fake*100:.1f}% manipulation likelihood ({'Spatial artifacts detected' if sem_prob_fake >= 0.5 else 'Normal facial texture'}).",
        f"Temporal Dynamic Model: {temp_prob_fake*100:.1f}% manipulation likelihood ({'Inter-frame motion anomaly' if temp_prob_fake >= 0.5 else 'Consistent temporal flow'}).",
        f"Distribution-matching distance from authentic baseline: {distribution_score:.2f} ({'Statistical divergence detected' if distribution_score > 35 else 'Conforms to authentic distribution'}).",
        f"Biological rPPG pulse variation: {rppg_pulse_var:.4f} with SRM noise residual energy of {srm_energy:.4f}.",
    ]

    return {
        "prediction": prediction,
        "verdict": "Manipulated / Deepfake" if prediction == "fake" else "Authentic / Real",
        "confidence": round(confidence, 2),
        "fake_probability": round(fake_probability, 4),
        "real_probability": round(real_probability, 4),
        "signals": {
            "distribution_score": round(distribution_score, 2),
            "landmark_motion_valid": landmark_stability > 0.01,
            "rppg_valid": rppg_pulse_var > 0.001,
        },
        "reasons": reasons,
        "branch_scores": {
            "semantic_spatial": {
                "score": round(sem_prob_fake, 4),
                "label": "Suspicious Spatial Artifacts" if sem_prob_fake >= 0.5 else "Normal Face Geometry",
            },
            "temporal_consistency": {
                "score": round(temp_prob_fake, 4),
                "label": "Inter-frame Inconsistency" if temp_prob_fake >= 0.5 else "Natural Temporal Flow",
            },
            "distribution_distance": {
                "mahalanobis_distance": round(distribution_score, 4),
                "status": "Divergent from genuine distribution" if distribution_score > 35 else "Conforms to authentic baseline",
            },
        },
        "forensic_signals": {
            "srm_residual_noise": round(srm_energy, 4),
            "texture_anomaly": round(texture_score, 4),
            "facial_landmark_jitter": round(landmark_stability, 4),
            "rppg_biological_pulse": round(rppg_pulse_var, 4),
        },
        "metadata": {
            "device": str(device).upper(),
            "frames_sampled": 16,
            "inference_time_ms": elapsed_ms,
        }
    }
