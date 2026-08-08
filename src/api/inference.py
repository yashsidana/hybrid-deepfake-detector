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


@torch.no_grad()
def predict_video(video_path):
    """
    Runs the full pipeline over `video_path` and returns:
        {
            "prediction": "real" | "fake",
            "fake_probability": float in [0, 1],
            "real_probability": float in [0, 1],
        }

    Raises ModelNotReadyError if a required model file is missing.
    """
    _bundle.ensure_loaded()
    device = _bundle.device

    # Semantic input: single representative frame + face crop -- exactly
    # matching precompute_faces.py's contract.
    single_frame = sample_frames(video_path, num_frames=1)
    face_image = detect_face(single_frame[0])  # [224, 224, 3] uint8 BGR

    # Temporal input: 16-frame face sequence -- exactly matching
    # precompute_temporal.py's contract.
    sequence_frames = sample_frames(video_path, num_frames=16, size=(224, 224))
    face_sequence = np.zeros((16, 224, 224, 3), dtype=np.uint8)
    for i in range(sequence_frames.shape[0]):
        face_sequence[i] = detect_face(sequence_frames[i])

    # Forensic vector: SRM + texture (from face_image) + landmark motion
    # (from face_sequence) + rPPG (reads video_path directly, for its own
    # denser frame sampling -- see forensic_extractor.py).
    forensic_vector = extract_forensic_vector(video_path, face_image, face_sequence)

    semantic_tensor = _prepare_semantic_input(face_image).unsqueeze(0).to(device)
    temporal_tensor = _prepare_temporal_input(face_sequence).unsqueeze(0).to(device)

    semantic_embedding, _ = _bundle.semantic_model(semantic_tensor)
    temporal_embedding, _ = _bundle.temporal_model(temporal_tensor)

    semantic_embedding = semantic_embedding.cpu().numpy()
    temporal_embedding = temporal_embedding.cpu().numpy()
    forensic_embedding = forensic_vector[np.newaxis, :]

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
    distribution_score = matcher.score(base_fused)
    fused = build_fused_vector(
        semantic_embedding, temporal_embedding,
        semantic_weight=semantic_weight, temporal_weight=temporal_weight,
        forensic_embeddings=forensic_arg, forensic_weight=forensic_weight,
        distribution_scores=distribution_score,
    )

    fused_scaled = bundle["scaler"].transform(fused)
    probs = bundle["svm"].predict_proba(fused_scaled)[0]  # [P(real), P(fake)]

    fake_probability = float(probs[1])
    real_probability = float(probs[0])

    return {
        "prediction": "fake" if fake_probability >= 0.5 else "real",
        "fake_probability": fake_probability,
        "real_probability": real_probability,
    }
