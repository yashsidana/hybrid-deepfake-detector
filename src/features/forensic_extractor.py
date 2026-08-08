"""
Handcrafted forensic feature extraction (proposal Methodology section 3,
the "Forensic Feature Extraction (Temporal + Handcrafted)" step -- this
module covers the HANDCRAFTED half; frame-to-frame motion modeling is
already covered by the temporal CNN+LSTM branch in
src/features/temporal_extractor.py).

Four signal families, each producing a fixed-length feature vector that
extract_forensic_vector() concatenates into one forensic embedding per
video, appended into the fused feature vector alongside the semantic and
temporal branch embeddings (see src/features/fusion.py):

  1. SRM (Spatial Rich Model) filtering  -> srm_features()
     Mid-frequency noise-residual statistics, robust to H.264/H.265
     compression in a way raw high-frequency FFT analysis isn't (per the
     proposal's need analysis) -- compression targets high frequencies
     specifically, but leaves the residual patterns these filters pick up
     comparatively intact.

  2. Statistical texture features        -> texture_features()
     Local Binary Pattern histogram + first-order intensity statistics.

  3. Facial landmark motion analysis     -> landmark_motion_features()
     Tracks 5-point landmarks (eyes, nose, mouth corners) across the
     cached 16-frame sequence and summarizes how they move -- unnatural
     face/lip motion is one of the clearer tells generative video leaves
     behind.

  4. rPPG (remote photoplethysmography)  -> rppg_features()
     Looks for a genuine pulse signal in subtle facial color variation.
     Unlike the other three, this reads the RAW VIDEO directly rather
     than the cached face crop/sequence -- see its docstring for why the
     16-frame cache is fundamentally too sparse for this.

Design note on dependencies: landmark detection reuses facenet-pytorch's
MTCNN (already a project dependency via face_detection.py) instead of
adding dlib or mediapipe. 5 points is coarser than a 68-point model, but
avoids a second heavy dependency and its own pretrained weight file for a
capstone-scale system.
"""

import cv2
import numpy as np
from scipy import signal as scipy_signal
from scipy import stats as scipy_stats

from src.preprocessing.face_detection import detect_face_box, detect_landmarks

# --- SRM (Spatial Rich Model) ------------------------------------------
#
# A reduced, practical subset of the classic SRM high-pass filter bank
# (Fridrich & Kodovsky's steganalysis rich model uses ~30 kernels designed
# as CNN input; a capstone-scale system summarizing each residual map into
# a handful of statistics -- rather than feeding raw residuals to a CNN --
# only needs a representative subset to characterize noise-residual
# behavior, not the full bank). Six kernels covering 1st-order,
# 2nd-order, and the two most commonly cited compact SRM kernels
# (SQUARE3x3, KV).
_SRM_KERNELS = [
    np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]], dtype=np.float32),   # 1st-order horizontal
    np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]], dtype=np.float32),   # 1st-order vertical
    np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]], dtype=np.float32),   # 2nd-order vertical
    np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]], dtype=np.float32),   # 2nd-order horizontal
    np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float32) / 4.0,   # SQUARE3x3
    np.array([                                                        # KV (5x5)
        [-1, 2, -2, 2, -1],
        [2, -6, 8, -6, 2],
        [-2, 8, -12, 8, -2],
        [2, -6, 8, -6, 2],
        [-1, 2, -2, 2, -1],
    ], dtype=np.float32) / 12.0,
]

SRM_DIM = len(_SRM_KERNELS) * 4  # mean(|.|), std, skew, kurtosis per kernel = 24


def _residual_stats(residual):
    flat = residual.astype(np.float64).ravel()
    return [
        float(np.mean(np.abs(flat))),
        float(np.std(flat)),
        float(scipy_stats.skew(flat)),
        float(scipy_stats.kurtosis(flat)),
    ]


def srm_features(face_image_bgr):
    """
    face_image_bgr: [224, 224, 3] uint8 BGR (precompute_faces.py's cached
    output). Returns a SRM_DIM-length float32 vector.
    """
    gray = cv2.cvtColor(face_image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    feats = []
    for kernel in _SRM_KERNELS:
        residual = cv2.filter2D(gray, ddepth=cv2.CV_32F, kernel=kernel)
        feats.extend(_residual_stats(residual))
    return np.array(feats, dtype=np.float32)


# --- Statistical texture (LBP + first-order intensity stats) -----------

_LBP_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
_LBP_HIST_BINS = 16

TEXTURE_DIM = _LBP_HIST_BINS + 1 + 4  # histogram + entropy + 4 intensity stats = 21


def _lbp_codes(gray):
    """
    Standard 8-neighbor, radius-1 LBP: for each pixel, an 8-bit code where
    bit i is 1 if that neighbor's intensity >= the center pixel's.
    Vectorized over the whole image (no per-pixel Python loop) via shifted
    array comparisons.
    """
    h, w = gray.shape
    center = gray[1:h - 1, 1:w - 1]
    code = np.zeros_like(center, dtype=np.uint8)
    for i, (dy, dx) in enumerate(_LBP_OFFSETS):
        neighbor = gray[1 + dy:h - 1 + dy, 1 + dx:w - 1 + dx]
        code |= ((neighbor >= center).astype(np.uint8) << i)
    return code


def texture_features(face_image_bgr):
    """
    face_image_bgr: [224, 224, 3] uint8 BGR. Returns a TEXTURE_DIM-length
    float32 vector: a coarse (16-bin, not the full 256-value) LBP
    histogram + its entropy, plus mean/std/skew/kurtosis of raw pixel
    intensity. The coarse binning keeps this compact and less prone to
    overfitting to per-video noise than a full 256-bin histogram would be
    on datasets with a few hundred to a few thousand videos.
    """
    gray = cv2.cvtColor(face_image_bgr, cv2.COLOR_BGR2GRAY)
    codes = _lbp_codes(gray)

    hist, _ = np.histogram(codes, bins=_LBP_HIST_BINS, range=(0, 256))
    hist = hist.astype(np.float64)
    hist = hist / (hist.sum() + 1e-8)
    hist_entropy = float(-(hist * np.log2(hist + 1e-8)).sum())

    gray_f = gray.astype(np.float64).ravel()
    intensity_stats = [
        float(np.mean(gray_f)),
        float(np.std(gray_f)),
        float(scipy_stats.skew(gray_f)),
        float(scipy_stats.kurtosis(gray_f)),
    ]

    return np.concatenate([
        hist.astype(np.float32),
        np.array([hist_entropy], dtype=np.float32),
        np.array(intensity_stats, dtype=np.float32),
    ])


# --- Facial landmark motion ---------------------------------------------

LANDMARK_MOTION_DIM = 12


def _fill_missing_landmarks(landmarks_per_frame):
    """
    Forward-fills frames where MTCNN found no face with the nearest prior
    successful detection (and backfills any leading gap with the first
    successful one), so one blurry or extreme-angle frame doesn't zero out
    the whole feature. Returns None if EVERY frame failed.
    """
    valid_idx = [i for i, lm in enumerate(landmarks_per_frame) if lm is not None]
    if not valid_idx:
        return None

    filled = list(landmarks_per_frame)
    last = filled[valid_idx[0]]
    for i in range(len(filled)):
        if filled[i] is None:
            filled[i] = last
        else:
            last = filled[i]
    return filled


def landmark_motion_features(face_sequence):
    """
    face_sequence: [T, 224, 224, 3] uint8 BGR -- the same array
    precompute_temporal.py caches (already face-cropped per frame).

    Returns (features, valid): features is a LANDMARK_MOTION_DIM-length
    float32 vector; valid is False (features all-zero) only if landmark
    detection failed on every single frame.
    """
    landmarks_per_frame = [detect_landmarks(frame) for frame in face_sequence]
    detection_rate = sum(lm is not None for lm in landmarks_per_frame) / len(landmarks_per_frame)

    filled = _fill_missing_landmarks(landmarks_per_frame)
    if filled is None:
        return np.zeros(LANDMARK_MOTION_DIM, dtype=np.float32), False

    filled = np.stack(filled).astype(np.float64)  # [T, 5, 2]
    velocity = np.diff(filled, axis=0)             # [T-1, 5, 2]
    speed = np.linalg.norm(velocity, axis=2)        # [T-1, 5] per-landmark displacement magnitude

    # facenet-pytorch's 5-point order: left eye, right eye, nose, left
    # mouth corner, right mouth corner.
    eye_speed = speed[:, 0:2].mean(axis=1)
    mouth_speed = speed[:, 3:5].mean(axis=1)
    jitter = np.diff(speed, axis=0).std() if len(speed) > 1 else 0.0

    features = np.array([
        speed.mean(), speed.std(), speed.max(), speed.var(),
        mouth_speed.mean(), mouth_speed.std(),
        eye_speed.mean(), eye_speed.std(),
        jitter,
        filled[:, :, 0].std(), filled[:, :, 1].std(),  # overall positional spread, x and y
        detection_rate,  # itself informative -- a face MTCNN struggles to lock onto consistently can be a signal too
    ], dtype=np.float32)

    return features, True


# --- rPPG (remote photoplethysmography) ---------------------------------

RPPG_DIM = 4
_RPPG_WINDOW_SECONDS = 5.0
_RPPG_MAX_FRAMES = 150
_RPPG_MIN_SECONDS = 1.5  # below this, there aren't enough samples to resolve anything in the 0.7-4 Hz band
_HR_BAND_HZ = (0.7, 4.0)  # 42-240 bpm, generous physiological bounds


def rppg_features(video_path):
    """
    video_path: path to the RAW video file -- NOT the cached 16-frame
    sequence precompute_temporal.py produces.

    Why raw video and not the cache: the temporal branch's 16 frames are
    sampled UNIFORMLY ACROSS THE ENTIRE VIDEO (see frame_sampler.py), which
    for a several-second clip means an effective sampling rate far below
    what's needed to resolve a ~1-2 Hz heartbeat signal (Nyquist). Running
    rPPG on that cache would alias into meaningless noise. This function
    instead reads a short, temporally DENSE, roughly-centered window of up
    to 150 consecutive frames (~5s at 30fps) directly from the source
    video, which is what rPPG actually needs.

    Approach: track a single face bounding box for the whole window
    (detected once, reused -- an accepted simplifying assumption for
    lightweight rPPG over a short clip with limited head motion), average
    the green channel within it per frame (green carries the strongest
    plethysmographic signal of the three color channels), detrend,
    band-pass filter to the physiological range, and characterize how much
    the resulting spectrum concentrates into one narrow peak -- a genuine
    pulse does; camera/compression noise, and reportedly the color
    dynamics of generated faces, largely don't.

    Returns (features, valid): features is an RPPG_DIM-length float32
    vector [peak_frequency_hz, peak_power, peak_to_total_power_ratio,
    signal_variance]; valid is False (features all-zero) if the video is
    too short/unreadable to attempt this at all.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return np.zeros(RPPG_DIM, dtype=np.float32), False

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    n_frames = min(_RPPG_MAX_FRAMES, total_frames, int(fps * _RPPG_WINDOW_SECONDS))
    if total_frames <= 0 or n_frames < int(fps * _RPPG_MIN_SECONDS):
        cap.release()
        return np.zeros(RPPG_DIM, dtype=np.float32), False

    start_frame = max(0, (total_frames - n_frames) // 2)  # centered window
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    face_box = None
    green_means = []

    for _ in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break

        if face_box is None:
            face_box = detect_face_box(frame)
            if face_box is None:
                h, w = frame.shape[:2]
                face_box = (w // 4, h // 4, 3 * w // 4, 3 * h // 4)  # central-crop fallback

        x1, y1, x2, y2 = face_box
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        green_means.append(float(roi[:, :, 1].mean()))  # BGR -> channel 1 is green

    cap.release()

    if len(green_means) < int(fps * _RPPG_MIN_SECONDS):
        return np.zeros(RPPG_DIM, dtype=np.float32), False

    raw_signal = np.array(green_means, dtype=np.float64)
    detrended = scipy_signal.detrend(raw_signal)

    nyquist = fps / 2.0
    low = _HR_BAND_HZ[0] / nyquist
    high = min(_HR_BAND_HZ[1], nyquist - 1e-3) / nyquist

    if 0 < low < high < 1.0:
        b, a = scipy_signal.butter(3, [low, high], btype="band")
        filtered = scipy_signal.filtfilt(b, a, detrended)
    else:
        # fps too low to meaningfully band-pass at these frequencies --
        # still compute a spectrum below on the detrended signal rather
        # than failing outright.
        filtered = detrended

    freqs = np.fft.rfftfreq(len(filtered), d=1.0 / fps)
    power = np.abs(np.fft.rfft(filtered)) ** 2
    total_power = float(power.sum())

    band_mask = (freqs >= _HR_BAND_HZ[0]) & (freqs <= _HR_BAND_HZ[1])
    if not band_mask.any() or total_power <= 0:
        return np.zeros(RPPG_DIM, dtype=np.float32), False

    band_power = power[band_mask]
    band_freqs = freqs[band_mask]
    peak_idx = int(np.argmax(band_power))

    features = np.array([
        band_freqs[peak_idx],                       # peak_frequency_hz
        band_power[peak_idx],                        # peak_power (unnormalized)
        band_power[peak_idx] / (total_power + 1e-8),  # peak_to_total_power_ratio -- narrow, strong peak = plausible pulse
        float(np.var(filtered)),                      # signal_variance
    ], dtype=np.float32)

    return features, True


# --- Orchestrator ---------------------------------------------------------

FORENSIC_VECTOR_DIM = SRM_DIM + TEXTURE_DIM + LANDMARK_MOTION_DIM + RPPG_DIM + 2  # +2 validity flags


def extract_forensic_vector(video_path, face_image, face_sequence):
    """
    video_path: absolute path to the raw video (rPPG only).
    face_image: [224, 224, 3] uint8 BGR, from precompute_faces.py's cache.
    face_sequence: [16, 224, 224, 3] uint8 BGR, from precompute_temporal.py's cache.

    Returns one FORENSIC_VECTOR_DIM-length float32 vector: SRM + texture
    (always computed -- these only need the single cached face image) +
    landmark motion + rPPG (each with its own validity flag appended at
    the end, since these two can legitimately fail on some videos -- e.g.
    a face MTCNN can't lock onto across a full 5-frame window, or an
    unreadable/too-short raw video -- without the whole video needing to
    be dropped from fusion training the way precompute_faces.py /
    precompute_temporal.py drop videos where face detection fails
    entirely).
    """
    srm = srm_features(face_image)
    texture = texture_features(face_image)
    landmark_motion, landmark_valid = landmark_motion_features(face_sequence)
    rppg, rppg_valid = rppg_features(video_path)

    validity = np.array([float(landmark_valid), float(rppg_valid)], dtype=np.float32)

    return np.concatenate([srm, texture, landmark_motion, rppg, validity]).astype(np.float32)
