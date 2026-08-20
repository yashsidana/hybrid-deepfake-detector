import cv2
import numpy as np
from facenet_pytorch import MTCNN

# Instantiated once at import time rather than per-call (the old Haar
# cascade version re-loaded cv2.CascadeClassifier on every detect_face()
# call, which was wasteful). MTCNN's weights load once here and are reused
# for every frame.
#
# select_largest=True + keep_all=False: mirrors the old behavior of
# "take the first/most prominent detected face" with a single return value.
_mtcnn = MTCNN(
    image_size=224,
    margin=20,
    post_process=False,
    select_largest=True,
    keep_all=False,
)


def detect_face(frame):
    """
    Input:
        frame -> tensor [3, 224, 224], BGR, values in [0, 1]
                 (this matches what frame_sampler.py produces, since cv2
                 reads frames in BGR and frame_sampler does no color
                 conversion)

    Output:
        face_crop -> numpy array [224, 224, 3], BGR, uint8
                     (identical contract to the previous Haar cascade
                     implementation — no downstream code needs to change)
    """
    img = frame.permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)  # BGR uint8

    # MTCNN expects RGB input.
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    face_tensor = _mtcnn(rgb)

    if face_tensor is None:
        # No face detected — same fallback as the old implementation:
        # return a plain resize of the full frame rather than crashing
        # or returning None (downstream code always expects an image back).
        return cv2.resize(img, (224, 224))

    # face_tensor: [3, 224, 224], float, range ~[0, 255] since post_process=False
    face_rgb = face_tensor.permute(1, 2, 0).numpy().astype(np.uint8)
    face_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)

    return face_bgr



def detect_face_box(frame):
    """
    Returns (x1, y1, x2, y2) pixel coordinates of the largest detected face
    in `frame`, or None if no face is found.

    Unlike detect_face() above, this takes a raw numpy frame straight from
    cv2.VideoCapture.read() (BGR, uint8, native resolution -- NOT the
    [3, 224, 224] float tensor detect_face() expects) and returns a
    location rather than a resized crop. Used by forensic_extractor.py's
    rPPG feature, which needs a fixed region to average color over across
    several consecutive raw frames.

    Reuses the same module-level MTCNN instance as detect_face() -- no
    second copy of the model's weights loaded.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, _ = _mtcnn.detect(rgb)

    if boxes is None or len(boxes) == 0:
        return None

    # .detect() doesn't itself apply select_largest -- pick the largest
    # box explicitly so behavior matches the rest of this module's intent.
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    x1, y1, x2, y2 = boxes[int(areas.argmax())]

    h, w = frame.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))

    if x2 <= x1 or y2 <= y1:
        return None

    return (x1, y1, x2, y2)


def detect_landmarks(frame):
    """
    Returns a [5, 2] float array of (x, y) facial landmark coordinates --
    left eye, right eye, nose, left mouth corner, right mouth corner, in
    that order (facenet-pytorch's MTCNN convention) -- for the largest
    detected face in `frame` (BGR, uint8, native resolution), or None if
    no face is found. Used by forensic_extractor.py's landmark motion
    feature.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, _, landmarks = _mtcnn.detect(rgb, landmarks=True)

    if boxes is None or landmarks is None or len(landmarks) == 0:
        return None

    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    idx = int(areas.argmax())

    return landmarks[idx]
