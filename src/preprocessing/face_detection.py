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
