import cv2
import numpy as np


def detect_face(frame):
    """
    Input:
        frame -> tensor [3, 224, 224]

    Output:
        face_crop -> numpy array [224, 224, 3]
    """

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Convert tensor -> numpy
    img = frame.permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    # If no face found
    if len(faces) == 0:
        return cv2.resize(img, (224, 224))

    # Take first face
    x, y, w, h = faces[0]

    face_crop = img[y:y+h, x:x+w]

    # Resize to standard input size
    face_crop = cv2.resize(face_crop, (224, 224))

    return face_crop