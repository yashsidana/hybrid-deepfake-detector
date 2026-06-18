import cv2
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN

class FaceDetector:
      """
          MTCNN-based Face Detection and Cropping Module.
              Locates the primary face in a frame, crops it, and resizes it to a target size (default: 224x224).
                  """
      def __init__(self, image_size=224, threshold=0.90, device=None):
                self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
                self.image_size = image_size

        # Initialize MTCNN
                # keep_all=False ensures we only detect the primary/largest face in the frame
                self.mtcnn = MTCNN(
                    image_size=image_size,
                    margin=20,
                    min_face_size=20,
                    thresholds=[0.6, 0.7, 0.7],
                    factor=0.709,
                    post_process=False, # Keep raw pixel values [0, 255]
                    device=self.device,
                    keep_all=False
                )

      def detect_and_crop(self, frame_bgr):
                """
                        Detects the main face in a BGR image frame, crops and returns it as a BGR image.
                                If no face is detected, returns None.
                                        """
                # Convert BGR (OpenCV default) to RGB (PIL/MTCNN default)
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)

        # Detect bounding box and landmarks
                boxes, probs = self.mtcnn.detect(pil_img)

        if boxes is not None and len(boxes) > 0:
                      # Check confidence threshold
                      if probs[0] >= 0.90:
                                        box = boxes[0].astype(int)
                                        # Ensure bounding boxes are within image bounds
                                        x1, y1, x2, y2 = box
                                        x1 = max(0, x1)
                                        y1 = max(0, y1)
                                        x2 = min(pil_img.width, x2)
                                        y2 = min(pil_img.height, y2)

                # Crop and resize
                          face_crop = pil_img.crop((x1, y1, x2, y2))
                face_crop = face_crop.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)

                # Convert back to BGR numpy array for standard OpenCV processing
                return cv2.cvtColor(np.array(face_crop), cv2.COLOR_RGB2BGR)

        return None

if __name__ == "__main__":
      # Test initialization
      detector = FaceDetector()
    print(f"FaceDetector initialized on device: {detector.device}")

    # Create a dummy image (e.g. random noise) and test detection
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    cropped = detector.detect_and_crop(dummy_frame)
    print(f"Dummy frame processing result (expected None for random noise): {cropped}")
