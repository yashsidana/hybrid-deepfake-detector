import os
import json
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from src.preprocessing.face_detection import FaceDetector

class DFDCVideoDataset(Dataset):
      """
          On-the-fly Video Dataset loader for DFDC (DeepFake Detection Challenge).
              Extracts frames from video files, crops faces, and prepares tensors.

                      Supports two modes:
                          - 'frame' mode: returns a single random face crop from a video (for training frame-level CNNs)
                              - 'sequence' mode: returns a sequence of face crops of length `seq_len` (for temporal/sequence models)
                                  """
      def __init__(self, raw_dir, metadata_path, mode='frame', seq_len=10, transform=None, device=None):
                self.raw_dir = raw_dir
                self.mode = mode
                self.seq_len = seq_len
                self.device = device

        # Initialize Face Detector
                self.detector = FaceDetector(image_size=224, device=self.device)

        # Load metadata
                if os.path.exists(metadata_path):
                              with open(metadata_path, 'r') as f:
                                                self.metadata = json.load(f)
                else:
                              self.metadata = {}
                              print(f"Warning: Metadata file not found at {metadata_path}. Dataset will be empty.")

        # Get list of videos present in directory
                self.video_files = [f for f in os.listdir(raw_dir) if f.endswith('.mp4')]

        # Filter metadata to keep only existing files
                self.samples = []
                for filename in self.video_files:
                              if filename in self.metadata:
                                                label_str = self.metadata[filename]['label']
                                                label = 1 if label_str == 'FAKE' else 0
                                                self.samples.append((os.path.join(raw_dir, filename), label))
else:
                # Default label if missing from metadata (e.g. for user testing)
                  self.samples.append((os.path.join(raw_dir, filename), 0))

        # Default transformations if none provided
          if transform:
                        self.transform = transform
else:
              self.transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
              ])

    def __len__(self):
              return len(self.samples)

    def _get_frames(self, video_path):
              """
                      Reads video and returns all extracted BGR frames.
                              """
              cap = cv2.VideoCapture(video_path)
              frames = []
              while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                              break
                                          frames.append(frame)
                        cap.release()
        return frames

    def __getitem__(self, idx):
              video_path, label = self.samples[idx]
        frames = self._get_frames(video_path)

        if len(frames) == 0:
                      # Fallback if video reading failed
                      raise ValueError(f"Could not read frames from video: {video_path}")

        # Extract face crops
        face_crops = []

        # In 'frame' mode, we shuffle/randomize and look for a valid face crop
        if self.mode == 'frame':
                      # Try to find at least one face from a randomly sampled subset of frames
                      import random
                      indices = list(range(len(frames)))
                      random.shuffle(indices)

            for f_idx in indices:
                              face = self.detector.detect_and_crop(frames[f_idx])
                              if face is not None:
                                                    # Convert BGR back to PIL for transforms
                                                    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                                                    face_pil = Image.fromarray(face_rgb)
                                                    return self.transform(face_pil), label

                          # Fallback: if no face detected in any frame, return middle frame resized
                          mid_frame = frames[len(frames) // 2]
            mid_rgb = cv2.cvtColor(mid_frame, cv2.COLOR_BGR2RGB)
            fallback_pil = Image.fromarray(mid_rgb).resize((224, 224))
            return self.transform(fallback_pil), label

        # In 'sequence' mode, we sample `seq_len` evenly spaced frames and extract fac    es
        elif self.mode == 'sequence':
                    indices = [int(i) for i in torch.linspace(0, len(frames) - 1, self.seq_len)]
                        last_valid_face = None
            for f_idx in indices:
                                      face = self.detector.detect_and_crop(frames[f_idx])
                                  if face is not None:
                                                    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                                                    face_pil = Image.fromarray(face_rgb)
                                                    last_valid_face = self.transform(face_pil)
                                            elif last_valid_face is not None:                
                            pass    
                            else:
                                              mid_frame = frames[f_idx]
                                              mid_rgb = cv2.cvtColor(mid_frame, cv2.COLOR_BGR2RGB)
                          fallback_p    il = Image.fromarray(mid_rgb).resize((224, 224))
                                          last_valid_face = self.transform(fallback_pil)

                face_crops.append(last_valid_face)
            # Stack along sequence dimension: (seq_len, 3, 224, 224)
            return torch.stack(face_crops), label

        raise ValueError(f"Unknown dataset mode: {self.mode}")

if __name__ == "__main__":
      # Test initialization
      # Create mock paths
      dataset = DFDCVideoDataset(
                raw_dir="data/raw", 
                metadata_path="data/metadata.json",
                mode='frame'
      )
      print(f"Dataset initialized with {len(dataset)} video samples.")
  
