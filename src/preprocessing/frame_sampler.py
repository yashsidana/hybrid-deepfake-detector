import cv2
import torch


def sample_frames(video_path, num_frames=8, size=(224, 224), max_stride=3):
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames > num_frames:
        stride = min(max_stride, max(1, total_frames // (num_frames * 2)))
    else:
        stride = 1

    frames = []
    count = 0

    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if count % stride == 0:
            frame = cv2.resize(frame, size)
            frame = torch.tensor(frame).permute(2, 0, 1).float() / 255.0
            frames.append(frame)
        count += 1

    cap.release()

    while len(frames) < num_frames:
        frames.append(torch.zeros(3, size[0], size[1]))

    return torch.stack(frames)