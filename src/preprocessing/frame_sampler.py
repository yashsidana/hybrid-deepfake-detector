import cv2
import torch


def sample_frames(video_path, num_frames=8, size=(224, 224)):
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total_frames // num_frames, 1)

    frames = []

    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, size)
        frame = torch.tensor(frame).permute(2, 0, 1).float() / 255.0

        frames.append(frame)

    cap.release()

    while len(frames) < num_frames:
        frames.append(torch.zeros(3, size[0], size[1]))

    return torch.stack(frames)