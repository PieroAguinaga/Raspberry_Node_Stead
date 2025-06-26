import cv2
import torch
import numpy as np

def read_frames(url, num_frames=16, stride=1, size=(256, 256)):
    cap = cv2.VideoCapture(url)
    device = "cpu"

    if not cap.isOpened():
        raise RuntimeError(f"❌ No se pudo abrir el stream: {url}")

    frames = []
    total_frames_needed = num_frames * stride
    read_count = 0
    saved_count = 0

    print(f"🎥 Capturando {num_frames} frames (stride={stride}) desde {url}")

    while saved_count < num_frames and read_count < total_frames_needed:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Frame no válido, deteniendo captura.")
            break

        if read_count % stride == 0:
            frame = cv2.resize(frame, size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.from_numpy(frame).float() / 255.0  # [H, W, C]
            frame_tensor = frame_tensor.permute(2, 0, 1)  # [C, H, W]
            frames.append(frame_tensor)
            saved_count += 1

        read_count += 1

    cap.release()

    if not frames:
        raise RuntimeError("❌ No se capturaron frames válidos.")

    video_tensor = torch.stack(frames).permute(1, 0, 2, 3).to(device)  # [C, T, H, W]
    print(f"✅ Tensor de video capturado: {video_tensor.shape}")
    return video_tensor