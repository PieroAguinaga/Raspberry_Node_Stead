import cv2
import torch
import numpy as np

def read_frames(url, num_frames=16, stride = 1,size =(224,224)):
    cap = cv2.VideoCapture(url)
    device = "cpu"

    size=(256, 256)

    if not cap.isOpened():
        raise RuntimeError(f"❌ No se pudo abrir el stream: {url}")

    frames = []
    print(f"🎥 Capturando {num_frames} frames desde {url}")

    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Frame no válido, deteniendo captura.")
            break

        # Redimensionar y convertir BGR -> RGB
        frame = cv2.resize(frame, size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Normalizar a [0, 1] y convertir a tensor
        frame_tensor = torch.from_numpy(frame).float() / 255.0  # [H, W, C]
        frame_tensor = frame_tensor.permute(2, 0, 1)  # [C, H, W]
        frames.append(frame_tensor)

    cap.release()

    if not frames:
        raise RuntimeError("❌ No se capturaron frames válidos.")

    # Stack en un solo tensor [N, C, H, W]
    video_tensor = torch.stack(frames).permute(1,0,2,3).to(device)
    # Stack en un solo tensor [C, N, H, W]
    print(f"✅ Tensor de video capturado: {video_tensor.shape}")
    return video_tensor