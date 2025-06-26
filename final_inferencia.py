import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import cv2
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from uuid import uuid4
import pandas as pd

from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms.v2 import CenterCrop, Normalize
from torchvision.transforms import Compose, Lambda
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample
)
from model import Model

# ... [IMPORTS MANTENIDOS IGUALES] ...

def process_video_and_generate_output(
    model_arch: str,
    model_name_suffix: str,
    video_path: str,
    output_folder: str,
    x3d_version: str,
):
    print("🚀 Iniciando proceso de inferencia...")
    device = torch.device("cpu")
    os.makedirs(output_folder, exist_ok=True)

    random_id = uuid4().hex[:3]
    print(f"🆔 ID de ejecución: {random_id}")

    output_filename = f"{model_name_suffix}_STEAD_{video_path}_{random_id}.mp4"
    output_filename_graph = f"{model_name_suffix}_STEAD_{video_path}_{random_id}.png"
    output_filename_graph = os.path.join(output_folder, output_filename_graph)
    output_video_path = os.path.join(output_folder, output_filename)
    video_path = "./videos_pruebas/" + str(video_path)
    print(f"📁 Ruta del video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("? No se pudo abrir el video")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"??? Total frames: {total_frames}, FPS: {fps}")

    # Cargar modelo X3D
    print(f"📦 Cargando extractor X3D: {x3d_version}")
    x3d_versions_map = {
        "xs": "x3d_xs",
        "s": "x3d_s",
        "m": "x3d_m",
        "l": "x3d_l"
    }

    if x3d_version not in x3d_versions_map:
        raise ValueError(f"Versión x3d desconocida: {x3d_version}")

    model_name = x3d_versions_map[x3d_version]
    model_x3d = torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=True)
    model_x3d = model_x3d.eval().to(device)
    print(f"✅ Modelo X3D cargado: {model_name}")

    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]

    model_transform_params = {
        "x3d_xs": {"side_size": 182, "crop_size": 182, "num_frames": 4, "sampling_rate": 12},
        "x3d_s":  {"side_size": 182, "crop_size": 182, "num_frames": 13, "sampling_rate": 6},
        "x3d_m":  {"side_size": 256, "crop_size": 256, "num_frames": 16, "sampling_rate": 5},
        "x3d_l":  {"side_size": 320, "crop_size": 320, "num_frames": 16, "sampling_rate": 5}
    }
    transform_params = model_transform_params[model_name]
    print(f"🧪 Parámetros de transformación: {transform_params}")

    class Permute(nn.Module):
        def __init__(self, dims): super().__init__(); self.dims = dims
        def forward(self, x): return torch.permute(x, self.dims)

    transform = ApplyTransformToKey(
        key="video",
        transform=Compose([
            UniformTemporalSubsample(transform_params["num_frames"]),
            Lambda(lambda x: x / 255.0),
            Permute((1, 0, 2, 3)),
            Normalize(mean, std),
            ShortSideScale(size=transform_params["side_size"]),
            CenterCrop((transform_params["crop_size"], transform_params["crop_size"])),
            Permute((1, 0, 2, 3))
        ])
    )

    
    clip_frame_count = int(transform_params["num_frames"] * transform_params["sampling_rate"])
    if hasattr(model_x3d, "blocks"):
        print(f"ℹ️ Eliminando última capa del modelo...")
        model_x3d.blocks = model_x3d.blocks[:-1]

    # Cargar modelo personalizado
    print("📦 Cargando modelo personalizado...")
    MODEL_PATH = os.path.join("./ckpt", model_name_suffix + ".pkl")

    if model_arch == 'base':
        model_custom = Model()
    elif model_arch in ['fast', 'tiny']:
        model_custom = Model(ff_mult=1, dims=(32, 32), depths=(1, 1))
    else:
        raise ValueError("Arquitectura desconocida.")

    model_custom.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model_custom.eval().to(device)
    print(f"✅ Modelo personalizado cargado desde {MODEL_PATH}")

    clip_frame_count = transform_params["num_frames"] * transform_params["sampling_rate"]
    print(f" Procesando clips de {clip_frame_count} frames")

    clip_idx = 0
    timestamps, durations, scores_list, predictions = [], [], [], []



    for start_frame in tqdm(range(0, total_frames - clip_frame_count, clip_frame_count)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames = []

        time1 = time.time()
        for _ in range(clip_frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
            tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1) 
            frames.append(tensor)
        
        if len(frames) < clip_frame_count:
            continue

        video_tensor = torch.stack(frames).permute(1,0,2,3)  # [T, C, H, W]
        print(video_tensor.shape)

        try:
            transformed = transform({"video": video_tensor.float()})["video"]
            with torch.no_grad():
                features = model_x3d(transformed.unsqueeze(0).to(device)).detach().cpu().numpy().squeeze()
            print(features.shape)
        except Exception as e:
            print(f"? Error al procesar clip {clip_idx}: {e}")
            continue

        try:
            vector = torch.from_numpy(np.expand_dims(features, axis=0)).to(device)
            with torch.no_grad():
                score, _ = model_custom(vector)
                score = torch.sigmoid(score).squeeze().item()
                prediction = 1 if score > 0.5 else 0
        except Exception as e:
            print(f"? Error en inferencia del clip {clip_idx}: {e}")
            continue
        time2= time.time()
        timestamp = (start_frame + clip_frame_count) / fps
        timestamps.append(round(timestamp, 2))
        durations.append(round(time2-time1,3))
        scores_list.append(score)
        predictions.append(prediction)
        clip_idx += 1

    cap.release()
    # Guardar CSV
    df_resultados = pd.DataFrame({
        'clip_idx': list(range(len(scores_list))),
        'start_time_sec': timestamps,
        'tiempo_de_inferencia': durations,
        'score': scores_list,
        'prediction': predictions
    })

    csv_output_path = os.path.join(output_folder, f"scores_{model_name_suffix}_{random_id}.csv")
    df_resultados.to_csv(csv_output_path, index=False)
    print(f"💾 CSV guardado en: {csv_output_path}")

    # ... [El resto del código de escritura del video y gráfico permanece igual, puedes pedírmelo si también deseas prints allí] ...

    print(f"✅ Video generado correctamente: {output_video_path}")



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_arch", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--x3d", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    process_video_and_generate_output(
        model_arch=args.model_arch,
        model_name_suffix=args.model_path,
        video_path=args.video_path,
        output_folder=args.output,
        x3d_version=args.x3d
    )
