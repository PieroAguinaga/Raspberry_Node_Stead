import torch
from load_custom_model import load_custom_model
from load_x3d import load_x3d
import numpy as np
from capture_frames import read_frames
import time 
import requests
import json
from datetime import datetime
from simluacion import start_camera
import argparse

def main(args):
    # Iniciar simulación de cámara IP con video de prueba
    start_camera(args.video)

    # Dirección de la cámara IP (stream de Flask)
    ip_cam = "http://localhost:5000/loop"

    # Cargar modelos
    device = torch.device("cpu")
    model_x3d, transform, transform_params = load_x3d(args.x3d_version, device, args.num_frames, args.stride)
    model_custom = load_custom_model(args.model_name, args.arch, device)

    clip_frame_count = args.num_frames * args.stride
    size = (transform_params["side_size"], transform_params["side_size"])

    while True:
        # Leer frames desde la IP cam simulada
        clip_tensor = read_frames(ip_cam, args.num_frames, args.stride, size)
        clip_dict = {"video": clip_tensor}
        transformed_clip = transform(clip_dict)["video"]  # (T, C, H, W)

        with torch.no_grad():
            features = model_x3d(transformed_clip.unsqueeze(0).to(device)).cpu().numpy().squeeze()

        vector = torch.from_numpy(np.expand_dims(features, axis=0)).to(device)

        with torch.no_grad():
            score, _ = model_custom(vector)
            score = torch.sigmoid(score).squeeze().item()

        # Enviar resultado al servidor
        try:
            payload = {
                "date": datetime.now().isoformat(),
                "camera_id": args.camera_id,
                "score": score
            }

            headers = {
                "Content-Type": "application/json"
            }

            response = requests.post(args.endpoint, data=json.dumps(payload), headers=headers)
            print(f"✅ Score enviado: {payload} → Código {response.status_code}")
        except Exception as e:
            print(f"❌ Error al enviar: {e}")
            continue

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--video", type=int, required=True, help="ID del video de demo (ej. 1 = video_1.mp4)")
    parser.add_argument("--x3d_version", type=str, default="x3d_xs", help="Versión del modelo X3D (ej. x3d_xs, x3d_m)")
    parser.add_argument("--num_frames", type=int, default=16, help="Cantidad de frames por clip")
    parser.add_argument("--stride", type=int, default=2, help="Stride entre frames")
    parser.add_argument("--model_name", type=str, required=True, help="Nombre del modelo custom sin .pkl")
    parser.add_argument("--arch", type=str, choices=["base", "fast", "tiny"], default="base", help="Arquitectura del modelo custom")
    parser.add_argument("--camera_id", type=int, default=1, help="ID de la cámara simulada")
    parser.add_argument("--endpoint", type=str, default="http://localhost:8080/anomaly", help="Endpoint para enviar resultados")

    args = parser.parse_args()
    main(args)

    #python main.py --video 1 x3d_version xs num_frames 13 --stride 6 --model_name STEAD_FAST_S_13_6final --arch fast --camera_id 1 --endpoint http://localhost:8080/anomaly