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
from buffer import FrameBuffer
from option import parse_args
import cv2
import threading

def enviar_score(payload, endpoint):
    try:
        print(f"📤 [Async] Enviando datos al endpoint: {endpoint}")
        r = requests.post(endpoint, json=payload, headers={"Content-Type": "application/json"})
        print(f"✅ [Async] Respuesta del servidor: Código {r.status_code}")
    except Exception as e:
        print(f"❌ [Async] Error al enviar: {e}")

def main(args):
    print("🚀 Iniciando sistema de detección de anomalías...")

    # 🟡 Lanzar simulación de cámara
    print(f"🟡 Lanzando simulación de cámara con video ID = {args.video}")
    start_camera(args.video)

    ip_cam = "http://localhost:5000/loop"
    print(f"🌐 Dirección del stream: {ip_cam}")

    # 🔄 FrameBuffer
    buffer_size = args.num_frames * args.stride * 2
    fb = FrameBuffer(ip_cam, maxlen=buffer_size).start()
    print("✅ FrameBuffer iniciado correctamente.")
    time.sleep(5)

    # 🧠 Cargar modelos
    device = torch.device("cpu")
    model_x3d, transform, params = load_x3d(args.x3d_version, device, args.num_frames, args.stride)
    model_custom = load_custom_model(args.model_name, args.arch, device)

    size = (params["side_size"], params["side_size"])
    print(f"📏 Tamaño de redimensionamiento: {size}")

    # 🕒 Calcular tiempo mínimo entre inferencias
    fps = 30  # puedes parametrizarlo si tu cámara no es 30fps
    intervalo_seg = (args.num_frames * args.stride) / fps
    print(f"⏲️ Ventana de inferencia cada {intervalo_seg:.2f} segundos")

    try:
        while True:
            T1 = time.time()
            date = datetime.now().isoformat()

            print("\n📸 Obteniendo frames del buffer...")
            raw_frames = fb.read_recent(args.num_frames, args.stride)
            print(f"🟢 {len(raw_frames)} frames obtenidos.")

            if len(raw_frames) < args.num_frames:
                print("⚠️ Insuficientes frames, esperando...")
                time.sleep(1)
                continue

            # Preprocesar
            proc = []
            for fr in raw_frames:
                fr = cv2.resize(fr, size)
                fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                t = torch.from_numpy(fr).permute(2, 0, 1).float() / 255.0
                proc.append(t)
            clip_tensor = torch.stack(proc)

            # Transform + X3D
            transformed = transform({"video": clip_tensor.float()})["video"]
            with torch.no_grad():
                feat = model_x3d(transformed.unsqueeze(0).to(device)).cpu().numpy()
            vector = torch.from_numpy(feat).to(device)

            # Custom inference
            with torch.no_grad():
                score, _ = model_custom(vector)
                score = torch.sigmoid(score).item()

            print(f"🔍 SCORE DE ANOMALÍA: {round(score, 4)}")

            # Envío
            payload = {"date": date, "camera_id": args.camera_id, "score": score}
            threading.Thread(target=enviar_score, args=(payload, args.endpoint), daemon=True).start()

            # ⏱ Esperar hasta que se cumpla el tiempo de ventana
            elapsed = time.time() - T1
            restante = intervalo_seg - elapsed
            if restante > 0:
                print(f"😴 Durmiendo {round(restante, 2)} s para sincronizar...")
                time.sleep(restante)
            else:
                print(f"⚠️ Procesamiento tardó más de lo estimado: {round(elapsed,2)} s")

    except KeyboardInterrupt:
        print("✋ Interrupción detectada. Finalizando...")
    finally:
        fb.stop()
        print("🛑 FrameBuffer detenido correctamente.")

if __name__ == '__main__':
    args = parse_args()
    main(args)