import torch
from load_custom_model import load_custom_model
from load_x3d import load_x3d
import numpy as np
import time
import requests
import json
from datetime import datetime
from simluacion import start_camera
from buffer import FrameBuffer  # Usa la nueva versión con read_next_window
from option import parse_args
import cv2
import threading
import os
from datetime import datetime

def enviar_score(payload, endpoint):
    try:
        print(f"📤 [Async] Enviando datos al endpoint: {endpoint}")
        r = requests.post(endpoint, json=payload, headers={"Content-Type": "application/json"})
        print(f"✅ [Async] Respuesta del servidor: Código {r.status_code}")
    except Exception as e:
        print(f"❌ [Async] Error al enviar: {e}")

def main(args):
    print("🚀 Iniciando sistema de detección de anomalías...")

    print(f"🟡 Lanzando simulación de cámara con video ID = {args.video}")
    start_camera(args.video)
    window_id = 0 
    ip_cam = "http://localhost:5000/loop"
    print(f"🌐 Dirección del stream: {ip_cam}")


    buffer_size = args.num_frames * args.stride * 10
    fb = FrameBuffer(ip_cam, maxlen=buffer_size).start()
    print(f"✅ FrameBuffer iniciado con tamaño máximo = {buffer_size}")
    

    # Cargar modelos
    device = torch.device("cpu")
    model_x3d, transform, params = load_x3d(args.x3d_version, device, args.num_frames, args.stride)
    model_custom = load_custom_model(args.model_name, args.arch, device)
    size = (params["side_size"], params["side_size"])

    fps = 30  # Puedes hacerlo variable si tu cámara no es 30 FPS
    intervalo_seg = (args.num_frames * args.stride) / fps
    print(f"⏲️ Ventana de inferencia cada {intervalo_seg:.2f} segundos")

    try:
        while True:
            T1 = time.time()


            print("\n📸 Leyendo ventana siguiente del buffer (sliding window)...")
            #Pooling pasivo
            try:
                raw_frames = fb.read_next_window(args.num_frames, args.stride)
            except RuntimeError as e:
                print(f"⏳ Esperando frames: {e}")
                time.sleep(0.1)
                continue
            date = datetime.now().isoformat()
            print(f"🟢 {len(raw_frames)} frames obtenidos.")
            os.makedirs("debug_frames", exist_ok=True)
            
            for idx, fr in enumerate(raw_frames):
                cv2.imwrite(f"debug_frames/window{window_id}_frame{idx}.jpg", fr)
                print(f"🖼️ Guardado frame {idx} de ventana {window_id}")

            # Preprocesar
            clip_tensor = torch.stack([
                torch.from_numpy(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()
                 for fr in raw_frames
                    ])



            # Extraer características
            transformed = transform({"video": clip_tensor.float()})["video"]
            with torch.no_grad():
                feat = model_x3d(transformed.unsqueeze(0).to(device)).cpu().numpy()
            vector = torch.from_numpy(feat).to(device)

            # Inferencia
            with torch.no_grad():
                score, _ = model_custom(vector)
                score = torch.sigmoid(score).item()

            print(window_id)
            print(f"🔍 SCORE DE ANOMALÍA: {round(score, 4)}")

           

            start_frame_index = fb.read_ptr - args.num_frames * args.stride
            # Tiempo en segundos desde que inició el buffer
            offset_seconds = start_frame_index / fps
            # Convertimos a ISO usando el tiempo de inicio del buffer
            start_time_iso = datetime.fromtimestamp(fb.start_time + offset_seconds).isoformat()
                        # Enviar score
            payload = {
                "date": start_time_iso,
                "camera_id": args.camera_id,
                "score": score,
                "window_id": window_id,
                "start_frame_index": fb.read_ptr - args.num_frames * args.stride,
                "last_frame_index": fb.read_ptr,
                "fps": fps,
                "window_size_frames": args.num_frames,
                "stride": args.stride,
                "duration_sec": intervalo_seg,
                "anomaly_detected": score > 0.5,  
                "window_resolution": f"{size[0]}x{size[1]}",
                "source_video_id": args.video
            }
            
            window_id += 1
            threading.Thread(target=enviar_score, args=(payload, args.endpoint), daemon=True).start()



    except KeyboardInterrupt:
        print("✋ Interrupción detectada. Finalizando...")
    finally:
        fb.stop()
        print("🛑 FrameBuffer detenido correctamente.")

if __name__ == '__main__':
    args = parse_args()
    main(args)
#python main.py --video 1 --model_name STEAD_BASE_XS_8_5final  --x3d_version xs --num_frames 8 --stride 5 --arch base --camera_id 1 --endpoint http://192.168.1.24:8080/anomaly