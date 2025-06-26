from flask import Flask, Response
import cv2
import threading
import time
import os

VIDEO_ID = int(os.environ.get("VIDEO_ID", 1))  # Valor por defecto: 1
import os
app = Flask(__name__)



def generate_looped_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video en: {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30  # Valor predeterminado si FPS no disponible
        delay = 1.0 / fps

        while True:
            success, frame = cap.read()
            if not success:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reiniciar al inicio
                continue
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(delay)
    finally:
        cap.release()


# Endpoint para videos en bucle
@app.route('/loop')
def loop_video():
    try:
        video_name = f"video_{VIDEO_ID}.mp4"
        video_path = os.path.join("demo_videos", video_name)
        return Response(generate_looped_video(video_path),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    except FileNotFoundError as e:
        return str(e), 404



if __name__ == '__main__':

    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
