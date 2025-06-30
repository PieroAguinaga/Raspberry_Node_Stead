import cv2
import threading
import time
from collections import deque

class FrameBuffer:
    def __init__(self, url, maxlen=256):
        self.cap = cv2.VideoCapture(url)
        if not self.cap.isOpened():
            raise RuntimeError(f"❌ No se pudo abrir el stream: {url}")
        
        self.buffer = deque(maxlen=maxlen)
        self.lock = threading.Lock()
        self.read_ptr = 0  # Índice lógico de lectura
        self.frame_index = 0  # Índice total de frames leídos
        self.stopped = False
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.start_time = None  # ⏱️ Tiempo de inicio en UNIX

    def start(self):
        self.start_time = time.time()  # Marca el tiempo de inicio de la captura
        self.thread.start()
        return self

    def _reader(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.frame_index = 0
                self.read_ptr = 0
                self.buffer.clear()
                continue
            with self.lock:
                self.buffer.append(frame)
                self.frame_index += 1
            time.sleep(1 / 30)  # Simula 30 FPS

    def read_next_window(self, num_frames, stride):
        with self.lock:
            total_needed = (num_frames - 1) * stride + 1
            available = self.frame_index - self.read_ptr

            if available < total_needed:
                raise RuntimeError("⚠️ No hay suficientes frames aún para la siguiente ventana.")

            offset = self.frame_index - len(self.buffer)
            buffer_start = self.read_ptr - offset

            if buffer_start < 0:
                print("⚠️ read_ptr está apuntando a frames que ya no están en el buffer. Reiniciando puntero...")
                self.read_ptr = self.frame_index - len(self.buffer)
                buffer_start = 0

            if (buffer_start + (num_frames - 1) * stride) >= len(self.buffer):
                raise RuntimeError("⚠️ Los frames requeridos aún no han sido llenados en el buffer.")

            frames = [self.buffer[buffer_start + i * stride] for i in range(num_frames)]
            self.read_ptr += num_frames * stride

        return frames

    def get_timestamp_for_frame(self, frame_index, fps=30):
        if self.start_time is None:
            raise RuntimeError("⚠️ El buffer no ha sido iniciado con start().")
        return self.start_time + (frame_index / fps)

    def get_last_timestamp(self, fps=30):
        return self.get_timestamp_for_frame(self.read_ptr, fps)

    def stop(self):
        self.stopped = True
        self.thread.join()
        self.cap.release()
