import cv2
import threading
import time
from collections import deque

class FrameBuffer:
    def __init__(self, url, maxlen=64):
        self.cap = cv2.VideoCapture(url)
        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir stream: {url}")
        self.buffer = deque(maxlen=maxlen)
        self.stopped = False
        self.thread = threading.Thread(target=self._reader, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _reader(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            self.buffer.append(frame)

    def read_recent(self, num_frames, stride):
        if len(self.buffer) < num_frames * stride:
            raise RuntimeError("No hay suficientes frames en buffer aún.")
        buf = list(self.buffer)
        # Tomar cada stride desde el final
        frames = [buf[-i*stride] for i in range(1, num_frames+1)]
        return list(reversed(frames))

    def stop(self):
        self.stopped = True
        self.thread.join()
        self.cap.release()