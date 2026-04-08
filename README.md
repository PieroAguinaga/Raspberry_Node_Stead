# STEAD — Spatiotemporal Anomaly Detection System

A real-time video anomaly detection system designed for surveillance cameras.
STEAD uses a two-stage deep learning pipeline to continuously monitor video
streams and report anomaly scores to a configurable HTTP endpoint.

---

## Overview

STEAD processes video streams frame by frame, extracts spatiotemporal features
using a pre-trained X3D model, and classifies each sliding window of frames
with a custom transformer-based model. Each detection result is asynchronously
reported as a JSON payload, making it easy to integrate with any monitoring
backend.

The system is designed to run on edge devices (Raspberry Pi) as well as
standard hardware, with model variants scaled accordingly.

---

## Architecture

```
Video Stream (HTTP/MJPEG)
        │
        ▼
┌───────────────────┐
│   FrameBuffer     │  Thread-safe circular buffer
│  (background)     │  Decouples capture from inference
└────────┬──────────┘
         │  Sliding window (N frames, stride S)
         ▼
┌───────────────────┐
│   X3D Feature     │  Pre-trained spatiotemporal CNN
│   Extractor       │  Output: [192-dim] feature vector
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  STEAD Anomaly    │  Custom transformer (Performer attention)
│  Classifier       │  Output: score ∈ [0, 1]
└────────┬──────────┘
         │  Async HTTP POST
         ▼
   External Endpoint
```

---

## Tech Stack

| Layer              | Technology                          |
|--------------------|-------------------------------------|
| Deep Learning      | PyTorch 2.0.1 + TorchVision 0.15.2  |
| Video Processing   | OpenCV                              |
| Feature Extraction | PyTorchVideo (X3D: xs / s / m / l)  |
| Anomaly Classifier | Custom Performer-based Transformer  |
| Streaming Server   | Flask                               |
| Numerics           | NumPy, Pandas                       |

---

## How It Works

1. **Frame Capture** — A `FrameBuffer` reads frames from the camera stream in
   a background thread, maintaining a circular buffer of up to 200+ frames.

2. **Sliding Window** — The main loop reads overlapping windows of `N` frames
   with a configurable stride `S`, enabling continuous monitoring without gaps.

3. **Feature Extraction** — Each window is preprocessed (normalize, scale,
   crop) and passed through an X3D model, producing a 192-dimensional feature
   vector.

4. **Anomaly Scoring** — The STEAD model processes the feature vector through
   Performer attention layers and returns a sigmoid-activated score between 0
   and 1 (>0.5 indicates an anomaly).

5. **Async Reporting** — Results are HTTP-POSTed to an external endpoint
   without blocking the inference loop.

---

## Model Variants

### X3D Feature Extractors (Facebook Research / PyTorchVideo)

| Variant | Input Size | Use Case                    |
|---------|------------|-----------------------------|
| xs      | 182×182    | Edge devices / Raspberry Pi |
| s       | 182×182    | Balanced                    |
| m       | 256×256    | Higher accuracy             |
| l       | 320×320    | Maximum accuracy            |

### STEAD Anomaly Classifiers

| Variant | Size    | Use Case                |
|---------|---------|-------------------------|
| base    | ~6.6 MB | Full accuracy           |
| fast    | ~96 KB  | Optimized for edge/RPi  |
| tiny    | < 96 KB | Minimum footprint       |

Nine pre-trained model combinations are included in `models/`.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/PieroAguinaga/Raspberry_Node_Stead.git
cd Raspberry_Node_Stead

# Install dependencies
pip install torch==2.0.1 torchvision==0.15.2
pip install pytorchvideo opencv-python flask numpy pandas tqdm matplotlib
```

---

## Usage

### Real-Time Stream Inference

```bash
python main.py \
  --video 1 \
  --model_name STEAD_BASE_XS_8_5final \
  --x3d_version xs \
  --num_frames 8 \
  --stride 5 \
  --arch base \
  --camera_id 1 \
  --endpoint http://<your-server>/anomaly
```

### Batch Video Inference (with visualization)

```bash
python final_inferencia.py \
  --video demo_videos/video_1.mp4 \
  --model_name STEAD_BASE_XS_8_5final \
  --x3d_version xs \
  --num_frames 8 \
  --stride 5 \
  --arch base
```

### CLI Arguments

| Argument        | Description                                       |
|-----------------|---------------------------------------------------|
| `--video`       | Video source: `0`=webcam, `1`=demo, or file path  |
| `--model_name`  | Name of the `.pkl` model file in `models/`        |
| `--x3d_version` | X3D size: `xs`, `s`, `m`, or `l`                 |
| `--num_frames`  | Temporal window size (recommended: 8–16)          |
| `--stride`      | Frame skip rate (lower = more overlap)            |
| `--arch`        | STEAD variant: `base`, `fast`, or `tiny`          |
| `--camera_id`   | Camera identifier included in result payload      |
| `--endpoint`    | HTTP endpoint to POST detection results           |

---

## Output Payload

Each detection window produces a JSON payload sent to the configured endpoint:

```json
{
  "date": "2024-01-15T14:32:01.123Z",
  "camera_id": 1,
  "score": 0.73,
  "window_id": 42,
  "anomaly_detected": true,
  "fps": 30,
  "window_size_frames": 8,
  "duration_sec": 0.27
}
```

---

## Project Structure

```
├── main.py               # Real-time inference loop (entry point)
├── app.py                # Flask server for camera simulation
├── buffer.py             # Thread-safe FrameBuffer implementation
├── model.py              # STEAD model architecture (base/fast/tiny)
├── load_x3d.py           # X3D model loader with transforms
├── load_custom_model.py  # STEAD model loader from .pkl files
├── option.py             # CLI argument parser
├── utils.py              # FeedForward and decoupled conv blocks
├── final_inferencia.py   # Batch inference with CSV + visualization
├── simluacion.py         # Camera simulator subprocess launcher
├── models/               # Pre-trained STEAD model weights (.pkl)
└── demo_videos/          # Sample test videos
```

---

## Hardware Targets

| Target              | Recommended Config              |
|---------------------|---------------------------------|
| Raspberry Pi 4      | X3D `xs` + STEAD `fast`         |
| Standard server/VM  | X3D `s` or `m` + STEAD `base`   |
| Development machine | Any variant, CPU inference      |

---

## Based On

This project uses a modified version of the model introduced in:

> **STEAD: Spatio-Temporal Efficient Anomaly Detection for Time and Compute Sensitive Applications**
> Andrew Gao, Jun Liu — 2025
> [https://arxiv.org/abs/2503.07942](https://arxiv.org/abs/2503.07942)

The original STEAD architecture achieves 91.34% AUC on the UCF-Crime benchmark using (2+1)D convolutions and Performer linear attention. This repository adapts and extends that model for real-time edge deployment on surveillance camera streams.

---

## License

MIT
