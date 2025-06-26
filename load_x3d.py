from torch import nn
import torch
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample
)
from torchvision.transforms.v2 import CenterCrop, Normalize
from torchvision.transforms import Compose, Lambda




def load_x3d(x3d_version: str, device, num_frames, stride):
    print(f"\n🔧 Cargando modelo X3D versión: {x3d_version} en dispositivo: {device}")
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
    print(f"✅ Modelo X3D {model_name} cargado.")

    if hasattr(model_x3d, "blocks"):
        print(f"ℹ️ Eliminando última capa del modelo...")
        model_x3d.blocks = model_x3d.blocks[:-1]

    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]

    model_transform_params = {
        "x3d_xs": {"side_size": 182, "crop_size": 182, "num_frames": num_frames, "sampling_rate": stride},
        "x3d_s":  {"side_size": 182, "crop_size": 182, "num_frames": num_frames, "sampling_rate": stride},
        "x3d_m":  {"side_size": 256, "crop_size": 256, "num_frames": num_frames, "sampling_rate": stride},
        "x3d_l":  {"side_size": 320, "crop_size": 320, "num_frames": num_frames, "sampling_rate": stride}
    }

    transform_params = model_transform_params[model_name]
    print(f"🌀 Transform Params: {transform_params}")

    class Permute(nn.Module):
        def __init__(self, dims): super().__init__(); self.dims = dims
        def forward(self, x): return torch.permute(x, self.dims)

    transform = ApplyTransformToKey(
        key="video",
        transform=Compose([
            Permute((3, 0, 1, 2)),
            UniformTemporalSubsample(transform_params["num_frames"]),
            Lambda(lambda x: x / 255.0),
            Permute((1, 0, 2, 3)),
            Normalize(mean, std),
            ShortSideScale(size=transform_params["side_size"]),
            CenterCrop((transform_params["crop_size"], transform_params["crop_size"])),
            Permute((1, 0, 2, 3))
        ])
    )


    return model_x3d, transform, transform_params