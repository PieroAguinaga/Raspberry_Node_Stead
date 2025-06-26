import torch
from model import Model

def load_custom_model(model_custom_name: str, arch: str, device: torch.device):
    """
    Carga un modelo personalizado de anomalía desde un archivo .pkl.

    Parámetros:
    - model_custom_name: Nombre del archivo del modelo sin la ruta (ej. "MTFL-205").
    - arch: Arquitectura del modelo ('base', 'fast', 'tiny').
    - device: Dispositivo donde se cargará el modelo (cpu o cuda).

    Retorna:
    - model_custom: Modelo cargado listo para inferencia.
    """
    model_path = f'./models/{model_custom_name}.pkl'
    print(f"\n📦 Cargando modelo de anomalía desde: {model_path}")

    if arch == 'base':
        model_custom = Model()
    elif arch in ['fast', 'tiny']:
        model_custom = Model(ff_mult=1, dims=(32, 32), depths=(1, 1))
    else:
        raise ValueError("Arquitectura desconocida. Usa 'base', 'fast' o 'tiny'.")

    model_custom.load_state_dict(torch.load(model_path, map_location=device))
    model_custom.eval().to(device)
    
    return model_custom