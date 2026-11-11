"""Servicio de YOLO: detecta objetos con un modelo preentrenado de Ultralytics.

El modelo se carga la primera vez que se usa y luego se reutiliza en
todas las peticiones (cargarlo tarda unos segundos).
"""

import logging
import threading
from pathlib import Path

import numpy as np

from app import config

logger = logging.getLogger(__name__)

_modelo = None
_candado = threading.Lock()  # evita que dos peticiones carguen el modelo a la vez


def elegir_dispositivo() -> str:
    """Devuelve el dispositivo configurado, o "cpu" si se pidió GPU y no hay CUDA."""
    dispositivo = config.YOLO_DISPOSITIVO.lower()
    if dispositivo == "cpu":
        return "cpu"

    import torch

    if not torch.cuda.is_available():
        logger.warning("Se pidió YOLO_DISPOSITIVO=%s pero PyTorch no tiene CUDA. Se usará la CPU.", dispositivo)
        return "cpu"
    return dispositivo


def obtener_modelo():
    """Carga el modelo YOLO una sola vez (carga perezosa) y lo devuelve."""
    global _modelo
    if _modelo is None:
        with _candado:
            if _modelo is None:
                from ultralytics import YOLO  # importación lenta: solo cuando hace falta

                ruta = Path(config.YOLO_MODELO)
                ruta.parent.mkdir(parents=True, exist_ok=True)
                logger.info("Cargando modelo YOLO desde %s", ruta)
                _modelo = YOLO(str(ruta))  # descarga los pesos si el archivo no existe
    return _modelo


def detectar_objetos(imagen: np.ndarray) -> list[dict]:
    """Detecta objetos en una imagen BGR.

    Devuelve una lista de diccionarios con `objeto`, `confianza` (0 a 1)
    y `coordenadas` [x1, y1, x2, y2] en píxeles de la imagen original.
    """
    modelo = obtener_modelo()
    resultado = modelo.predict(
        imagen,
        device=elegir_dispositivo(),
        conf=config.YOLO_CONFIANZA_MINIMA,
        verbose=False,
    )[0]

    alto, ancho = imagen.shape[:2]
    detecciones = []
    for caja in resultado.boxes:
        x1, y1, x2, y2 = caja.xyxy[0].tolist()
        coordenadas = [
            max(0, min(ancho, round(x1))),
            max(0, min(alto, round(y1))),
            max(0, min(ancho, round(x2))),
            max(0, min(alto, round(y2))),
        ]
        detecciones.append(
            {
                "objeto": resultado.names[int(caja.cls[0])],
                "confianza": round(float(caja.conf[0]), 2),
                "coordenadas": coordenadas,
            }
        )
    return detecciones
