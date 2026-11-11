"""Servicio de OpenCV: escala de grises y mejora de contraste."""

import cv2
import numpy as np


def mejorar_imagen(imagen: np.ndarray) -> np.ndarray:
    """Convierte la imagen a escala de grises y mejora su contraste con CLAHE.

    CLAHE (ecualización adaptativa del histograma) mejora el contraste por zonas,
    así un letrero oscuro dentro de una foto clara también se vuelve legible.
    """
    if imagen.size == 0:
        return imagen

    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY) if imagen.ndim == 3 else imagen
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gris)


def codificar_png(imagen: np.ndarray) -> bytes:
    """Convierte una matriz de OpenCV en los bytes de un archivo PNG."""
    exito, buffer = cv2.imencode(".png", imagen)
    if not exito:
        raise ValueError("No se pudo codificar la imagen como PNG.")
    return buffer.tobytes()
