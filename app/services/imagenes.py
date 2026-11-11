"""Utilidades compartidas: convertir los bytes recibidos en una imagen de OpenCV."""

import cv2
import numpy as np

FIRMA_PNG = b"\x89PNG\r\n\x1a\n"
FIRMA_JPEG = b"\xff\xd8\xff"


class ImagenInvalidaError(ValueError):
    """El archivo recibido está vacío o no es una imagen JPEG/PNG válida."""


def decodificar_imagen(contenido: bytes) -> np.ndarray:
    """Convierte los bytes de un JPEG o PNG en una matriz BGR (formato de OpenCV)."""
    if not contenido:
        raise ImagenInvalidaError("El archivo está vacío. Envía una imagen JPEG o PNG.")

    if not (contenido.startswith(FIRMA_PNG) or contenido.startswith(FIRMA_JPEG)):
        raise ImagenInvalidaError("El archivo no es una imagen JPEG o PNG.")

    datos = np.frombuffer(contenido, dtype=np.uint8)
    imagen = cv2.imdecode(datos, cv2.IMREAD_COLOR)
    if imagen is None:
        raise ImagenInvalidaError("No se pudo decodificar la imagen. El archivo puede estar dañado.")
    return imagen
