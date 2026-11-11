"""Servicio de OCR: lee texto de una imagen con Tesseract (pytesseract)."""

import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytesseract

from app import config

# Ruta donde el instalador de Windows (UB Mannheim) deja Tesseract por defecto.
RUTA_WINDOWS_POR_DEFECTO = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")


class OCRNoDisponibleError(RuntimeError):
    """Tesseract no está instalado, no se encuentra o le falta un idioma."""


def _configurar_tesseract() -> None:
    """Indica a pytesseract qué ejecutable usar."""
    if config.TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD
    elif shutil.which("tesseract") is None and RUTA_WINDOWS_POR_DEFECTO.exists():
        pytesseract.pytesseract.tesseract_cmd = str(RUTA_WINDOWS_POR_DEFECTO)

    # Carpeta de idiomas propia. Se usa la variable de entorno porque admite rutas con espacios.
    if config.TESSDATA_DIR:
        os.environ["TESSDATA_PREFIX"] = config.TESSDATA_DIR


_configurar_tesseract()


def _opciones_tesseract() -> str:
    return f"--psm {config.TESSERACT_PSM}"


def leer_texto(imagen: np.ndarray) -> str:
    """Devuelve el texto de la imagen en una sola línea, o "" si no hay texto.

    Acepta imágenes en color (BGR) o en escala de grises.
    """
    if imagen.size == 0:
        return ""

    if imagen.ndim == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

    try:
        texto = pytesseract.image_to_string(imagen, lang=config.TESSERACT_IDIOMA, config=_opciones_tesseract())
    except pytesseract.TesseractNotFoundError as error:
        raise OCRNoDisponibleError(
            "No se encontró Tesseract. Instálalo o define TESSERACT_CMD con la ruta a tesseract.exe."
        ) from error
    except pytesseract.TesseractError as error:
        raise OCRNoDisponibleError(
            f"Tesseract falló. Revisa que estén instalados los idiomas '{config.TESSERACT_IDIOMA}'. "
            f"Detalle: {error.message}"
        ) from error

    # Une líneas y espacios repetidos: "LA\n\nPAZ " -> "LA PAZ".
    return " ".join(texto.split())
