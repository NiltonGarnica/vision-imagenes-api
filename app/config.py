"""Configuración del proyecto.

Todos los valores se leen de variables de entorno (o del archivo `.env`)
para no tener que tocar el código al cambiar de equipo.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Carpeta raíz del proyecto (la que contiene `app/`).
RAIZ_PROYECTO = Path(__file__).resolve().parent.parent

# Carga el archivo .env si existe. Las variables ya definidas en el sistema tienen prioridad.
load_dotenv(RAIZ_PROYECTO / ".env")


def _texto(nombre: str, por_defecto: str = "") -> str:
    return os.getenv(nombre, por_defecto).strip()


# --- Tesseract (OCR) ---------------------------------------------------------
# Ruta completa a tesseract.exe. Vacío = buscarlo en el PATH o en la ruta típica de Windows.
TESSERACT_CMD = _texto("TESSERACT_CMD")
# Carpeta con los archivos *.traineddata. Vacío = la carpeta tessdata de la instalación.
TESSDATA_DIR = _texto("TESSDATA_DIR")
# Idiomas para el OCR. "spa+eng" lee español e inglés.
TESSERACT_IDIOMA = _texto("TESSERACT_IDIOMA", "spa+eng")
# Modo de segmentación de página (PSM). 11 = texto disperso, útil en fotos y letreros.
TESSERACT_PSM = int(_texto("TESSERACT_PSM", "11"))

# --- YOLO (detección de objetos) ---------------------------------------------
# Nombre o ruta de los pesos. Si el archivo no existe, Ultralytics lo descarga.
# Las rutas relativas se toman desde la raíz del proyecto.
YOLO_MODELO = str(RAIZ_PROYECTO / _texto("YOLO_MODELO", "modelos/yolo11n.pt"))
# "cpu" por defecto. Para usar la GPU: "0" o "cuda:0" (requiere PyTorch con CUDA).
YOLO_DISPOSITIVO = _texto("YOLO_DISPOSITIVO", "cpu")
# Confianza mínima (0 a 1) para aceptar una detección.
YOLO_CONFIANZA_MINIMA = float(_texto("YOLO_CONFIANZA_MINIMA", "0.25"))
