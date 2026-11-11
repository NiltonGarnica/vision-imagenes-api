"""Genera ejemplos/texto_la_paz.png: texto grande en negro sobre fondo blanco.

Uso:  python ejemplos/crear_imagen_texto.py
"""

from pathlib import Path

import cv2
import numpy as np


def crear_imagen_texto(texto: str = "LA PAZ") -> np.ndarray:
    imagen = np.full((300, 900, 3), 255, dtype=np.uint8)
    cv2.putText(imagen, texto, (60, 200), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 12, cv2.LINE_AA)
    return imagen


if __name__ == "__main__":
    destino = Path(__file__).parent / "texto_la_paz.png"
    cv2.imwrite(str(destino), crear_imagen_texto())
    print(f"Imagen creada: {destino}")
