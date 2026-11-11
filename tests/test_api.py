"""Pruebas automáticas de las cuatro rutas.

Uso:  python -m pytest -v
"""

from pathlib import Path

import cv2
import numpy as np
import ultralytics
from fastapi.testclient import TestClient

from app.main import app
from ejemplos.crear_imagen_texto import crear_imagen_texto

cliente = TestClient(app)

# Foto de ejemplo que viene incluida en el paquete de Ultralytics.
RUTA_BUS = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"


def a_png(imagen: np.ndarray) -> bytes:
    return cv2.imencode(".png", imagen)[1].tobytes()


def enviar(ruta: str, contenido: bytes, nombre: str = "imagen.png", tipo: str = "image/png"):
    return cliente.post(ruta, files={"imagen": (nombre, contenido, tipo)})


# --- Validación de la imagen (igual en todas las rutas) ----------------------


def test_archivo_vacio_devuelve_400():
    respuesta = enviar("/ocr", b"")
    assert respuesta.status_code == 400
    assert "vacío" in respuesta.json()["detail"]


def test_archivo_que_no_es_imagen_devuelve_400():
    respuesta = enviar("/ocr", b"esto no es una imagen", "nota.txt", "text/plain")
    assert respuesta.status_code == 400


def test_png_danado_devuelve_400():
    respuesta = enviar("/ocr", b"\x89PNG\r\n\x1a\n" + b"basura")
    assert respuesta.status_code == 400


# --- /ocr --------------------------------------------------------------------


def test_ocr_lee_texto_grande():
    respuesta = enviar("/ocr", a_png(crear_imagen_texto("LA PAZ")))
    assert respuesta.status_code == 200
    assert respuesta.json() == {"texto": "LA PAZ"}


def test_ocr_sin_texto_devuelve_cadena_vacia():
    blanca = np.full((200, 200, 3), 255, dtype=np.uint8)
    respuesta = enviar("/ocr", a_png(blanca))
    assert respuesta.status_code == 200
    assert respuesta.json() == {"texto": ""}


# --- /opencv -----------------------------------------------------------------


def test_opencv_devuelve_png_en_escala_de_grises():
    rng = np.random.default_rng(0)
    color = rng.integers(80, 160, size=(120, 160, 3), dtype=np.uint8)  # imagen de poco contraste
    respuesta = enviar("/opencv", a_png(color))

    assert respuesta.status_code == 200
    assert respuesta.headers["content-type"] == "image/png"
    procesada = cv2.imdecode(np.frombuffer(respuesta.content, np.uint8), cv2.IMREAD_UNCHANGED)
    assert procesada.ndim == 2  # un solo canal = escala de grises
    assert procesada.shape == (120, 160)
    gris_original = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    assert procesada.std() > gris_original.std()  # más contraste


def test_opencv_acepta_jpeg():
    jpeg = cv2.imencode(".jpg", crear_imagen_texto())[1].tobytes()
    respuesta = enviar("/opencv", jpeg, "foto.jpg", "image/jpeg")
    assert respuesta.status_code == 200
    assert respuesta.content.startswith(b"\x89PNG")


# --- /yolo -------------------------------------------------------------------


def test_yolo_detecta_el_bus_de_ejemplo():
    respuesta = enviar("/yolo", RUTA_BUS.read_bytes(), "bus.jpg", "image/jpeg")
    assert respuesta.status_code == 200

    detecciones = respuesta.json()["detecciones"]
    objetos = {d["objeto"] for d in detecciones}
    assert "bus" in objetos
    for d in detecciones:
        assert 0 <= d["confianza"] <= 1
        x1, y1, x2, y2 = d["coordenadas"]
        assert 0 <= x1 < x2 and 0 <= y1 < y2


def test_yolo_sin_objetos_devuelve_lista_vacia():
    blanca = np.full((320, 320, 3), 255, dtype=np.uint8)
    respuesta = enviar("/yolo", a_png(blanca))
    assert respuesta.status_code == 200
    assert respuesta.json() == {"detecciones": []}


# --- /analizar ---------------------------------------------------------------


def test_analizar_agrega_texto_a_cada_deteccion():
    respuesta = enviar("/analizar", RUTA_BUS.read_bytes(), "bus.jpg", "image/jpeg")
    assert respuesta.status_code == 200

    detecciones = respuesta.json()["detecciones"]
    assert any(d["objeto"] == "bus" for d in detecciones)
    for d in detecciones:
        assert set(d) == {"objeto", "confianza", "coordenadas", "texto"}
        assert isinstance(d["texto"], str)


def test_analizar_sin_objetos_devuelve_lista_vacia():
    blanca = np.full((320, 320, 3), 255, dtype=np.uint8)
    respuesta = enviar("/analizar", a_png(blanca))
    assert respuesta.status_code == 200
    assert respuesta.json() == {"detecciones": []}


def test_analizar_imagen_invalida_devuelve_400():
    respuesta = enviar("/analizar", b"")
    assert respuesta.status_code == 400
