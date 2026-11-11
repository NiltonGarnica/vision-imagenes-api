"""API para probar OCR, OpenCV y YOLO por separado y en conjunto.

Arranque:  python -m uvicorn app.main:app --reload
Swagger:   http://127.0.0.1:8000/docs
"""

from fastapi import FastAPI, File, HTTPException, Response, UploadFile

from app.schemas import DeteccionConTexto, RespuestaAnalisis, RespuestaError, RespuestaOCR, RespuestaYOLO
from app.services.imagenes import ImagenInvalidaError, decodificar_imagen
from app.services.ocr_service import OCRNoDisponibleError, leer_texto
from app.services.opencv_service import codificar_png, mejorar_imagen
from app.services.yolo_service import detectar_objetos

app = FastAPI(
    title="API de visión por computadora",
    description="Sube una imagen JPEG o PNG en el campo `imagen` para probar OCR, OpenCV y YOLO.",
    version="1.0.0",
)

ERRORES_COMUNES = {
    400: {"model": RespuestaError, "description": "Archivo vacío o imagen inválida."},
}


def leer_imagen_subida(imagen: UploadFile):
    """Lee el archivo subido y lo convierte en imagen de OpenCV, o responde 400."""
    try:
        return decodificar_imagen(imagen.file.read())
    except ImagenInvalidaError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


def ejecutar_ocr(imagen) -> str:
    """Llama al OCR y convierte la falta de Tesseract en un error 503 claro."""
    try:
        return leer_texto(imagen)
    except OCRNoDisponibleError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post(
    "/ocr",
    response_model=RespuestaOCR,
    responses={**ERRORES_COMUNES, 503: {"model": RespuestaError, "description": "Tesseract no disponible."}},
    summary="Leer texto con Tesseract",
)
def ruta_ocr(imagen: UploadFile = File(..., description="Imagen JPEG o PNG")):
    imagen_cv = leer_imagen_subida(imagen)
    return RespuestaOCR(texto=ejecutar_ocr(imagen_cv))


@app.post(
    "/opencv",
    response_class=Response,
    responses={
        200: {"content": {"image/png": {}}, "description": "Imagen en escala de grises con más contraste."},
        **ERRORES_COMUNES,
    },
    summary="Escala de grises y contraste con OpenCV",
)
def ruta_opencv(imagen: UploadFile = File(..., description="Imagen JPEG o PNG")):
    imagen_cv = leer_imagen_subida(imagen)
    procesada = mejorar_imagen(imagen_cv)
    return Response(content=codificar_png(procesada), media_type="image/png")


@app.post(
    "/yolo",
    response_model=RespuestaYOLO,
    responses=ERRORES_COMUNES,
    summary="Detectar objetos con YOLO",
)
def ruta_yolo(imagen: UploadFile = File(..., description="Imagen JPEG o PNG")):
    imagen_cv = leer_imagen_subida(imagen)
    return RespuestaYOLO(detecciones=detectar_objetos(imagen_cv))


@app.post(
    "/analizar",
    response_model=RespuestaAnalisis,
    responses={**ERRORES_COMUNES, 503: {"model": RespuestaError, "description": "Tesseract no disponible."}},
    summary="YOLO + OpenCV + OCR en una sola petición",
)
def ruta_analizar(imagen: UploadFile = File(..., description="Imagen JPEG o PNG")):
    """Detecta objetos y lee el texto que hay dentro de cada uno.

    Pasos: 1) YOLO sobre la imagen original, 2) recortar cada caja,
    3) mejorar el recorte con OpenCV, 4) leer el texto con OCR.
    """
    imagen_cv = leer_imagen_subida(imagen)

    resultados = []
    for deteccion in detectar_objetos(imagen_cv):  # 1) la misma función que /yolo
        x1, y1, x2, y2 = deteccion["coordenadas"]
        recorte = imagen_cv[y1:y2, x1:x2]  # 2) filas = y, columnas = x
        recorte_mejorado = mejorar_imagen(recorte)  # 3) la misma función que /opencv
        texto = ejecutar_ocr(recorte_mejorado)  # 4) la misma función que /ocr
        resultados.append(DeteccionConTexto(**deteccion, texto=texto))

    return RespuestaAnalisis(detecciones=resultados)
