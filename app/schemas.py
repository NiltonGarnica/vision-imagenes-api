"""Modelos Pydantic que describen las respuestas JSON de la API."""

from pydantic import BaseModel, Field


class RespuestaOCR(BaseModel):
    texto: str = Field(description="Texto reconocido. Cadena vacía si no se encontró texto.")

    model_config = {"json_schema_extra": {"examples": [{"texto": "LA PAZ"}]}}


class Deteccion(BaseModel):
    objeto: str = Field(description="Clase detectada por YOLO, por ejemplo 'bus'.")
    confianza: float = Field(ge=0, le=1, description="Confianza de la detección, entre 0 y 1.")
    coordenadas: list[int] = Field(
        min_length=4,
        max_length=4,
        description="Caja en píxeles en el orden [x1, y1, x2, y2].",
    )


class RespuestaYOLO(BaseModel):
    detecciones: list[Deteccion]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"detecciones": [{"objeto": "bus", "confianza": 0.94, "coordenadas": [120, 80, 640, 450]}]}
            ]
        }
    }


class DeteccionConTexto(Deteccion):
    texto: str = Field(description="Texto leído por OCR dentro de la región detectada.")


class RespuestaAnalisis(BaseModel):
    detecciones: list[DeteccionConTexto]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "detecciones": [
                        {
                            "objeto": "bus",
                            "confianza": 0.94,
                            "coordenadas": [120, 80, 640, 450],
                            "texto": "LA PAZ",
                        }
                    ]
                }
            ]
        }
    }


class RespuestaError(BaseModel):
    detail: str
