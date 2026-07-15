"""
FUENTE 3 (API REST) — Condiciones meteorológicas por comuna (Open-Meteo).

Consume la API pública de Open-Meteo (https://open-meteo.com), que es
gratuita y no requiere API key, para obtener temperatura, humedad y
viento en las coordenadas de cada comuna. Las coordenadas provienen de
la FUENTE SQL (data/comunas.db), de modo que las tres fuentes quedan
encadenadas.

Estrategia de resiliencia: si la API no responde (sin red durante la
presentación), se reutiliza el último snapshot guardado en
data/clima_api.json. El pipeline nunca queda bloqueado por la red.

Uso:
    python src/fuente_api.py          # refresca el snapshot desde la API
Salida:
    data/clima_api.json
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

from fuente_sql import leer_comunas

RUTA_SNAPSHOT = "data/clima_api.json"
URL_API = "https://api.open-meteo.com/v1/forecast"
TIMEOUT_S = 20


def _construir_url(comunas: pd.DataFrame) -> dict:
    """Arma los parámetros para consultar las 14 comunas en una sola llamada."""
    return {
        "latitude": ",".join(str(v) for v in comunas["latitud"]),
        "longitude": ",".join(str(v) for v in comunas["longitud"]),
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation",
        "timezone": "America/Santiago",
    }


def extraer_desde_api() -> pd.DataFrame:
    """Extract real contra la API REST. Lanza excepción si falla."""
    comunas = leer_comunas()
    resp = requests.get(URL_API, params=_construir_url(comunas), timeout=TIMEOUT_S)
    resp.raise_for_status()
    datos = resp.json()

    # Con múltiples coordenadas la API devuelve una lista; con una sola, un dict.
    if isinstance(datos, dict):
        datos = [datos]

    filas = []
    for comuna, punto in zip(comunas["comuna"], datos):
        actual = punto["current"]
        filas.append({
            "comuna": comuna,
            "temperatura_c": actual["temperature_2m"],
            "humedad_pct": actual["relative_humidity_2m"],
            "viento_kmh": actual["wind_speed_10m"],
            "precipitacion_mm": actual["precipitation"],
        })
    return pd.DataFrame(filas)


def guardar_snapshot(df: pd.DataFrame, ruta: str = RUTA_SNAPSHOT) -> None:
    """Load de la fuente API: deja el JSON versionado como respaldo."""
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fuente": "Open-Meteo API (https://open-meteo.com)",
        "extraido_en": datetime.now().isoformat(timespec="seconds"),
        "registros": df.to_dict(orient="records"),
    }
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def leer_clima(ruta: str = RUTA_SNAPSHOT, refrescar: bool = False) -> pd.DataFrame:
    """
    Devuelve el clima por comuna.

    refrescar=True fuerza la llamada a la API. En modo normal se intenta
    la API y, si falla, se cae al snapshot local para no romper el ETL.
    """
    if not refrescar and Path(ruta).exists():
        with open(ruta, encoding="utf-8") as f:
            return pd.DataFrame(json.load(f)["registros"])

    try:
        df = extraer_desde_api()
        guardar_snapshot(df, ruta)
        print(f"   API OK -> {len(df)} comunas consultadas en Open-Meteo")
        return df
    except Exception as e:  # sin red, timeout, API caída
        if Path(ruta).exists():
            print(f"   API no disponible ({type(e).__name__}); uso snapshot local")
            with open(ruta, encoding="utf-8") as f:
                return pd.DataFrame(json.load(f)["registros"])
        raise RuntimeError(
            f"La API falló ({e}) y no existe snapshot en {ruta}. "
            "Ejecuta 'python src/fuente_api.py' con conexión al menos una vez."
        ) from e


def main() -> None:
    df = leer_clima(refrescar=True)
    print(df.to_string(index=False))
    print(f"\nOK -> {RUTA_SNAPSHOT}")


if __name__ == "__main__":
    main()
