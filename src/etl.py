"""
Pipeline ETL — Pedidos de delivery. Integra TRES fuentes de datos.

EXTRACT
    Fuente 1 (CSV)  : data/pedidos_delivery.csv        -> 52.600 pedidos
    Fuente 2 (SQL)  : data/comunas.db (SQLite)         -> datos demográficos
    Fuente 3 (API)  : Open-Meteo REST / clima_api.json -> clima por comuna

TRANSFORM
    Limpieza  : duplicados, categorías inconsistentes, outliers, nulos
    Integración: merge de las 3 fuentes por 'comuna' (con validación)
    Enriquecimiento: variables derivadas (feature engineering)

LOAD
    data/pedidos_limpio.csv + reports/reporte_calidad.json

Uso:
    python src/etl.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fuente_api import leer_clima
from fuente_sql import leer_comunas

RUTA_ENTRADA = "data/pedidos_delivery.csv"
RUTA_SALIDA = "data/pedidos_limpio.csv"
RUTA_REPORTE = "reports/reporte_calidad.json"

CATEGORICAS_TEXTO = ["clima", "trafico", "tipo_vehiculo", "tipo_comercio", "comuna"]
NUMERICAS_A_IMPUTAR = ["distancia_km", "tiempo_preparacion_min",
                       "calificacion_repartidor", "valor_pedido_clp"]


# ============================ EXTRACT ============================

def extract() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Lee las tres fuentes y las devuelve sin transformar."""
    print("EXTRACT")
    pedidos = pd.read_csv(RUTA_ENTRADA)
    print(f"   Fuente 1 (CSV)  -> {len(pedidos):,} pedidos")

    comunas = leer_comunas()
    print(f"   Fuente 2 (SQL)  -> {len(comunas)} comunas desde SQLite")

    clima = leer_clima()
    print(f"   Fuente 3 (API)  -> {len(clima)} registros meteorológicos")

    return pedidos, comunas, clima


# =========================== TRANSFORM ===========================

def limpiar(df: pd.DataFrame, reporte: dict) -> pd.DataFrame:
    """Limpieza de calidad: duplicados, texto, outliers y nulos."""
    reporte["filas_entrada"] = len(df)

    # 1. Duplicados
    dups = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    reporte["duplicados_eliminados"] = dups

    # 2. Normalizar categorías (espacios y mayúsculas inconsistentes)
    for col in CATEGORICAS_TEXTO:
        df[col] = df[col].astype("string").str.strip().str.capitalize()
    df["comuna"] = df["comuna"].str.title()

    # 3. Outliers imposibles
    #    - distancias > 40 km no existen en delivery urbano -> nulo e imputar
    #    - tiempos de entrega negativos -> fila inválida, se elimina
    outliers_dist = int((df["distancia_km"] > 40).sum())
    df.loc[df["distancia_km"] > 40, "distancia_km"] = np.nan
    filas_invalidas = int((df["tiempo_entrega_min"] <= 0).sum())
    df = df[df["tiempo_entrega_min"] > 0].reset_index(drop=True)
    reporte["outliers_distancia_corregidos"] = outliers_dist
    reporte["filas_tiempo_invalido_eliminadas"] = filas_invalidas

    # 4. Imputación de nulos: mediana en numéricas, moda en categóricas
    nulos_antes = int(df.isna().sum().sum())
    for col in NUMERICAS_A_IMPUTAR:
        df[col] = df[col].fillna(df[col].median())
    df["clima"] = df["clima"].fillna(df["clima"].mode()[0])
    reporte["nulos_imputados"] = nulos_antes
    reporte["nulos_restantes"] = int(df.isna().sum().sum())

    return df


def integrar(pedidos: pd.DataFrame, comunas: pd.DataFrame,
             clima: pd.DataFrame, reporte: dict) -> pd.DataFrame:
    """
    Une las tres fuentes por 'comuna' validando que no se pierdan pedidos.

    Se usa merge 'left' para conservar todos los pedidos aunque una comuna
    no tuviera referencia externa, y se verifica explícitamente que el
    cruce haya sido completo.
    """
    filas_antes = len(pedidos)

    df = pedidos.merge(comunas, on="comuna", how="left", validate="many_to_one")
    df = df.merge(clima, on="comuna", how="left", validate="many_to_one")

    # Validación del cruce: ningún pedido puede quedar sin datos externos
    sin_sql = int(df["poblacion"].isna().sum())
    sin_api = int(df["temperatura_c"].isna().sum())
    if sin_sql or sin_api:
        faltantes = sorted(df.loc[df["poblacion"].isna(), "comuna"].unique())
        raise ValueError(
            f"Cruce incompleto: {sin_sql} pedidos sin datos SQL y {sin_api} "
            f"sin datos API. Comunas sin match: {faltantes}"
        )
    if len(df) != filas_antes:
        raise ValueError(
            f"El merge alteró el número de filas: {filas_antes} -> {len(df)}"
        )

    reporte["integracion"] = {
        "pedidos_cruzados": len(df),
        "comunas_sql_unidas": int(df["comuna"].nunique()),
        "columnas_desde_sql": ["poblacion", "superficie_km2",
                               "densidad_hab_km2", "nse", "latitud", "longitud"],
        "columnas_desde_api": ["temperatura_c", "humedad_pct",
                               "viento_kmh", "precipitacion_mm"],
        "pedidos_sin_match": sin_sql + sin_api,
    }
    return df


def enriquecer(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering: variables derivadas del negocio."""
    df["es_hora_punta"] = df["hora_pedido"].isin([13, 14, 20, 21]).astype(int)
    df["es_fin_de_semana"] = (df["dia_semana"] >= 6).astype(int)
    df["valor_por_item"] = (df["valor_pedido_clp"] / df["items_pedido"]).round(0)
    df["repartidor_novato"] = (df["experiencia_repartidor_meses"] < 6).astype(int)
    # Derivada del cruce con la fuente SQL: comunas densas = más congestión
    df["comuna_alta_densidad"] = (
        df["densidad_hab_km2"] > df["densidad_hab_km2"].median()
    ).astype(int)
    return df


def transform(pedidos: pd.DataFrame, comunas: pd.DataFrame,
              clima: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    print("TRANSFORM")
    reporte: dict = {}

    df = limpiar(pedidos, reporte)
    print(f"   Limpieza    -> {reporte['duplicados_eliminados']} duplicados, "
          f"{reporte['nulos_imputados']} nulos imputados, "
          f"{reporte['outliers_distancia_corregidos']} outliers")

    df = integrar(df, comunas, clima, reporte)
    print(f"   Integración -> {len(df):,} pedidos cruzados con SQL + API")

    df = enriquecer(df)
    print("   Enriquecimiento -> 5 variables derivadas")

    reporte["filas_salida"] = len(df)
    reporte["columnas_salida"] = len(df.columns)
    reporte["tasa_retraso"] = round(float(df["retrasado"].mean()), 4)
    return df, reporte


# ============================= LOAD ==============================

def load(df: pd.DataFrame, reporte: dict) -> None:
    print("LOAD")
    Path(RUTA_SALIDA).parent.mkdir(parents=True, exist_ok=True)
    Path(RUTA_REPORTE).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RUTA_SALIDA, index=False)
    with open(RUTA_REPORTE, "w", encoding="utf-8") as f:
        json.dump(reporte, f, indent=2, ensure_ascii=False)
    print(f"   -> {RUTA_SALIDA} ({len(df):,} filas x {len(df.columns)} columnas)")
    print(f"   -> {RUTA_REPORTE}")


def main() -> None:
    pedidos, comunas, clima = extract()
    df, reporte = transform(pedidos, comunas, clima)
    load(df, reporte)
    print("\n" + json.dumps(reporte, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
