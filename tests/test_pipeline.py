"""
Tests del pipeline de datos.

Se ejecutan en segundos: usan DataFrames pequeños construidos a mano en vez
del dataset completo, para que la CI dé feedback rápido en cada push.

    pytest -v
"""

import numpy as np
import pandas as pd
import pytest

import clustering
import etl
import modelo
from fuente_sql import COMUNAS, crear_base, leer_comunas


# ============================ Fuente SQL ============================

def test_base_sqlite_se_crea_y_tiene_las_14_comunas(tmp_path):
    ruta = tmp_path / "comunas.db"
    crear_base(str(ruta))
    df = leer_comunas(str(ruta))
    assert len(df) == 14
    assert df["comuna"].is_unique


def test_densidad_se_calcula_en_la_query_sql(tmp_path):
    ruta = tmp_path / "comunas.db"
    crear_base(str(ruta))
    df = leer_comunas(str(ruta))
    esperado = round(df.loc[0, "poblacion"] / df.loc[0, "superficie_km2"], 1)
    assert df.loc[0, "densidad_hab_km2"] == pytest.approx(esperado, abs=0.1)
    assert (df["densidad_hab_km2"] > 0).all()


# ========================== Limpieza (ETL) ==========================

def _pedidos_sucios() -> pd.DataFrame:
    """Dataset mínimo con los 4 problemas de calidad que el ETL debe resolver."""
    base = {
        "distancia_km": 3.0, "tiempo_preparacion_min": 10.0, "items_pedido": 2,
        "valor_pedido_clp": 10000.0, "hora_pedido": 13, "dia_semana": 6,
        "comuna": "Providencia", "clima": "Despejado", "trafico": "Bajo",
        "tipo_vehiculo": "Moto", "tipo_comercio": "Restaurante",
        "experiencia_repartidor_meses": 12, "calificacion_repartidor": 4.5,
        "pedidos_activos_repartidor": 2, "tiempo_entrega_min": 30.0,
        "retrasado": 0,
    }
    filas = [
        base,
        dict(base),                                   # duplicado exacto
        {**base, "trafico": "alto", "clima": " Lluvia "},  # categorías sucias
        {**base, "distancia_km": 150.0},              # outlier imposible
        {**base, "tiempo_entrega_min": -1.0},         # fila inválida
        {**base, "distancia_km": np.nan, "clima": np.nan},  # nulos
    ]
    return pd.DataFrame(filas)


def test_limpieza_elimina_duplicados():
    """
    De las 6 filas de entrada se elimina 1 duplicado exacto y 1 fila con
    tiempo inválido, quedando 4.

    Nota: no se exige que el resultado quede sin duplicados. La
    deduplicación ocurre ANTES de la imputación, y al rellenar nulos con la
    mediana una fila puede volverse idéntica a otra. Es el comportamiento
    esperado del orden del pipeline, no un defecto.
    """
    reporte = {}
    df = etl.limpiar(_pedidos_sucios(), reporte)
    assert reporte["duplicados_eliminados"] == 1
    assert len(df) == 4


def test_limpieza_normaliza_categorias_inconsistentes():
    df = etl.limpiar(_pedidos_sucios(), {})
    # "alto" -> "Alto" y " Lluvia " -> "Lluvia"
    assert set(df["trafico"].unique()) <= {"Bajo", "Medio", "Alto"}
    assert set(df["clima"].unique()) <= {"Despejado", "Nublado", "Lluvia"}


def test_limpieza_corrige_outliers_y_filas_invalidas():
    reporte = {}
    df = etl.limpiar(_pedidos_sucios(), reporte)
    assert reporte["outliers_distancia_corregidos"] == 1
    assert reporte["filas_tiempo_invalido_eliminadas"] == 1
    assert (df["distancia_km"] <= 40).all()
    assert (df["tiempo_entrega_min"] > 0).all()


def test_limpieza_no_deja_nulos():
    reporte = {}
    df = etl.limpiar(_pedidos_sucios(), reporte)
    assert reporte["nulos_restantes"] == 0
    assert not df.isna().any().any()


# ========================= Integración (ETL) =========================

def test_integracion_une_las_tres_fuentes_sin_perder_pedidos():
    pedidos = etl.limpiar(_pedidos_sucios(), {})
    comunas = leer_comunas()
    clima = pd.DataFrame([{"comuna": c[0], "temperatura_c": 12.0,
                           "humedad_pct": 70, "viento_kmh": 2.0,
                           "precipitacion_mm": 0.0} for c in COMUNAS])
    reporte = {}
    df = etl.integrar(pedidos, comunas, clima, reporte)

    assert len(df) == len(pedidos)          # el merge no duplica ni pierde filas
    assert reporte["integracion"]["pedidos_sin_match"] == 0
    assert "densidad_hab_km2" in df.columns  # llegó de la fuente SQL
    assert "temperatura_c" in df.columns     # llegó de la fuente API


def test_integracion_falla_si_una_comuna_no_cruza():
    """El ETL debe caerse ruidosamente, no producir nulos en silencio."""
    pedidos = etl.limpiar(_pedidos_sucios(), {})
    pedidos.loc[0, "comuna"] = "Comuna Inexistente"
    comunas = leer_comunas()
    clima = pd.DataFrame([{"comuna": c[0], "temperatura_c": 12.0,
                           "humedad_pct": 70, "viento_kmh": 2.0,
                           "precipitacion_mm": 0.0} for c in COMUNAS])

    with pytest.raises(ValueError, match="Cruce incompleto"):
        etl.integrar(pedidos, comunas, clima, {})


# ======================= Feature engineering =======================

def test_enriquecer_crea_las_variables_derivadas():
    pedidos = etl.limpiar(_pedidos_sucios(), {})
    comunas = leer_comunas()
    clima = pd.DataFrame([{"comuna": c[0], "temperatura_c": 12.0,
                           "humedad_pct": 70, "viento_kmh": 2.0,
                           "precipitacion_mm": 0.0} for c in COMUNAS])
    df = etl.enriquecer(etl.integrar(pedidos, comunas, clima, {}))

    for col in ["es_hora_punta", "es_fin_de_semana", "valor_por_item",
                "repartidor_novato", "comuna_alta_densidad"]:
        assert col in df.columns
    assert df["es_hora_punta"].iloc[0] == 1      # hora 13 es punta
    assert df["es_fin_de_semana"].iloc[0] == 1   # día 6 es sábado
    assert df["valor_por_item"].iloc[0] == 5000  # 10000 / 2 ítems


# ==================== Guardas metodológicas ====================

def test_el_modelo_no_usa_informacion_posterior_al_evento():
    """tiempo_entrega_min determina el target: usarlo sería fuga de datos."""
    features = modelo.NUMERICAS + modelo.CATEGORICAS
    assert "tiempo_entrega_min" not in features
    assert modelo.TARGET not in features
    assert "id_pedido" not in features


def test_el_clustering_no_usa_la_etiqueta():
    """En no supervisado la etiqueta no existe; solo sirve para interpretar."""
    assert "retrasado" not in clustering.VARIABLES
    assert "tasa_retraso" not in clustering.VARIABLES
    assert "tiempo_entrega_min" not in clustering.VARIABLES


def test_el_k_elegido_esta_dentro_del_rango_evaluado():
    assert clustering.K_ELEGIDO in clustering.RANGO_K
