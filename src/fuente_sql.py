"""
FUENTE 2 (SQL) — Base de datos SQLite con datos de referencia de comunas.

Construye data/comunas.db a partir de datos públicos aproximados del
Censo 2017 (INE) y los deja disponibles para que el ETL los consulte
mediante una query SQL, tal como se haría contra una base corporativa.

Los datos de población y superficie son valores públicos aproximados;
el nivel socioeconómico dominante es una clasificación referencial.

Uso:
    python src/fuente_sql.py          # crea/recrea la base
Salida:
    data/comunas.db  (tabla: comunas)
"""

import sqlite3
from pathlib import Path

import pandas as pd

RUTA_DB = "data/comunas.db"

# comuna, población (Censo 2017), superficie km², NSE dominante, lat, lon
COMUNAS = [
    ("Santiago Centro",  404495,  22.4, "C2",   -33.4489, -70.6693),
    ("Providencia",      142079,  14.4, "ABC1", -33.4314, -70.6093),
    ("Las Condes",       294838,  99.4, "ABC1", -33.4103, -70.5680),
    ("Ñuñoa",            208237,  16.9, "ABC1", -33.4569, -70.5975),
    ("Maipú",            521627, 135.5, "C3",   -33.5110, -70.7580),
    ("La Florida",       366916,  70.2, "C3",   -33.5326, -70.5990),
    ("Puente Alto",      568106,  88.2, "D",    -33.6110, -70.5756),
    ("San Miguel",       107954,   9.5, "C2",   -33.4969, -70.6520),
    ("Estación Central", 147041,  14.1, "C3",   -33.4610, -70.6970),
    ("Recoleta",         157851,  16.2, "C3",   -33.4100, -70.6400),
    ("Independencia",    100281,   7.4, "C3",   -33.4150, -70.6650),
    ("Macul",            116534,  12.9, "C2",   -33.4900, -70.5980),
    ("Peñalolén",        241599,  54.9, "C3",   -33.4880, -70.5420),
    ("Vitacura",          85384,  28.3, "ABC1", -33.3800, -70.5800),
]

DDL = """
DROP TABLE IF EXISTS comunas;
CREATE TABLE comunas (
    comuna          TEXT PRIMARY KEY,
    poblacion       INTEGER NOT NULL,
    superficie_km2  REAL    NOT NULL,
    nse             TEXT    NOT NULL,
    latitud         REAL    NOT NULL,
    longitud        REAL    NOT NULL
);
"""

# El ETL consume la base a través de esta query: la densidad se calcula
# en SQL, no en Pandas, para mostrar el uso real del motor.
QUERY_ETL = """
SELECT
    comuna,
    poblacion,
    superficie_km2,
    ROUND(poblacion / superficie_km2, 1) AS densidad_hab_km2,
    nse,
    latitud,
    longitud
FROM comunas
ORDER BY comuna;
"""


def crear_base(ruta: str = RUTA_DB) -> None:
    """Crea la base SQLite y carga la tabla de comunas."""
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(ruta) as con:
        con.executescript(DDL)
        con.executemany(
            "INSERT INTO comunas "
            "(comuna, poblacion, superficie_km2, nse, latitud, longitud) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            COMUNAS,
        )
    print(f"OK -> {ruta} | tabla 'comunas' con {len(COMUNAS)} registros")


def leer_comunas(ruta: str = RUTA_DB) -> pd.DataFrame:
    """Extract de la fuente SQL: ejecuta la query y devuelve un DataFrame."""
    if not Path(ruta).exists():
        crear_base(ruta)
    with sqlite3.connect(ruta) as con:
        return pd.read_sql_query(QUERY_ETL, con)


def main() -> None:
    crear_base()
    df = leer_comunas()
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
