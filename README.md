# 🛵 Predicción de retrasos en pedidos de delivery

**Evaluación Final Transversal — Programación para la Ciencia de Datos (SCY1101)**
Integrantes: Nicolas Salas · Francisco Gaete · Nicolás Cisternas

Solución end-to-end de ciencia de datos: **tres fuentes de datos** integradas, pipeline ETL con validación, modelos supervisados y no supervisados, y dashboard interactivo en Dash.

---

## Problema

Las plataformas de delivery pierden clientes cuando los pedidos llegan tarde. El objetivo es **predecir, en el momento en que se crea el pedido, si llegará retrasado** (más de 50 minutos), para poder reasignar repartidores, ajustar la promesa de entrega o priorizar los pedidos en riesgo.

- **Unidad de análisis:** el pedido
- **Target:** `retrasado` (1 = superó los 50 min). Tasa base: **38,2 %**
- **Tipo de problema:** clasificación binaria

---

## Las tres fuentes de datos

| # | Tipo | Origen | Aporta | Cruce |
|---|---|---|---|---|
| 1 | **CSV** | `data/pedidos_delivery.csv` — simulado | 52.600 pedidos en 14 comunas | — |
| 2 | **SQL** | `data/comunas.db` (SQLite) — **datos reales** del Censo 2017 (INE) | Población, superficie, densidad, NSE, coordenadas | `comuna` |
| 3 | **API REST** | [Open-Meteo](https://open-meteo.com) — **datos reales**, sin API key | Temperatura, humedad, viento y precipitación por comuna | `comuna` |

La fuente API usa las coordenadas que entrega la fuente SQL, así que las tres quedan encadenadas. Si la API no responde, el módulo cae automáticamente al snapshot en `data/clima_api.json` y el pipeline no se bloquea.

> **Sobre los datos:** el CSV de pedidos es una **simulación académica** declarada explícitamente (ver notebook y `src/generar_dataset.py`). Se generó con señal realista más ruido controlado, y se ensució a propósito con nulos, duplicados, outliers y texto inconsistente para justificar un ETL real. Las fuentes 2 y 3 sí contienen datos reales.

---

## Arquitectura

```
FUENTE 1 (CSV)      FUENTE 2 (SQL)        FUENTE 3 (API REST)
pedidos_delivery    comunas.db            Open-Meteo
52.600 pedidos      Censo 2017 INE        clima por comuna
       |                   |                      |
       +---------+---------+----------+-----------+
                           |
                     src/etl.py
        limpieza -> integración (merge validado) -> features
                           |
                  data/pedidos_limpio.csv
                           |
          +----------------+-----------------+
          |                                  |
   src/modelo.py                     src/clustering.py
   SUPERVISADO                       NO SUPERVISADO
   LogReg / Árbol / RF               K-Means + PCA
   GridSearchCV + CV 5-fold          codo + silhouette
          |                                  |
          +----------------+-----------------+
                           |
                  dashboard/app.py
              Dash + Plotly (3 pestañas)
```

---

## Cómo ejecutar

```bash
pip install -r requirements.txt

python src/generar_dataset.py   # Fuente 1: genera el CSV crudo
python src/fuente_sql.py        # Fuente 2: crea la base SQLite
python src/fuente_api.py        # Fuente 3: refresca el snapshot de la API
python src/etl.py               # Limpieza + integración + feature engineering
python src/modelo.py            # Modelos supervisados + GridSearchCV  (~95 s)
python src/clustering.py        # Segmentación K-Means + PCA           (~9 s)
python dashboard/app.py         # Dashboard -> http://127.0.0.1:8050

pytest -v                       # 12 tests del pipeline (~3 s)
```