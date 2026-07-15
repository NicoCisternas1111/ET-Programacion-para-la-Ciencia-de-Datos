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

## Resultados — modelos supervisados

Los tres modelos usan un `Pipeline` de Scikit-learn (escalado + one-hot integrados), para que la validación cruzada no filtre información.

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC | Acc. train | Gap train-test | Sobreajuste |
|---|---|---|---|---|---|---|---|---|
| **Regresión Logística** | 0.847 | 0.813 | 0.777 | **0.795** | 0.928 | 0.855 | **0.009** | no detectado |
| Random Forest (GridSearchCV) | 0.845 | 0.830 | 0.746 | 0.786 | 0.919 | 0.958 | **0.113** | ⚠️ **posible** |
| Árbol de Decisión (GridSearchCV) | 0.823 | 0.788 | 0.732 | 0.759 | 0.895 | 0.849 | 0.027 | no detectado |

- **Modelo elegido:** Regresión Logística — mejor F1 **y** el que mejor generaliza.
- **Validación cruzada (5 folds):** F1 = **0.805 ± 0.004**
- **Sobreajuste:** el **Random Forest lo muestra**: acierta 95,8 % en entrenamiento pero solo 84,5 % en prueba (gap 0.113, sobre el umbral de 0.10). El bosque memoriza parte del entrenamiento. La Regresión Logística y el Árbol no: sus gaps son 0.009 y 0.027.
- Es un argumento adicional para la logística: no solo gana en F1, además es el único de los tres que combina el mejor rendimiento con la mejor generalización. El `min_samples_leaf=50` que GridSearchCV eligió para el árbol es precisamente lo que a él lo protege del sobreajuste.
- **Métrica de optimización:** F1, no accuracy — con 38 % de clase positiva, el accuracy sobrevalora al modelo. El error caro es el **falso negativo** (un retraso no anticipado).

**Nota metodológica (fuga de datos):** `tiempo_entrega_min` se excluye de las features porque solo se conoce *después* de entregar el pedido — usarla sería predecir el retraso con la respuesta. Hay un test automático que falla si alguien la reintroduce.

### ¿Aportaron las fuentes externas?

Se midió en vez de asumir. Reentrenando la línea base solo con variables operativas:

| Configuración | F1 |
|---|---|
| Solo variables operativas | 0.7945 |
| Con variables externas (SQL + API) | 0.7947 |
| **Diferencia** | **+0.0002** |

La ganancia es **nula**, y ninguna variable externa aparece en el top 12 de importancia. **El retraso se explica por las condiciones operativas del pedido, no por el territorio ni el clima de la comuna.** Se reporta el hallazgo tal cual: un modelo más complejo sin ganancia medible es complejidad injustificada. Las fuentes externas sí resultan útiles para la segmentación.

---

## Resultados — modelo no supervisado

Segmentación de las **14 comunas** con K-Means sobre los datos reales del Censo y la API.

| K | Inertia | Silhouette |
|---|---|---|
| **2** | 50.86 | **0.331** |
| 3 | 38.12 | 0.315 |
| 4 | 26.81 | 0.289 |
| 5 | 20.56 | 0.275 |
| 6 | 14.30 | 0.300 |

- **K elegido: 2**, por mayor silhouette y por dar segmentos interpretables. La inertia baja siempre al aumentar K, por eso no se usa como criterio de selección.
- **PCA (2 componentes):** resume el **69,6 %** de la varianza. *PCA no crea los clusters — los crea K-Means; PCA solo permite dibujarlos en 2D.*
- **Lectura honesta del silhouette:** 0.331 indica estructura **moderada**. Los grupos existen y son interpretables, pero no están fuertemente separados.

| Segmento | Comunas | Superficie media | Densidad media | Tasa de retraso |
|---|---|---|---|---|
| Periférica extensa | La Florida, Las Condes, Maipú, Peñalolén, Puente Alto | 89.6 km² | 4.577 hab/km² | 0.3832 |
| Céntrica compacta | Estación Central, Independencia, Macul, Providencia, Recoleta, San Miguel, Santiago Centro, Vitacura, Ñuñoa | 15.8 km² | 10.820 hab/km² | 0.3809 |

Ambos segmentos tienen **la misma tasa de retraso**: la segmentación es **territorial, no de desempeño**. Sirve para planificar flota (bicicletas en comunas compactas, moto/auto en las extensas), no para predecir retrasos — coherente con el hallazgo anterior.

**Las dos decisiones de diseño no se afirman, se miden.** `src/clustering.py` recalcula en cada ejecución el mejor silhouette de cada alternativa y lo registra en `reports/metricas_clustering.json`:

| Alternativa | Mejor silhouette | Veredicto |
|---|---|---|
| Segmentar los 51.975 **pedidos** | 0.126 | Descartada: estructura nula, los pedidos no forman grupos naturales |
| Comunas **incluyendo variables operativas** | 0.239 | Descartada: las operativas son homogéneas entre comunas y solo aportan ruido |
| **Comunas solo con variables externas** | **0.331** | ✅ Elegida |

Además, la comuna es la unidad de decisión del negocio: la flota se planifica por territorio, no pedido a pedido.

---

## Hallazgos principales

- El **tráfico alto** y la **lluvia** son los factores que más aumentan la probabilidad de retraso.
- La **distancia** en bicicleta penaliza mucho más que en moto o auto — es la palanca operativa de mayor impacto.
- Las **horas punta** (13–14 h almuerzo, 20–21 h cena) concentran los retrasos.
- Los repartidores con más **experiencia** reducen el riesgo de forma sostenida hasta los ~36 meses.
- La **comuna no influye** en el retraso: las 14 tienen tasas entre 0.36 y 0.40, diferencias atribuibles a ruido muestral.

---

## Estructura del proyecto

```
├── analisis_delivery.ipynb    # INFORME: importa src/, no lo duplica
├── data/
│   ├── pedidos_delivery.csv   # Fuente 1 (crudo)
│   ├── comunas.db             # Fuente 2 (SQLite)
│   ├── clima_api.json         # Fuente 3 (snapshot de respaldo de la API)
│   ├── pedidos_limpio.csv     # salida del ETL
│   └── comunas_segmentadas.csv# salida del clustering
├── src/
│   ├── generar_dataset.py     # generador del dataset simulado
│   ├── fuente_sql.py          # FUENTE 2: base SQLite + query
│   ├── fuente_api.py          # FUENTE 3: API REST + fallback
│   ├── etl.py                 # ETL: limpieza + integración + features
│   ├── modelo.py              # supervisado: LogReg / Árbol / RF + GridSearchCV
│   └── clustering.py          # no supervisado: K-Means + PCA
├── tests/
│   └── test_pipeline.py       # 12 tests (calidad, integración, anti-leakage)
├── models/
│   └── modelo_retrasos.joblib # Pipeline serializado (preproceso + modelo)
├── reports/                   # métricas, reporte de calidad y figuras
├── dashboard/app.py           # dashboard Dash + Plotly (3 pestañas)
├── DICCIONARIO_DATOS.md
└── requirements.txt
```

---

## Dashboard

`python dashboard/app.py` → <http://127.0.0.1:8050>

1. **Análisis** — KPIs, filtros por comuna/clima/vehículo y 4 visualizaciones interactivas.
2. **Segmentación** — resultado del K-Means: proyección PCA, método del codo, silhouette por K y perfil de cada segmento.
3. **Simulador** — predicción en vivo: se ajustan las condiciones del pedido y el modelo devuelve la probabilidad de retraso. El contexto de la comuna (densidad, clima) se completa automáticamente desde las fuentes SQL y API.

---

## Limitaciones

- El dataset de pedidos es **simulado**: las conclusiones validan la metodología, no describen el mercado real de delivery en Santiago.
- Como la comuna se asigna al azar en la simulación, **no existe señal territorial que capturar**; con datos reales las fuentes externas podrían sí aportar.
- La segmentación trabaja con 14 observaciones (todas las comunas de operación, no una muestra).
- No se abordó el desbalance de clases (38/62) con técnicas específicas (SMOTE, ajuste de umbral): es el siguiente paso natural.