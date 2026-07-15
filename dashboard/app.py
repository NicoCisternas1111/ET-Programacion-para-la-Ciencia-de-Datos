"""
Dashboard interactivo — Retrasos en delivery (Dash + Plotly).

Tres pestañas:
    1. Análisis      : KPIs, filtros y visualizaciones exploratorias
    2. Segmentación  : resultado del K-Means + PCA sobre las comunas
    3. Simulador     : predicción en vivo con el modelo entrenado

Uso:
    python dashboard/app.py
    -> abrir http://127.0.0.1:8050

Requiere haber ejecutado antes el pipeline completo:
    python src/generar_dataset.py
    python src/etl.py
    python src/modelo.py
    python src/clustering.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, dcc, html

# ================= Rutas =================
# Acepta los artefactos generados por los scripts de src/ (data/, models/)
# o por el notebook ejecutado en Colab (raíz del proyecto).
RAIZ = Path(__file__).resolve().parent.parent


def _buscar(*rutas):
    for r in rutas:
        p = RAIZ / r
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No se encontró ninguno de: {rutas}. "
        "Ejecuta el pipeline (ver README) antes de levantar el dashboard."
    )


# ================= Carga de datos y modelo =================
df = pd.read_csv(_buscar("data/pedidos_limpio.csv", "pedidos_limpio.csv"))
modelo = joblib.load(_buscar("models/modelo_retrasos.joblib",
                             "modelo_retrasos.joblib"))

with open(_buscar("reports/metricas.json", "metricas.json"), encoding="utf-8") as f:
    METRICAS = json.load(f)
MEJOR = METRICAS["mejor_modelo"]
F1 = METRICAS["resultados"][MEJOR]["f1"]

# Segmentación: si aún no se corrió clustering.py, la pestaña lo avisa.
try:
    COMUNAS_SEG = pd.read_csv(_buscar("data/comunas_segmentadas.csv"))
    with open(_buscar("reports/metricas_clustering.json"), encoding="utf-8") as f:
        MET_CLUSTER = json.load(f)
except FileNotFoundError:
    COMUNAS_SEG, MET_CLUSTER = None, None

DIAS = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]

# Contexto externo por comuna (fuentes SQL y API). El modelo espera estas
# columnas, así que el simulador las resuelve a partir de la comuna elegida.
CONTEXTO_COMUNA = (
    df.groupby("comuna")
      .agg(densidad_hab_km2=("densidad_hab_km2", "first"),
           comuna_alta_densidad=("comuna_alta_densidad", "first"),
           temperatura_c=("temperatura_c", "first"),
           humedad_pct=("humedad_pct", "first"),
           viento_kmh=("viento_kmh", "first"),
           nse=("nse", "first"))
      .to_dict(orient="index")
)

app = Dash(__name__, title="Retrasos en Delivery")
server = app.server  # expuesto para gunicorn en Docker

TARJETA = {"flex": "1", "background": "#f7f7f7", "borderRadius": "10px",
           "padding": "16px", "textAlign": "center"}


def kpi(id_, etiqueta):
    return html.Div(
        [html.H2(id=id_, style={"margin": "0"}),
         html.P(etiqueta, style={"margin": "0", "color": "#666"})],
        style=TARJETA,
    )


# ================= Pestaña 1: Análisis =================
tab_analisis = html.Div([
    html.Div(style={"display": "flex", "gap": "12px", "margin": "16px 0"}, children=[
        dcc.Dropdown(sorted(df["comuna"].unique()), multi=True, id="f-comuna",
                     placeholder="Comuna", style={"flex": "1"}),
        dcc.Dropdown(sorted(df["clima"].unique()), multi=True, id="f-clima",
                     placeholder="Clima", style={"flex": "1"}),
        dcc.Dropdown(sorted(df["tipo_vehiculo"].unique()), multi=True, id="f-vehiculo",
                     placeholder="Vehículo", style={"flex": "1"}),
    ]),
    html.Div(style={"display": "flex", "gap": "12px", "marginBottom": "20px"}, children=[
        kpi("kpi-pedidos", "Pedidos"),
        kpi("kpi-retraso", "Tasa de retraso"),
        kpi("kpi-tiempo", "Tiempo medio de entrega"),
        html.Div([html.H2(f"{F1:.3f}", style={"margin": "0"}),
                  html.P(f"F1 del modelo ({MEJOR})",
                         style={"margin": "0", "color": "#666"})],
                 style=TARJETA),
    ]),
    html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr",
                    "gap": "12px"}, children=[
        dcc.Graph(id="g-hora"),
        dcc.Graph(id="g-comuna"),
        dcc.Graph(id="g-heatmap"),
        dcc.Graph(id="g-dist"),
    ]),
])


# ================= Pestaña 2: Segmentación =================
def construir_tab_segmentacion():
    if COMUNAS_SEG is None:
        return html.Div([
            html.H3("Segmentación no disponible"),
            html.P("Ejecuta 'python src/clustering.py' para generar los "
                   "resultados del K-Means y luego recarga esta página."),
        ], style={"padding": "40px"})

    k = MET_CLUSTER["k_elegido"]
    sil = MET_CLUSTER["silhouette_k_elegido"]
    var_pca = MET_CLUSTER["pca_varianza_explicada"]["total"]

    fig_pca = px.scatter(
        COMUNAS_SEG, x="PC1", y="PC2", color="segmento", text="comuna",
        size="poblacion", size_max=38,
        title=f"Segmentación de comunas (K={k}) proyectada con PCA",
        labels={"PC1": f"PC1 ({MET_CLUSTER['pca_varianza_explicada']['PC1']:.1%})",
                "PC2": f"PC2 ({MET_CLUSTER['pca_varianza_explicada']['PC2']:.1%})",
                "segmento": "Segmento"},
    )
    fig_pca.update_traces(textposition="top center")

    tabla_k = pd.DataFrame(MET_CLUSTER["comparacion_k"])
    fig_k = px.line(tabla_k, x="k", y="silhouette", markers=True,
                    title="Silhouette por K — se elige el máximo",
                    labels={"k": "Número de clusters (K)",
                            "silhouette": "Silhouette Score"})
    fig_k.add_vline(x=k, line_dash="dash", line_color="red")

    fig_codo = px.line(tabla_k, x="k", y="inertia", markers=True,
                       title="Método del codo",
                       labels={"k": "Número de clusters (K)",
                               "inertia": "Inertia"})
    fig_codo.add_vline(x=k, line_dash="dash", line_color="red")

    resumen = (COMUNAS_SEG.groupby("segmento")
               .agg(comunas=("comuna", "count"),
                    poblacion_media=("poblacion", "mean"),
                    superficie_media=("superficie_km2", "mean"),
                    densidad_media=("densidad_hab_km2", "mean"),
                    tasa_retraso=("tasa_retraso", "mean"))
               .round(3).reset_index())

    return html.Div([
        html.Div(style={"display": "flex", "gap": "12px", "margin": "16px 0"},
                 children=[
            html.Div([html.H2(f"K = {k}", style={"margin": "0"}),
                      html.P("Clusters elegidos", style={"margin": "0", "color": "#666"})],
                     style=TARJETA),
            html.Div([html.H2(f"{sil:.3f}", style={"margin": "0"}),
                      html.P("Silhouette Score", style={"margin": "0", "color": "#666"})],
                     style=TARJETA),
            html.Div([html.H2(f"{var_pca:.1%}", style={"margin": "0"}),
                      html.P("Varianza explicada por PCA (2D)",
                             style={"margin": "0", "color": "#666"})],
                     style=TARJETA),
        ]),
        dcc.Graph(figure=fig_pca),
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr",
                        "gap": "12px"}, children=[
            dcc.Graph(figure=fig_codo),
            dcc.Graph(figure=fig_k),
        ]),
        html.H3("Perfil de cada segmento"),
        dcc.Graph(figure=px.bar(
            resumen, x="segmento", y="densidad_media", color="segmento",
            title="Densidad media por segmento (hab/km²)",
            labels={"densidad_media": "hab/km²", "segmento": "Segmento"})),
        html.Div([
            html.P("Lectura: K-Means creó los grupos; PCA solo permite "
                   "dibujarlos en 2D. Las variables operativas (tasa de "
                   "retraso, distancia, tiempo de entrega) resultaron "
                   "homogéneas entre comunas y se excluyeron del clustering: "
                   "la segmentación es territorial, no de desempeño.",
                   style={"color": "#555", "fontStyle": "italic"}),
        ], style={"padding": "12px", "background": "#f7f7f7",
                  "borderRadius": "10px", "marginTop": "12px"}),
    ])


# ================= Pestaña 3: Simulador =================
tab_simulador = html.Div([
    html.H3("¿Llegará retrasado este pedido?"),
    html.P("El modelo predice con la información disponible al crear el "
           "pedido. Las condiciones de la comuna (densidad, clima actual) "
           "se completan automáticamente desde las fuentes SQL y API.",
           style={"color": "#666"}),
    html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr",
                    "gap": "20px", "marginTop": "16px"}, children=[
        html.Div([
            html.Label("Distancia (km)"),
            dcc.Slider(0.5, 20, 0.5, value=4, id="s-dist",
                       marks={i: str(i) for i in range(0, 21, 5)}),
            html.Label("Preparación (min)"),
            dcc.Slider(3, 45, 1, value=14, id="s-prep",
                       marks={i: str(i) for i in range(5, 46, 10)}),
            html.Label("Ítems"),
            dcc.Slider(1, 15, 1, value=3, id="s-items",
                       marks={i: str(i) for i in range(1, 16, 2)}),
        ]),
        html.Div([
            html.Label("Hora del pedido"),
            dcc.Slider(9, 23, 1, value=20, id="s-hora",
                       marks={i: str(i) for i in range(9, 24, 2)}),
            html.Label("Día de la semana"),
            dcc.Dropdown([{"label": d, "value": i + 1} for i, d in enumerate(DIAS)],
                         value=5, id="s-dia", clearable=False),
            html.Label("Tráfico"),
            dcc.Dropdown(["Bajo", "Medio", "Alto"], "Medio", id="s-trafico",
                         clearable=False),
            html.Label("Clima"),
            dcc.Dropdown(["Despejado", "Nublado", "Lluvia"], "Despejado",
                         id="s-clima", clearable=False),
        ]),
        html.Div([
            html.Label("Comuna"),
            dcc.Dropdown(sorted(df["comuna"].unique()), "Providencia",
                         id="s-comuna", clearable=False),
            html.Label("Vehículo"),
            dcc.Dropdown(["Bicicleta", "Moto", "Auto"], "Moto", id="s-vehiculo",
                         clearable=False),
            html.Label("Comercio"),
            dcc.Dropdown(sorted(df["tipo_comercio"].unique()), "Restaurante",
                         id="s-comercio", clearable=False),
            html.Label("Experiencia repartidor (meses)"),
            dcc.Slider(0, 60, 6, value=12, id="s-exp",
                       marks={i: str(i) for i in range(0, 61, 12)}),
            html.Label("Pedidos activos del repartidor"),
            dcc.Slider(1, 4, 1, value=2, id="s-activos"),
        ]),
    ]),
    html.Div(id="resultado-pred", style={"marginTop": "20px", "padding": "18px",
                                         "borderRadius": "10px",
                                         "fontSize": "20px",
                                         "textAlign": "center"}),
])


# ================= Layout =================
app.layout = html.Div(style={"fontFamily": "Arial", "maxWidth": "1200px",
                             "margin": "auto", "padding": "20px"}, children=[
    html.H1("🛵 Análisis y predicción de retrasos en delivery"),
    html.P("Pedidos en Santiago — 3 fuentes de datos (CSV + SQLite + API REST), "
           "pipeline ETL, modelos supervisado y no supervisado.",
           style={"color": "#666"}),
    dcc.Tabs([
        dcc.Tab(label="Análisis", children=tab_analisis),
        dcc.Tab(label="Segmentación (K-Means + PCA)",
                children=construir_tab_segmentacion()),
        dcc.Tab(label="Simulador", children=tab_simulador),
    ]),
])


# ================= Callbacks =================
@app.callback(
    Output("kpi-pedidos", "children"),
    Output("kpi-retraso", "children"),
    Output("kpi-tiempo", "children"),
    Output("g-hora", "figure"),
    Output("g-comuna", "figure"),
    Output("g-heatmap", "figure"),
    Output("g-dist", "figure"),
    Input("f-comuna", "value"),
    Input("f-clima", "value"),
    Input("f-vehiculo", "value"),
)
def actualizar_dashboard(comunas, climas, vehiculos):
    f = df
    if comunas:
        f = f[f["comuna"].isin(comunas)]
    if climas:
        f = f[f["clima"].isin(climas)]
    if vehiculos:
        f = f[f["tipo_vehiculo"].isin(vehiculos)]

    por_hora = f.groupby("hora_pedido", as_index=False)["retrasado"].mean()
    fig_hora = px.bar(por_hora, x="hora_pedido", y="retrasado",
                      title="Tasa de retraso por hora del día",
                      labels={"retrasado": "tasa de retraso", "hora_pedido": "hora"})
    fig_hora.update_layout(yaxis_tickformat=".0%")

    por_comuna = (f.groupby("comuna", as_index=False)["retrasado"].mean()
                  .sort_values("retrasado", ascending=False))
    fig_comuna = px.bar(por_comuna, x="comuna", y="retrasado",
                        title="Tasa de retraso por comuna (diferencias no significativas)",
                        labels={"retrasado": "tasa de retraso"})
    fig_comuna.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 0.6])

    tabla = f.pivot_table(index="trafico", columns="clima",
                          values="retrasado", aggfunc="mean")
    tabla = tabla.reindex(index=["Bajo", "Medio", "Alto"],
                          columns=["Despejado", "Nublado", "Lluvia"])
    fig_heat = px.imshow(tabla, text_auto=".0%", color_continuous_scale="RdYlGn_r",
                         title="Tasa de retraso: tráfico × clima",
                         labels={"color": "tasa"})

    fig_dist = px.histogram(f, x="tiempo_entrega_min", nbins=60,
                            title="Distribución del tiempo de entrega",
                            labels={"tiempo_entrega_min": "minutos"})
    fig_dist.add_vline(x=50, line_dash="dash", line_color="red",
                       annotation_text="promesa 50 min")

    return (f"{len(f):,}", f"{f['retrasado'].mean():.1%}",
            f"{f['tiempo_entrega_min'].mean():.0f} min",
            fig_hora, fig_comuna, fig_heat, fig_dist)


@app.callback(
    Output("resultado-pred", "children"),
    Output("resultado-pred", "style"),
    Input("s-dist", "value"), Input("s-prep", "value"), Input("s-items", "value"),
    Input("s-hora", "value"), Input("s-dia", "value"), Input("s-trafico", "value"),
    Input("s-clima", "value"), Input("s-comuna", "value"), Input("s-vehiculo", "value"),
    Input("s-comercio", "value"), Input("s-exp", "value"), Input("s-activos", "value"),
)
def predecir(dist, prep, items, hora, dia, trafico, clima, comuna,
             vehiculo, comercio, exp, activos):
    valor = float(df["valor_pedido_clp"].median())
    ctx = CONTEXTO_COMUNA[comuna]  # datos externos de las fuentes SQL y API
    fila = pd.DataFrame([{
        "distancia_km": dist, "tiempo_preparacion_min": prep,
        "items_pedido": items, "valor_pedido_clp": valor, "hora_pedido": hora,
        "dia_semana": dia, "experiencia_repartidor_meses": exp,
        "calificacion_repartidor": float(df["calificacion_repartidor"].median()),
        "pedidos_activos_repartidor": activos,
        "es_hora_punta": int(hora in [13, 14, 20, 21]),
        "es_fin_de_semana": int(dia >= 6),
        "valor_por_item": valor / items,
        "repartidor_novato": int(exp < 6),
        "comuna": comuna, "clima": clima, "trafico": trafico,
        "tipo_vehiculo": vehiculo, "tipo_comercio": comercio,
        **ctx,
    }])
    proba = modelo.predict_proba(fila)[0, 1]
    base = {"marginTop": "20px", "padding": "18px", "borderRadius": "10px",
            "fontSize": "20px", "textAlign": "center", "fontWeight": "bold"}
    if proba >= 0.5:
        base |= {"background": "#fdecea", "color": "#b71c1c"}
        return f"⚠️ Probabilidad de retraso: {proba:.0%} — pedido en riesgo", base
    base |= {"background": "#e8f5e9", "color": "#1b5e20"}
    return f"✅ Probabilidad de retraso: {proba:.0%} — llegaría a tiempo", base


if __name__ == "__main__":
    app.run(debug=True)