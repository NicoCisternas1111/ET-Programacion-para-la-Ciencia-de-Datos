"""
Modelo NO SUPERVISADO — Segmentación de comunas con K-Means + PCA.

Complementa al modelo supervisado: mientras la clasificación predice SI un
pedido se retrasará, la segmentación descubre QUÉ TIPOS DE TERRITORIO opera
la plataforma, sin usar la etiqueta 'retrasado'.

Por qué comunas y no pedidos:
    Se probó primero segmentar los 51.975 pedidos. El silhouette fue de
    0.13 (estructura prácticamente nula): los pedidos individuales no
    forman grupos naturales. A nivel de comuna, en cambio, los datos
    externos (Censo + API) sí muestran estructura. La unidad de decisión
    del negocio también es la comuna: la flota se planifica por territorio,
    no pedido a pedido.

Por qué solo variables externas en X:
    Las métricas operativas (tasa de retraso, distancia, tiempo de entrega)
    resultaron homogéneas entre las 14 comunas — ver reporte generado.
    Incluirlas bajaba el silhouette de 0.33 a 0.23 porque solo aportaban
    ruido. Se excluyen del clustering y se usan para INTERPRETAR.

Metodología (la vista en clase):
    1. Agregar los pedidos a nivel comuna y unir datos externos
    2. Seleccionar X (sin identificadores, sin target)
    3. Escalar con StandardScaler (K-Means mide distancias)
    4. Probar K = 2..6 -> inertia + silhouette
    5. Método del codo + silhouette para justificar K
    6. Entrenar K-Means con el K elegido
    7. PCA(n_components=2) SOLO para visualizar
    8. Interpretar con los promedios de las variables ORIGINALES

Importante: PCA no crea los clusters. Los clusters los crea K-Means.
PCA solo permite dibujar en 2D un espacio de 6 variables.

Uso:
    python src/clustering.py
Salidas:
    reports/metricas_clustering.json
    reports/fig_codo_silhouette.png
    reports/fig_clusters_pca.png
    data/comunas_segmentadas.csv
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
RANGO_K = range(2, 7)
K_ELEGIDO = 2

# X del clustering: datos externos reales (fuentes SQL y API).
# id_cliente/comuna no entra: solo identifica, no aporta para agrupar.
VARIABLES = [
    "poblacion",        # fuente SQL (Censo 2017)
    "superficie_km2",   # fuente SQL
    "densidad_hab_km2", # fuente SQL (calculada en la query)
    "temperatura_c",    # fuente API (Open-Meteo)
    "humedad_pct",      # fuente API
    "viento_kmh",       # fuente API
]

# Variables operativas: NO entran al clustering, sirven para interpretar.
OPERATIVAS = ["n_pedidos", "tasa_retraso", "distancia_media",
              "tiempo_entrega_medio", "valor_medio"]

NOMBRES_SEGMENTO = {
    0: "Periférica extensa (baja densidad)",
    1: "Céntrica compacta (alta densidad)",
}


def construir_tabla_comunas(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega los pedidos a nivel comuna y adjunta los datos externos."""
    return df.groupby("comuna").agg(
        n_pedidos=("retrasado", "size"),
        tasa_retraso=("retrasado", "mean"),
        distancia_media=("distancia_km", "mean"),
        tiempo_entrega_medio=("tiempo_entrega_min", "mean"),
        valor_medio=("valor_pedido_clp", "mean"),
        poblacion=("poblacion", "first"),
        superficie_km2=("superficie_km2", "first"),
        densidad_hab_km2=("densidad_hab_km2", "first"),
        nse=("nse", "first"),
        temperatura_c=("temperatura_c", "first"),
        humedad_pct=("humedad_pct", "first"),
        viento_kmh=("viento_kmh", "first"),
    ).reset_index()


def diagnosticar_homogeneidad(comunas: pd.DataFrame) -> dict:
    """
    Evidencia que justifica excluir las variables operativas del clustering:
    su dispersión entre comunas es despreciable.
    """
    diag = {}
    for col in OPERATIVAS:
        s = comunas[col]
        diag[col] = {
            "min": round(float(s.min()), 3),
            "max": round(float(s.max()), 3),
            "coef_variacion": round(float(s.std() / s.mean()), 4),
        }
    return diag


def comparar_k(X_escalado) -> pd.DataFrame:
    """Prueba varios K y devuelve inertia y silhouette de cada uno."""
    filas = []
    for k in RANGO_K:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        etiquetas = km.fit_predict(X_escalado)
        sil = silhouette_score(X_escalado, etiquetas)
        filas.append({"k": k,
                      "inertia": round(float(km.inertia_), 2),
                      "silhouette": round(float(sil), 4)})
        print(f"   K={k} -> inertia {km.inertia_:7.2f} | silhouette {sil:.4f}")
    return pd.DataFrame(filas)


def graficar_codo(tabla: pd.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(tabla["k"], tabla["inertia"], marker="o", color="#264653")
    ax1.axvline(K_ELEGIDO, color="#e76f51", ls="--", lw=1)
    ax1.set_title("Método del codo")
    ax1.set_xlabel("Número de clusters (K)")
    ax1.set_ylabel("Inertia")

    ax2.plot(tabla["k"], tabla["silhouette"], marker="o", color="#2a9d8f")
    ax2.axvline(K_ELEGIDO, color="#e76f51", ls="--", lw=1)
    ax2.set_title("Silhouette Score por K")
    ax2.set_xlabel("Número de clusters (K)")
    ax2.set_ylabel("Silhouette")

    fig.suptitle(f"Selección de K — se elige K={K_ELEGIDO} (mayor silhouette)")
    fig.tight_layout()
    fig.savefig("reports/fig_codo_silhouette.png", dpi=120)
    plt.close(fig)


def graficar_pca(comunas: pd.DataFrame, pca: PCA) -> None:
    """Dibuja las comunas en el espacio reducido por PCA, con su nombre."""
    fig, ax = plt.subplots(figsize=(8.5, 6))
    colores = ["#e76f51", "#2a9d8f", "#264653", "#e9c46a"]
    for c in sorted(comunas["cluster"].unique()):
        sub = comunas[comunas["cluster"] == c]
        ax.scatter(sub["PC1"], sub["PC2"], s=140, alpha=0.85,
                   color=colores[c % len(colores)],
                   label=f"Cluster {c} — {NOMBRES_SEGMENTO.get(c, '')}")
    for _, fila in comunas.iterrows():
        ax.annotate(fila["comuna"], (fila["PC1"], fila["PC2"]),
                    fontsize=8, xytext=(6, 4), textcoords="offset points")
    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var[0]:.1%} de la varianza)")
    ax.set_ylabel(f"PC2 ({var[1]:.1%} de la varianza)")
    ax.set_title(f"Segmentación de comunas (K={K_ELEGIDO}) vista con PCA")
    ax.legend(title="Segmento", loc="best", fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig("reports/fig_clusters_pca.png", dpi=120)
    plt.close(fig)


def main() -> None:
    Path("reports").mkdir(exist_ok=True)
    df = pd.read_csv("data/pedidos_limpio.csv")

    # ---- 1. Agregar a nivel comuna ----
    comunas = construir_tabla_comunas(df)
    print(f"Segmentando {len(comunas)} comunas "
          f"(agregadas desde {len(df):,} pedidos)\n")

    # Evidencia de por qué las operativas quedan fuera de X
    diag = diagnosticar_homogeneidad(comunas)
    print("Diagnóstico de homogeneidad (por eso se excluyen del clustering):")
    for col, v in diag.items():
        print(f"   {col:<22} rango [{v['min']}, {v['max']}]  CV={v['coef_variacion']}")

    # ---- 2 y 3. Seleccionar X y escalar ----
    X = comunas[VARIABLES]
    X_escalado = StandardScaler().fit_transform(X)

    # ---- 4 y 5. Comparar K y método del codo ----
    print(f"\nComparación de K sobre {len(VARIABLES)} variables externas:")
    tabla = comparar_k(X_escalado)
    graficar_codo(tabla)

    # ---- 6. Entrenar con el K elegido ----
    kmeans = KMeans(n_clusters=K_ELEGIDO, random_state=RANDOM_STATE, n_init=10)
    comunas["cluster"] = kmeans.fit_predict(X_escalado)
    comunas["segmento"] = comunas["cluster"].map(NOMBRES_SEGMENTO)

    # ---- 7. PCA solo para visualizar ----
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    componentes = pca.fit_transform(X_escalado)
    comunas["PC1"], comunas["PC2"] = componentes[:, 0], componentes[:, 1]
    graficar_pca(comunas, pca)
    var_total = float(pca.explained_variance_ratio_.sum())
    print(f"\nPCA: PC1 {pca.explained_variance_ratio_[0]:.1%} + "
          f"PC2 {pca.explained_variance_ratio_[1]:.1%} = "
          f"{var_total:.1%} de la varianza original resumida en 2D")

    # ---- 8. Interpretar con las variables ORIGINALES ----
    perfil = comunas.groupby("cluster")[VARIABLES + OPERATIVAS].mean().round(1)
    # tasa_retraso vive en [0,1]: con 1 decimal se pierde toda la diferencia
    perfil["tasa_retraso"] = (
        comunas.groupby("cluster")["tasa_retraso"].mean().round(4))
    perfil["comunas"] = comunas.groupby("cluster")["comuna"].apply(
        lambda s: ", ".join(sorted(s)))
    print("\nPerfil de cada segmento:")
    for c, fila in perfil.iterrows():
        print(f"\n   Cluster {c} — {NOMBRES_SEGMENTO.get(c, '')}")
        print(f"      Comunas         : {fila['comunas']}")
        print(f"      Población media : {fila['poblacion']:>10,.0f}")
        print(f"      Superficie media: {fila['superficie_km2']:>10,.1f} km²")
        print(f"      Densidad media  : {fila['densidad_hab_km2']:>10,.0f} hab/km²")
        print(f"      Tasa de retraso : {fila['tasa_retraso']:>10.4f}  (interpretación)")

    sil_elegido = float(tabla.loc[tabla["k"] == K_ELEGIDO, "silhouette"].iloc[0])
    comunas.to_csv("data/comunas_segmentadas.csv", index=False)
    with open("reports/metricas_clustering.json", "w", encoding="utf-8") as f:
        json.dump({
            "unidad_de_analisis": "comuna",
            "n_observaciones": len(comunas),
            "k_elegido": K_ELEGIDO,
            "criterio_seleccion_k": (
                "Mayor silhouette del rango probado (K=2..6) y segmentos "
                "interpretables para la operación."
            ),
            "variables_usadas": VARIABLES,
            "variables_excluidas_por_homogeneidad": OPERATIVAS,
            "diagnostico_homogeneidad": diag,
            "comparacion_k": tabla.to_dict(orient="records"),
            "inertia_k_elegido": round(float(kmeans.inertia_), 2),
            "silhouette_k_elegido": sil_elegido,
            "lectura_silhouette": (
                f"{sil_elegido:.3f} indica estructura moderada: los grupos "
                "existen y son interpretables, pero no están fuertemente "
                "separados. Se reporta tal cual, sin sobreinterpretar."
            ),
            "pca_varianza_explicada": {
                "PC1": round(float(pca.explained_variance_ratio_[0]), 4),
                "PC2": round(float(pca.explained_variance_ratio_[1]), 4),
                "total": round(var_total, 4),
            },
            "perfil_clusters": perfil.reset_index().to_dict(orient="records"),
        }, f, indent=2, ensure_ascii=False)

    print("\nOK -> data/comunas_segmentadas.csv")
    print("OK -> reports/metricas_clustering.json")
    print("OK -> reports/fig_codo_silhouette.png, reports/fig_clusters_pca.png")


if __name__ == "__main__":
    main()