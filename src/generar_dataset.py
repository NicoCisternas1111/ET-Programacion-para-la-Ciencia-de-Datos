"""
Generador de dataset sintético: Pedidos de delivery en Santiago de Chile.
Caso: predecir si un pedido llegará RETRASADO (clasificación binaria)
y opcionalmente el tiempo de entrega en minutos (regresión).

El dataset se genera "sucio" a propósito (nulos, duplicados, categorías
inconsistentes, outliers) para justificar un pipeline ETL real.

Uso:
    python src/generar_dataset.py
Salida:
    data/pedidos_delivery.csv  (~52.000 filas)
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 52_000

COMUNAS = [
    "Santiago Centro", "Providencia", "Las Condes", "Ñuñoa", "Maipú",
    "La Florida", "Puente Alto", "San Miguel", "Estación Central",
    "Recoleta", "Independencia", "Macul", "Peñalolén", "Vitacura",
]
CLIMA = ["Despejado", "Nublado", "Lluvia"]
TRAFICO = ["Bajo", "Medio", "Alto"]
VEHICULO = ["Bicicleta", "Moto", "Auto"]
TIPO_COMERCIO = ["Restaurante", "Comida rápida", "Supermercado", "Farmacia", "Tienda"]


def main() -> None:
    # ----- Variables base -----
    hora = RNG.choice(range(9, 24), size=N, p=_pesos_hora())
    dia = RNG.integers(1, 8, size=N)  # 1=lunes ... 7=domingo
    comuna = RNG.choice(COMUNAS, size=N)
    clima = RNG.choice(CLIMA, size=N, p=[0.55, 0.30, 0.15])
    vehiculo = RNG.choice(VEHICULO, size=N, p=[0.20, 0.60, 0.20])
    comercio = RNG.choice(TIPO_COMERCIO, size=N, p=[0.35, 0.30, 0.15, 0.10, 0.10])

    es_hora_punta = np.isin(hora, [13, 14, 20, 21]).astype(int)
    trafico_p = np.where(
        es_hora_punta[:, None],
        [0.10, 0.35, 0.55],
        [0.45, 0.40, 0.15],
    )
    trafico = np.array([RNG.choice(TRAFICO, p=p) for p in trafico_p])

    distancia = np.round(RNG.gamma(shape=2.2, scale=1.6, size=N) + 0.4, 2)  # km
    items = RNG.poisson(lam=3.2, size=N) + 1
    valor = np.round((items * RNG.normal(6500, 1800, size=N)).clip(2500, 90000), 0)
    prep = np.round(RNG.normal(14, 5, size=N).clip(3, 45) + (comercio == "Restaurante") * 6, 1)
    exp_meses = RNG.integers(0, 61, size=N)
    calif = np.round((4.1 + exp_meses / 120 + RNG.normal(0, 0.35, size=N)).clip(2.5, 5.0), 2)
    pedidos_activos = RNG.integers(1, 5, size=N)

    # ----- Tiempo de entrega (señal real) -----
    base = 10 + distancia * 3.2 + prep * 0.7 + pedidos_activos * 2.5
    base += (trafico == "Medio") * 5 + (trafico == "Alto") * 12
    base += (clima == "Lluvia") * 8 + (clima == "Nublado") * 2
    base += (vehiculo == "Bicicleta") * distancia * 1.6
    base -= np.minimum(exp_meses, 36) * 0.15
    base += es_hora_punta * 3.5
    tiempo = np.round(base + RNG.normal(0, 6, size=N), 1).clip(8, None)

    # Retrasado = supera la promesa de entrega (50 min)
    retrasado = (tiempo > 50).astype(int)

    df = pd.DataFrame({
        "id_pedido": [f"PED_{i:05d}" for i in range(1, N + 1)],
        "distancia_km": distancia,
        "tiempo_preparacion_min": prep,
        "items_pedido": items,
        "valor_pedido_clp": valor,
        "hora_pedido": hora,
        "dia_semana": dia,
        "comuna": comuna,
        "clima": clima,
        "trafico": trafico,
        "tipo_vehiculo": vehiculo,
        "tipo_comercio": comercio,
        "experiencia_repartidor_meses": exp_meses,
        "calificacion_repartidor": calif,
        "pedidos_activos_repartidor": pedidos_activos,
        "tiempo_entrega_min": tiempo,
        "retrasado": retrasado,
    })

    # ----- Ensuciar el dataset (para el ETL) -----
    for col, frac in [
        ("distancia_km", 0.025), ("tiempo_preparacion_min", 0.03),
        ("calificacion_repartidor", 0.02), ("valor_pedido_clp", 0.015),
        ("clima", 0.01),
    ]:
        idx = RNG.choice(N, size=int(N * frac), replace=False)
        df.loc[idx, col] = np.nan

    # Categorías inconsistentes (mayúsculas/minúsculas y espacios)
    idx = RNG.choice(N, size=int(N * 0.02), replace=False)
    df.loc[idx, "trafico"] = df.loc[idx, "trafico"].str.lower()
    idx = RNG.choice(N, size=int(N * 0.015), replace=False)
    df.loc[idx, "clima"] = " " + df.loc[idx, "clima"].astype(str) + " "

    # Outliers imposibles
    idx = RNG.choice(N, size=40, replace=False)
    df.loc[idx, "distancia_km"] = RNG.uniform(80, 300, size=40).round(1)
    idx = RNG.choice(N, size=25, replace=False)
    df.loc[idx, "tiempo_entrega_min"] = -1

    # Filas duplicadas
    dups = df.sample(600, random_state=7)
    df = pd.concat([df, dups], ignore_index=True).sample(frac=1, random_state=7).reset_index(drop=True)

    df.to_csv("data/pedidos_delivery.csv", index=False)
    print(f"OK -> data/pedidos_delivery.csv | filas: {len(df)} | tasa retraso: {df['retrasado'].mean():.2%}")


def _pesos_hora():
    horas = list(range(9, 24))
    w = np.array([1, 2, 3, 5, 8, 8, 4, 3, 3, 4, 6, 9, 9, 5, 2], dtype=float)
    assert len(w) == len(horas)
    return w / w.sum()


if __name__ == "__main__":
    main()
