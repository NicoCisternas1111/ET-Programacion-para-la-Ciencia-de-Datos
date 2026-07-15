"""
Modelos SUPERVISADOS — Clasificación de pedidos retrasados.

Compara tres modelos usando Pipelines de Scikit-learn (escalado + one-hot
integrados, para que la validación cruzada no filtre información):

    1. Regresión Logística           -> línea base, interpretable
    2. Árbol de Decisión             -> optimizado con GridSearchCV
    3. Random Forest                 -> optimizado con GridSearchCV

Incluye:
    - Optimización de hiperparámetros con GridSearchCV + validación cruzada
    - Análisis train vs test para detectar sobreajuste
    - Métricas: accuracy, precision, recall, F1 y ROC-AUC
    - Evaluación del aporte de las variables externas (fuentes SQL y API)

Uso:
    python src/modelo.py
Salidas:
    models/modelo_retrasos.joblib
    reports/metricas.json
    reports/fig_matriz_confusion.png
    reports/fig_importancia_variables.png
"""

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
CV_FOLDS = 5

# tiempo_entrega_min se EXCLUYE: es información posterior al evento (fuga de datos)
# Variables operativas, conocidas al momento de crear el pedido
NUMERICAS_OPERATIVAS = [
    "distancia_km", "tiempo_preparacion_min", "items_pedido",
    "valor_pedido_clp", "hora_pedido", "dia_semana",
    "experiencia_repartidor_meses", "calificacion_repartidor",
    "pedidos_activos_repartidor", "es_hora_punta", "es_fin_de_semana",
    "valor_por_item", "repartidor_novato",
]
# Variables provenientes de las fuentes externas (SQL y API). Se incorporan
# para evaluar si el contexto territorial/meteorológico mejora la predicción.
NUMERICAS_EXTERNAS = [
    "densidad_hab_km2",       # fuente SQL
    "comuna_alta_densidad",   # derivada de la fuente SQL
    "temperatura_c",          # fuente API
    "humedad_pct",            # fuente API
    "viento_kmh",             # fuente API
]
NUMERICAS = NUMERICAS_OPERATIVAS + NUMERICAS_EXTERNAS
CATEGORICAS = ["comuna", "clima", "trafico", "tipo_vehiculo", "tipo_comercio",
               "nse"]  # nse viene de la fuente SQL
TARGET = "retrasado"


def construir_preprocesador() -> ColumnTransformer:
    """
    Escalado para numéricas + one-hot para categóricas.

    Se construye uno NUEVO por modelo: un ColumnTransformer guarda estado al
    entrenarse, y compartir la misma instancia entre pipelines distintos es
    una fuente silenciosa de errores.
    """
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERICAS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
    ])


def evaluar(pipe, X_train, X_test, y_train, y_test) -> tuple[dict, any]:
    """Calcula las métricas en test y el gap train-test (sobreajuste)."""
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    acc_train = accuracy_score(y_train, pipe.predict(X_train))
    acc_test = accuracy_score(y_test, pred)
    return {
        "accuracy": round(acc_test, 4),
        "precision": round(precision_score(y_test, pred), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "accuracy_train": round(acc_train, 4),
        "gap_train_test": round(acc_train - acc_test, 4),
        "sobreajuste": "posible" if acc_train - acc_test > 0.10 else "no detectado",
    }, pred


def main() -> None:
    Path("models").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    df = pd.read_csv("data/pedidos_limpio.csv")
    X = df[NUMERICAS + CATEGORICAS]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {X_train.shape} | Test: {X_test.shape}\n")

    pre = construir_preprocesador()
    resultados: dict = {}

    # ---------- Modelo 1: Regresión Logística (línea base) ----------
    print("Modelo 1/3 — Regresión Logística (línea base)")
    pipe_log = Pipeline([("pre", construir_preprocesador()),
                         ("clf", LogisticRegression(max_iter=1000))])
    pipe_log.fit(X_train, y_train)
    resultados["regresion_logistica"], pred_log = evaluar(
        pipe_log, X_train, X_test, y_train, y_test)
    resultados["regresion_logistica"]["hiperparametros"] = {"max_iter": 1000}
    print(f"   F1 {resultados['regresion_logistica']['f1']}\n")

    # ---------- Modelo 2: Árbol de Decisión + GridSearchCV ----------
    print(f"Modelo 2/3 — Árbol de Decisión (GridSearchCV, cv={CV_FOLDS})")
    grid_arbol = GridSearchCV(
        Pipeline([("pre", construir_preprocesador()),("clf", DecisionTreeClassifier(
            random_state=RANDOM_STATE))]),
        param_grid={
            "clf__max_depth": [4, 8, 12, None],
            "clf__min_samples_leaf": [1, 20, 50],
        },
        scoring="f1", cv=CV_FOLDS, n_jobs=-1,
    )
    grid_arbol.fit(X_train, y_train)
    print(f"   Mejores hiperparámetros: {grid_arbol.best_params_}")
    print(f"   F1 en validación cruzada: {grid_arbol.best_score_:.4f}")
    resultados["arbol_decision"], pred_arbol = evaluar(
        grid_arbol.best_estimator_, X_train, X_test, y_train, y_test)
    resultados["arbol_decision"]["hiperparametros"] = grid_arbol.best_params_
    resultados["arbol_decision"]["f1_cv"] = round(grid_arbol.best_score_, 4)
    print(f"   F1 {resultados['arbol_decision']['f1']}\n")

    # ---------- Modelo 3: Random Forest + GridSearchCV ----------
    print(f"Modelo 3/3 — Random Forest (GridSearchCV, cv={CV_FOLDS})")
    grid_rf = GridSearchCV(
        Pipeline([("pre", construir_preprocesador()), ("clf", RandomForestClassifier(n_jobs=-1, random_state=RANDOM_STATE))]),
        param_grid={
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [10, 14, None],
        },
        scoring="f1", cv=CV_FOLDS, n_jobs=-1,
    )
    grid_rf.fit(X_train, y_train)
    print(f"   Mejores hiperparámetros: {grid_rf.best_params_}")
    print(f"   F1 en validación cruzada: {grid_rf.best_score_:.4f}")
    resultados["random_forest"], pred_rf = evaluar(
        grid_rf.best_estimator_, X_train, X_test, y_train, y_test)
    resultados["random_forest"]["hiperparametros"] = grid_rf.best_params_
    resultados["random_forest"]["f1_cv"] = round(grid_rf.best_score_, 4)
    print(f"   F1 {resultados['random_forest']['f1']}\n")

    # ---------- Selección del mejor modelo por F1 ----------
    pipes = {"regresion_logistica": pipe_log,
             "arbol_decision": grid_arbol.best_estimator_,
             "random_forest": grid_rf.best_estimator_}
    preds = {"regresion_logistica": pred_log,
             "arbol_decision": pred_arbol,
             "random_forest": pred_rf}
    mejor = max(resultados, key=lambda k: resultados[k]["f1"])
    mejor_pipe = pipes[mejor]
    print(f"Mejor modelo por F1: {mejor} ({resultados[mejor]['f1']})")

    # ---------- Validación cruzada del modelo ganador ----------
    scores = cross_val_score(mejor_pipe, X_train, y_train,
                             cv=CV_FOLDS, scoring="f1", n_jobs=-1)
    print(f"Validación cruzada ({CV_FOLDS} folds) del ganador: "
          f"F1 {scores.mean():.4f} +/- {scores.std():.4f}\n")

    # ---------- Aporte de las variables externas (SQL + API) ----------
    # Se reentrena el ganador SOLO con variables operativas para medir
    # cuánto aportó realmente la integración de fuentes externas.
    pre_solo_op = ColumnTransformer([
        ("num", StandardScaler(), NUMERICAS_OPERATIVAS),
        ("cat", OneHotEncoder(handle_unknown="ignore"),
         [c for c in CATEGORICAS if c != "nse"]),
    ])
    pipe_sin_ext = Pipeline([("pre", pre_solo_op),
                             ("clf", LogisticRegression(max_iter=1000))])
    pipe_sin_ext.fit(X_train, y_train)
    f1_sin_ext = f1_score(y_test, pipe_sin_ext.predict(X_test))
    aporte = {
        "f1_solo_variables_operativas": round(float(f1_sin_ext), 4),
        "f1_con_variables_externas": resultados["regresion_logistica"]["f1"],
        "diferencia": round(
            resultados["regresion_logistica"]["f1"] - float(f1_sin_ext), 4),
    }
    print("Aporte de las fuentes externas (SQL + API) sobre la línea base:")
    print(f"   Solo operativas : F1 {aporte['f1_solo_variables_operativas']}")
    print(f"   Con externas    : F1 {aporte['f1_con_variables_externas']}")
    print(f"   Diferencia      : {aporte['diferencia']:+}\n")

    # ---------- Guardar artefactos ----------
    joblib.dump(mejor_pipe, "models/modelo_retrasos.joblib")
    salida = {
        "mejor_modelo": mejor,
        "resultados": resultados,
        "validacion_cruzada_ganador": {
            "folds": CV_FOLDS,
            "f1_medio": round(float(scores.mean()), 4),
            "f1_desviacion": round(float(scores.std()), 4),
            "f1_por_fold": [round(float(s), 4) for s in scores],
        },
        "aporte_fuentes_externas": aporte,
    }
    with open("reports/metricas.json", "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False)

    # Matriz de confusión del ganador
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix(y_test, preds[mejor]),
        display_labels=["A tiempo", "Retrasado"],
    ).plot(ax=ax, colorbar=False)
    ax.set_title(f"Matriz de confusión — {mejor}")
    fig.tight_layout()
    fig.savefig("reports/fig_matriz_confusion.png", dpi=120)
    plt.close(fig)

    # Importancia de variables del ganador
    nombres = mejor_pipe.named_steps["pre"].get_feature_names_out()
    clf = mejor_pipe.named_steps["clf"]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    if hasattr(clf, "feature_importances_"):
        imp = (pd.Series(clf.feature_importances_, index=nombres)
               .sort_values(ascending=True).tail(12))
        imp.plot.barh(ax=ax, color="#2a9d8f")
        ax.set_title(f"Top 12 variables más importantes — {mejor}")
        ax.set_xlabel("Importancia (Gini)")
    else:
        coefs = pd.Series(clf.coef_[0], index=nombres)
        top = coefs.reindex(coefs.abs().sort_values(ascending=True).tail(12).index)
        top.plot.barh(ax=ax, color=["#e76f51" if v > 0 else "#2a9d8f" for v in top])
        ax.set_title(f"Top 12 coeficientes (|β|) — {mejor}")
        ax.set_xlabel("Coeficiente (+ aumenta prob. de retraso)")
        ax.axvline(0, color="gray", lw=0.8)
    fig.tight_layout()
    fig.savefig("reports/fig_importancia_variables.png", dpi=120)
    plt.close(fig)

    print(json.dumps(salida, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()