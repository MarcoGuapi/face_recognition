"""Precision, recall, F1 y matriz de confusion de un pickle de encodings.

A cada rostro se le busca el vecino mas cercano entre todos los demas (nunca
contra si mismo) y ese vecino da la prediccion. Las distancias se calculan con
face_recognition.face_distance, la misma funcion que usa
recognize_faces_image.py, para que lo que se mide sea lo que se ejecuta.

Ademas guarda en output/metricas/, con el nombre del pickle como prefijo:
    <pickle>_matriz_confusion.png    mapa de calor
    <pickle>_metricas.png            grafico de barras
    <pickle>_tabla_metricas.png      tabla precision / recall / F1 como imagen

USAGE
    python evaluar.py
    python evaluar.py runs/05_cnn.pickle
"""

import pickle
import sys
from pathlib import Path

import face_recognition
import matplotlib

matplotlib.use("Agg")  # solo guarda PNG, no abre ventanas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import classification_report, confusion_matrix

# =============================================================================
# CONFIGURACION
# =============================================================================

CARPETA_SALIDA = Path("output") / "metricas"
DPI = 150

# colores
TEXTO = "#0b0b0b"         # texto principal
TEXTO_2 = "#52514e"       # ejes y etiquetas
GRILLA = "#e4e3df"        # lineas de grilla y bordes
FONDO_ENCABEZADO = "#f0efec"
FONDO_RESUMEN = "#f7f7f5"
AZULES = ["#fcfcfb", "#cde2fb", "#86b6ef", "#2a78d6", "#184f95"]  # claro -> oscuro
COLOR_SERIE = {"precision": "#2a78d6", "recall": "#eb6834", "f1-score": "#1baf7a"}

plt.rcParams.update({
    "font.size": 10,
    "text.color": TEXTO,
    "axes.labelcolor": TEXTO_2,
    "xtick.color": TEXTO_2,
    "ytick.color": TEXTO_2,
    "axes.edgecolor": GRILLA,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


# =============================================================================
# 1. EVALUACION
# =============================================================================

def cargar_encodings(ruta):
    """Lee el pickle y devuelve los encodings (X) y los nombres (y)."""
    with open(ruta, "rb") as f:
        data = pickle.load(f)
    return np.array(data["encodings"]), np.array(data["names"])


def predecir_vecino_mas_cercano(X, y):
    """Predice cada rostro con el nombre de su vecino mas cercano."""
    # distancia de cada rostro a todos, con la misma llamada de
    # recognize_faces_image.py
    D = np.array([face_recognition.face_distance(X, encoding) for encoding in X])

    # la diagonal es la distancia de cada rostro a si mismo (cero): se pone en
    # infinito para que nadie se reconozca a si mismo, lo que daria 1.000 falso
    np.fill_diagonal(D, np.inf)
    return y[D.argmin(axis=1)]


def tabla_de_metricas(y, y_pred):
    """precision / recall / F1 / support por persona + accuracy y promedios."""
    metricas = pd.DataFrame(classification_report(
        y, y_pred, zero_division=0, output_dict=True)).T
    # en la fila accuracy solo tiene sentido un valor (como en la consola)
    metricas.loc["accuracy", ["precision", "recall"]] = np.nan
    metricas.loc["accuracy", "support"] = len(y)
    return metricas


def imprimir_resultados(ruta, y, y_pred, etiquetas, cm):
    """Muestra en consola el reporte y la matriz de confusion."""
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    print(f"\n{ruta}: {len(y)} rostros, {len(etiquetas)} personas\n")

    print("--- precision / recall / F1 ---")
    print(classification_report(y, y_pred, zero_division=0, digits=3))

    print("--- matriz de confusion ---")
    print(pd.DataFrame(cm, index=pd.Index(etiquetas, name="real"),
                       columns=[e[:8] for e in etiquetas]))


# =============================================================================
# 2. GRAFICO: MATRIZ DE CONFUSION (mapa de calor)
# =============================================================================

def grafico_matriz_confusion(cm, etiquetas, titulo, archivo):
    """Mapa de calor en un solo tono azul, con la cantidad en cada celda."""
    fig, ax = plt.subplots(figsize=(8, 7))

    # mapa de calor
    cmap = LinearSegmentedColormap.from_list("azules", AZULES)
    im = ax.imshow(cm, cmap=cmap, vmin=0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).outline.set_visible(False)

    # ejes
    ax.set_xticks(range(len(etiquetas)), etiquetas, rotation=45, ha="right")
    ax.set_yticks(range(len(etiquetas)), etiquetas)
    ax.set_xlabel("Predicha")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusión — {titulo}", loc="left", pad=12)
    ax.tick_params(length=0)
    for borde in ax.spines.values():
        borde.set_visible(False)

    # numero en cada celda (blanco sobre azul oscuro, negro sobre claro)
    umbral = cm.max() * 0.55
    for i in range(len(etiquetas)):
        for j in range(len(etiquetas)):
            if cm[i, j]:
                color = "white" if cm[i, j] > umbral else TEXTO
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        fontsize=9, color=color)

    guardar(fig, archivo)


# =============================================================================
# 3. GRAFICO: BARRAS DE PRECISION / RECALL / F1 POR PERSONA
# =============================================================================

def grafico_barras_metricas(metricas, etiquetas, accuracy, titulo, archivo):
    """Barras agrupadas: una barra por metrica, un grupo por persona."""
    fig, ax = plt.subplots(figsize=(11, 5))

    # barras: tres por persona, desplazadas a izquierda / centro / derecha
    x = np.arange(len(etiquetas))
    ancho = 0.26
    for k, (columna, color) in enumerate(COLOR_SERIE.items()):
        valores = metricas.loc[etiquetas, columna]
        ax.bar(x + (k - 1) * ancho, valores, ancho * 0.9,
               label=columna.replace("-score", "").capitalize(),
               color=color, zorder=2)

    # ejes y grilla
    ax.set_xticks(x, etiquetas, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Valor")
    ax.yaxis.grid(True, color=GRILLA, linewidth=0.8, zorder=0)
    ax.tick_params(length=0)
    for borde in ("top", "right", "left"):
        ax.spines[borde].set_visible(False)

    # titulo y leyenda en la misma linea
    ax.set_title(f"Precision, recall y F1 por persona — {titulo} "
                 f"(accuracy {accuracy:.3f})", loc="left", pad=12)
    ax.legend(frameon=False, ncol=3, loc="lower right", bbox_to_anchor=(1, 1.0))

    guardar(fig, archivo)


# =============================================================================
# 4. GRAFICO: TABLA DE METRICAS COMO IMAGEN
# =============================================================================

def grafico_tabla_metricas(metricas, n_personas, titulo, archivo):
    """La tabla del classification_report dibujada con matplotlib."""
    columnas = ["precision", "recall", "f1-score", "support"]

    # texto de cada celda: 3 decimales, support entero, vacio si no aplica
    def formato(valor, columna):
        if pd.isna(valor):
            return ""
        return f"{int(valor)}" if columna == "support" else f"{valor:.3f}"

    filas = list(metricas.index)
    celdas = [[formato(metricas.loc[f, c], c) for c in columnas] for f in filas]

    # figura: el alto crece con la cantidad de filas
    fig, ax = plt.subplots(figsize=(7, 0.42 * (len(filas) + 2)))
    ax.axis("off")
    ax.set_title(f"Precision / recall / F1 — {titulo}", loc="left", pad=8)

    tabla = ax.table(cellText=celdas, rowLabels=filas, colLabels=columnas,
                     cellLoc="right", rowLoc="left", loc="upper center",
                     bbox=[0.25, 0, 0.75, 1])
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(10)

    # estilo: fila 0 = encabezado; despues de las personas = resumen
    for (fila, _), celda in tabla.get_celld().items():
        celda.set_edgecolor(GRILLA)
        celda.set_linewidth(0.6)
        celda.set_text_props(color=TEXTO)
        if fila == 0:
            celda.set_facecolor(FONDO_ENCABEZADO)
            celda.set_text_props(weight="bold")
        elif fila > n_personas:
            celda.set_facecolor(FONDO_RESUMEN)
            celda.set_text_props(weight="bold")

    guardar(fig, archivo, ajustar_bordes=True)


# =============================================================================
# UTILIDADES
# =============================================================================

def guardar(fig, archivo, ajustar_bordes=False):
    """Guarda la figura como PNG y la cierra."""
    fig.tight_layout()
    fig.savefig(archivo, dpi=DPI,
                bbox_inches="tight" if ajustar_bordes else None)
    plt.close(fig)


# =============================================================================
# PROGRAMA PRINCIPAL
# =============================================================================

def main():
    ruta = sys.argv[1] if len(sys.argv) > 1 else "encodings.pickle"

    # evaluacion
    X, y = cargar_encodings(ruta)
    y_pred = predecir_vecino_mas_cercano(X, y)
    etiquetas = sorted(set(y))
    cm = confusion_matrix(y, y_pred, labels=etiquetas)
    metricas = tabla_de_metricas(y, y_pred)
    accuracy = (y == y_pred).mean()

    imprimir_resultados(ruta, y, y_pred, etiquetas, cm)

    # graficos: el nombre del pickle va como prefijo para poder comparar
    # corridas (hog vs cnn) sin que se sobrescriban
    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)
    prefijo = Path(ruta).stem
    archivos = {
        "matriz": CARPETA_SALIDA / f"{prefijo}_matriz_confusion.png",
        "barras": CARPETA_SALIDA / f"{prefijo}_metricas.png",
        "tabla": CARPETA_SALIDA / f"{prefijo}_tabla_metricas.png",
    }

    grafico_matriz_confusion(cm, etiquetas, prefijo, archivos["matriz"])
    grafico_barras_metricas(metricas, etiquetas, accuracy, prefijo,
                            archivos["barras"])
    grafico_tabla_metricas(metricas, len(etiquetas), prefijo, archivos["tabla"])

    print(f"\narchivos guardados en {CARPETA_SALIDA}/:")
    for archivo in archivos.values():
        print(f"  {archivo.name}")


if __name__ == "__main__":
    main()
