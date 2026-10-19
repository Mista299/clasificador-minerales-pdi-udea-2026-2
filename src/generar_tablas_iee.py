# -*- coding: utf-8 -*-
"""Genera las 3 tablas del informe IEEE como imágenes PNG con matplotlib.

Versión robusta: usa tamaños en pulgadas fijos y NO usa bbox_inches='tight'.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "figures" / "ieee_tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["DejaVu Sans", "Liberation Sans"]
mpl.rcParams["pdf.fonttype"] = 42


def render_table(headers, rows, out_name, *, col_widths_in, row_h_in=0.27,
                 header_h_in=0.32):
    n_cols = len(headers)
    n_rows = len(rows) + 1
    fig_w = sum(col_widths_in) + 0.05
    # Sumar altura para header + todas las filas + pequeño padding
    fig_h = (header_h_in + row_h_in * len(rows)) * 1.15

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_position([0, 0, 1, 1])

    tabla = ax.table(
        cellText=rows,
        colLabels=headers,
        colWidths=[w / fig_w for w in col_widths_in],
        loc="center",
        cellLoc="center",
    )
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(9)

    # Ajustar altura de cada celda en coordenadas relativas
    total_h = 0.85  # 85% del eje (margen alrededor)
    header_h = header_h_in / (header_h_in + row_h_in * len(rows))
    row_h = row_h_in / (header_h_in + row_h_in * len(rows))

    for j in range(n_cols):
        cell = tabla[(0, j)]
        cell.set_facecolor("#D9E5F4")
        cell.set_text_props(weight="bold", color="#1A2A52")
        cell.set_edgecolor("#8AA5C4")
        cell.set_height(header_h * 0.9)
    for i in range(len(rows)):
        for j in range(n_cols):
            cell = tabla[(i + 1, j)]
            cell.set_facecolor("#FFFFFF" if i % 2 == 0 else "#F4F8FD")
            cell.set_edgecolor("#C0CCDC")
            cell.set_height(row_h * 0.95)
            if j == 0:
                cell.set_text_props(ha="left")

    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    out = OUT_DIR / out_name
    plt.savefig(out, dpi=200, facecolor="white")
    plt.close()
    print(f"  {out_name}: {out.stat().st_size/1024:.1f} KB")


# ============================== TABLA I ==============================
render_table(
    ["Descriptor", "Dims", "Invarianzas", "Uso"],
    [
        ["HOG",                  "1764", "Trasl.; rot. parc.",  "E1–E4"],
        ["Color HSV (μ,σ,skew)", "9",    "Brillo",              "E3"],
        ["Color LAB (μ,σ,skew)", "9",    "Perceptual",          "E3"],
        ["Gabor 4θ × 2λ",        "16",   "Rot. (multi-or.)",    "E2–E4"],
        ["LBP P=8, R=1",         "10",   "Iluminación",         "E2, E2b"],
        ["GLCM (5 × 4 dirs)",    "20",   "Dirección",           "E2,E2b,E3"],
        ["Momentos de Hu",       "7",    "T / E / R",           "E2b, E3b"],
        ["Total E2b",            "1835", "—",                   "Pipeline"],
    ],
    "tabla1_descriptores.png",
    col_widths_in=[2.4, 0.7, 1.7, 1.7],
)

# ============================== TABLA II =============================
render_table(
    ["ID", "Descriptores", "Dims"],
    [
        ["E1",  "HOG",                          "1764"],
        ["E2",  "HOG + Col + Gab + LBP + GLCM", "1828"],
        ["E2b", "E2 + Hu",                      "1835"],
        ["E3",  "Color + GLCM  (mejor)",        "38"],
        ["E3b", "E3 + Hu",                      "45"],
        ["E4",  "HOG + Gabor",                  "1780"],
        ["E5",  "Hu solo",                      "7"],
    ],
    "tabla2_experimentos.png",
    col_widths_in=[0.5, 4.5, 1.5],
)

# ============================== TABLA III ============================
# (Datos del pipeline v3, donde el RF con class_weight='balanced' gana)
render_table(
    ["", "SVM", "k-NN", "RF†", "MLP", "Voting"],
    [
        ["E1 (1764d)",  "0.58", "0.42", "0.58", "0.36", "0.58"],
        ["E2 (1828d)",  "0.75", "0.50", "0.72", "0.69", "0.72"],
        ["E2b (1835d)", "0.75", "0.50", "0.72", "0.69", "0.72"],
        ["E3 (38d) ★",  "0.83", "0.50", "0.83", "0.81", "0.78"],
        ["E3b (45d)",   "0.78", "0.50", "0.72", "0.69", "0.75"],
        ["E4 (1780d)",  "0.72", "0.42", "0.67", "0.64", "0.72"],
        ["E5 (7d)",     "0.33", "0.25", "0.31", "0.31", "0.33"],
    ],
    "tabla3_resultados.png",
    col_widths_in=[1.4, 0.9, 0.9, 0.9, 0.9, 1.0],
)
print("† RF con class_weight='balanced'")