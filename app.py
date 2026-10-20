"""Interfaz web — Clasificador de minerales (Streamlit).

Uso:
    streamlit run app.py
"""
import sys
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import clasificador  # noqa: E402

# Modelo por defecto: v3 (mejor accuracy). Si no existe, fallback al v2.
# Tras reentrenar con 6 minerales, regenerar y mantener este orden.
MODEL_PATH = ROOT / "model" / "modelo_minerales_v3.pkl"
if not MODEL_PATH.exists():
    MODEL_PATH = ROOT / "model" / "modelo_minerales.pkl"

CLASE_COLOR = {
    "cuarzo": "#5B9BD5",
    "pirita": "#D4A017",
    "malaquita": "#2E9E5B",
    "hematita": "#C0504D",
    "calcita": "#F5F5DC",   # NUEVO v6 — blanco/crema (beige)
    "amatista": "#9966CC",  # NUEVO v6 — violeta
}

COMPARAR = "Comparar los 4 modelos"

MODELO_INFO = {
    "Voting Ensemble": "RF + MLP + SVM con soft voting — el más preciso",
    "Random Forest": "100 árboles · profundidad máx. 10",
    "MLP": "Red neuronal (128, 64, 32)",
    "SVM (RBF)": "Kernel RBF · C=10 · gamma=scale",
    COMPARAR: "Ejecuta los 4 modelos y compara sus predicciones",
}

CSS = """
<style>
.pred-card { padding: 1.1rem 1.3rem; border-radius: 14px;
             background: rgba(140,140,140,.07);
             border: 1px solid rgba(140,140,140,.16); }
.pred  { font-size: 2rem; font-weight: 800; letter-spacing: .04em; line-height: 1.1; }
.conf  { opacity: .7; margin: .2rem 0 .9rem; font-size: .95rem; }
.bar-row { display: flex; align-items: center; gap: .65rem; margin: .3rem 0; }
.bar-label { width: 5.6rem; text-transform: capitalize; font-size: .9rem; opacity: .9; }
.bar-bg { flex: 1; height: .8rem; border-radius: 999px;
          background: rgba(140,140,140,.16); overflow: hidden; }
.bar-fill { height: 100%; border-radius: 999px; }
.bar-val { width: 3.4rem; text-align: right; font-size: .85rem; opacity: .8;
           font-variant-numeric: tabular-nums; }
</style>
"""


@st.cache_resource
def cargar_modelos():
    """Carga el bundle y expone el ensemble + sus 3 estimadores ya entrenados."""
    bundle = joblib.load(MODEL_PATH)
    ens = bundle["modelo"]
    modelos = {
        "Voting Ensemble": ens,
        "Random Forest": ens.named_estimators_["rf"],
        "MLP": ens.named_estimators_["mlp"],
        "SVM (RBF)": ens.named_estimators_["svm"],
    }
    return bundle, modelos


def leer_imagen(uploaded_file) -> np.ndarray:
    data = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"No se pudo decodificar {uploaded_file.name}")
    return img


def predecir(img_bgr, modelo, bundle):
    """Devuelve (índice de clase, vector de probabilidades)."""
    feats = clasificador.extract_features(img_bgr)
    sl = slice(bundle["slice_start"], bundle["slice_end"])
    feats_s = bundle["scaler"].transform(feats[sl].reshape(1, -1))
    idx = int(modelo.predict(feats_s)[0])
    proba = modelo.predict_proba(feats_s)[0]
    return idx, proba


def barras_html(clases, proba) -> str:
    rows = []
    for c, p in zip(clases, proba):
        color = CLASE_COLOR.get(c, "#888")
        rows.append(
            f'<div class="bar-row"><span class="bar-label">{c}</span>'
            f'<div class="bar-bg"><div class="bar-fill" '
            f'style="width:{p * 100:.1f}%;background:{color}"></div></div>'
            f'<span class="bar-val">{p * 100:.1f}%</span></div>'
        )
    return '<div class="bars">' + "".join(rows) + "</div>"


def mostrar_resultado(nombre_archivo, img_bgr, idx, proba, dt_ms, clases, comparar=None):
    pred = clases[idx]
    col_img, col_res = st.columns([1, 1.6], gap="large")

    with col_img:
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                 caption=nombre_archivo, use_container_width=True)
        with st.expander("Ver preprocesamiento"):
            img_eq, gray, mask = clasificador.preprocess(img_bgr)
            c1, c2, c3 = st.columns(3)
            c1.image(cv2.cvtColor(img_eq, cv2.COLOR_BGR2RGB),
                     caption="CLAHE", use_container_width=True)
            c2.image(gray, caption="Gris", use_container_width=True)
            c3.image(mask, caption="Máscara Otsu", use_container_width=True)

    with col_res:
        color = CLASE_COLOR.get(pred, "#888")
        st.markdown(
            f'<div class="pred-card" style="border-left:6px solid {color}">'
            f'<div class="pred" style="color:{color}">{pred.upper()}</div>'
            f'<div class="conf">{proba.max() * 100:.1f}% de confianza · {dt_ms:.0f} ms</div>'
            + barras_html(clases, proba)
            + "</div>",
            unsafe_allow_html=True,
        )
        if proba.max() < 0.5:
            st.warning("Confianza baja: la imagen puede no corresponder a "
                       "ninguna de las clases entrenadas.")
        if comparar is not None:
            st.dataframe(comparar, hide_index=True, use_container_width=True)


st.set_page_config(page_title="Clasificador de Minerales",
                   page_icon="🔬", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

bundle, modelos = cargar_modelos()
clases = bundle["clases"]

with st.sidebar:
    st.header("Configuración")
    nombre_modelo = st.selectbox("Modelo de reconocimiento",
                                 list(MODELO_INFO)[:-1] + [COMPARAR])
    st.caption(MODELO_INFO[nombre_modelo])
    st.divider()
    st.markdown(
        f"**Descriptor:** Color + GLCM "
        f"({bundle['slice_end'] - bundle['slice_start']} características)\n\n"
        f"**Accuracy test:** {bundle['accuracy_test'] * 100:.1f}%\n\n"
        f"**Accuracy CV5:** {bundle['accuracy_cv5'] * 100:.1f}%"
    )
    st.divider()
    st.caption("Consejo: fotos centradas, con fondo uniforme y buena luz.")

st.title("🔬 Clasificador de Minerales")
st.caption("Universidad de Antioquia · Procesamiento Digital de Imágenes\n\n"
           + " · ".join(c.capitalize() for c in clases))

archivos = st.file_uploader(
    "Arrastra o selecciona imágenes de minerales",
    type=["jpg", "jpeg", "png", "bmp"],
    accept_multiple_files=True,
)

if not archivos:
    st.info("Sube una o varias imágenes para clasificarlas.")
    st.stop()

resumen = []
for f in archivos:
    try:
        img = leer_imagen(f)
    except ValueError as e:
        st.error(str(e))
        continue

    if nombre_modelo == COMPARAR:
        t0 = time.time()
        filas = []
        for nombre, m in modelos.items():
            i, p = predecir(img, m, bundle)
            filas.append({"Modelo": nombre,
                          "Predicción": clases[i].capitalize(),
                          "Confianza": f"{p.max() * 100:.1f}%"})
        idx, proba = predecir(img, modelos["Voting Ensemble"], bundle)
        dt_ms = (time.time() - t0) * 1000
        mostrar_resultado(f.name, img, idx, proba, dt_ms, clases,
                          comparar=pd.DataFrame(filas))
    else:
        t0 = time.time()
        idx, proba = predecir(img, modelos[nombre_modelo], bundle)
        dt_ms = (time.time() - t0) * 1000
        mostrar_resultado(f.name, img, idx, proba, dt_ms, clases)

    resumen.append({"Imagen": f.name,
                    "Predicción": clases[idx].capitalize(),
                    "Confianza": f"{proba.max() * 100:.1f}%"})
    st.divider()

if len(resumen) > 1:
    st.subheader("Resumen")
    st.dataframe(pd.DataFrame(resumen), hide_index=True,
                 use_container_width=True)
    conteo = (pd.Series([r["Predicción"] for r in resumen])
              .value_counts().reindex([c.capitalize() for c in clases],
                                      fill_value=0))
    st.bar_chart(conteo)
