# 🔬 Clasificador de Minerales — Tarea 2 Parte 2

**Universidad de Antioquia — Facultad de Ingeniería — Procesamiento Digital de Imágenes — 2026-II**

## Clasificación de 4 minerales (Cuarzo, Pirita, Malaquita, Hematita) usando extracción de características manuales y clasificadores clásicos.

---

## 📊 Resultados principales

| Métrica | Valor |
|---|---|
| **Accuracy test (Voting ensemble)** | **0.833** |
| **F1 macro test** | **0.836** |
| **Accuracy CV 5-fold** | **0.928 ± 0.025** |
| **Mejor descriptor** | Color+GLCM (38 dims) ⭐ |

**Hallazgo clave:** el descriptor **Color + GLCM** (38 dims) supera a combinaciones con HOG (1835 dims). Los minerales se distinguen mejor por **color y textura global** que por **forma local**.

---

## 🚀 Inicio rápido

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Ejecutar TODO de una vez (Parte 1 filtros + Parte 2 pipeline completo)
#    Descarga el dataset de HuggingFace automáticamente si hace falta.
python ejecutar_todo.py

# Opciones del orquestador:
python ejecutar_todo.py --solo-parte1   # solo taller de filtros (Módulos A–E)
python ejecutar_todo.py --solo-parte2   # solo pipeline del clasificador
python ejecutar_todo.py --solo-app      # solo la interfaz web
python ejecutar_todo.py --app           # pipeline y al final abre la interfaz web

# 3. Predecir sobre una imagen
python src/predecir.py mi_foto.jpg

# 4. Demo rápido
python src/predecir.py --demo

# 5. Interfaz web (subir imágenes, elegir modelo)
python -m streamlit run app.py
```

Alternativa por pasos (Linux/Git Bash): `./run.sh` ejecuta solo la Parte 2.
En Windows usar siempre `python -m streamlit` (el ejecutable `streamlit` puede no estar en el PATH).

Tiempo del pipeline completo: ~3–5 min (la primera descarga del dataset toma más, ~7 GB temporales que se limpian solos).

---

## 📂 Estructura del proyecto

```
trabajo2/
├── README.md                       # este archivo
├── app.py                          # ⭐ interfaz web (Streamlit)
├── run.sh                          # ⭐ pipeline end-to-end (10 pasos)
├── requirements.txt                # dependencias
├── INSTALACION.md                  # guía de instalación
│
├── src/                           # código fuente (Python)
│   ├── clasificador.py             # ⭐ extractor de features (1835 dims)
│   ├── predecir.py                 # CLI para predicciones
│   ├── guardar_modelo.py           # serializa voting ensemble
│   ├── extraer_imagenes.py         # descarga dataset HuggingFace
│   ├── aumentar_datos.py           # augmentation para fotos caseras
│   ├── agregar_fotos_propias.py    # añade fotos del celular al test
│   ├── experimentos.py             # comparación E1..E5
│   ├── validacion_cruzada.py       # CV 5-fold
│   ├── gridsearch.py               # GridSearchCV (RF + MLP)
│   ├── ensemble.py                 # voting ensemble (RF+MLP+SVM)
│   ├── curvas_aprendizaje.py       # learning curves
│   ├── galeria_errores.py          # galería FP/FN
│   ├── demo.py                     # demo visual de predicción
│   ├── build_notebook_clasificador.py  # genera el notebook
│   └── generar_informe.py          # ⭐ genera informe IEEE (DOCX/HTML/PDF)
│
├── notebook/                      # Jupyter notebooks
│   └── pipeline_clasificador.ipynb # ⭐ 31 celdas ejecutables
│
├── data/                           # dataset (regenerable con src/extraer_imagenes.py)
│   ├── README.md                   # descripción del banco
│   └── dataset/                    # imágenes (excluidas del repo por tamaño)
│       ├── train/  (90 imgs/clase)
│       ├── val/    (9 imgs/clase)
│       └── test/   (9 imgs/clase + N propias)
│
├── model/                         # modelos serializados
│   └── modelo_minerales.pkl        # ⭐ Voting ensemble (1.1 MB)
│
├── docs/                           # documentación
│   ├── TAREA_2_FILTROS_Y_CLASIFICACION_2026-2.odt  # enunciado original
│   ├── informe_ieee.pdf            # ⭐ informe final (6 páginas, TNR 12)
│   ├── informe_ieee.docx           # editable
│   ├── diapositivas.md             # slides (8 diapositivas)
│   ├── PREGUNTAS_CONCEPTUALES.md   # respuestas 2.1, 2.2, 2.3, 2.4
│   ├── TOMAR_FOTOS.md              # guía para fotos con celular
│   ├── INSTALACION.md              # instalación detallada
│   └── MODELO_README.md            # cómo usar el .pkl en la GUI
│
├── results/                        # resultados generados
│   ├── figures/                    # 8 figuras PNG
│   └── tables/                     # 6 tablas CSV
│
└── tests/                          # (futuro: tests unitarios)
```

---

## 🔬 Pipeline técnico

### 1. Preprocesamiento (`src/clasificador.py`)
- Resize bicúbico 128×128
- GaussianBlur 5×5 σ=1.0
- CLAHE en L de LAB (clip=2.0, tile=8×8)
- Segmentación Otsu + apertura/cierre morfológico 3×3

### 2. Descriptores (`extract_features`, **1835 dims**)

| Familia | Dims | Notas |
|---|---|---|
| HOG | 1764 | orientations=9, cells=16×16, blocks=2×2 |
| Color HSV | 9 | μ, σ, **skew** por canal (H, S, V) |
| Color LAB | 9 | μ, σ, **skew** por canal (L, a, b) |
| Gabor | 16 | 4θ × 2λ × 2 (mean + std) |
| LBP uniforme | 10 | P=8, R=1, histograma de 10 bins |
| **GLCM / Haralick** | 20 | **5 props × 4 dirs** |
| **Hu Moments** | 7 | **invariantes a traslación/escala/rotación** |
| **Total** | **1835** | |

### 3. Clasificadores comparados
- **SVM** (RBF, C=10, γ=scale)
- **k-NN** (k=5)
- **Random Forest** (n=100, max_depth=10 después de GridSearch)
- **MLP** ((128, 64, 32) con GridSearchCV)
- **Voting ensemble** ⭐ (RF + MLP + SVM, soft voting) — **modelo final**

### 4. Métricas
- Accuracy, F1 macro, classification report
- Matriz de confusión normalizada
- ROC one-vs-rest + micro-promedio
- Validación cruzada 5-fold
- GridSearchCV para hiperparámetros
- Curvas de aprendizaje
- Galería visual de errores

---

## ⚠️ Estado vs requisitos del taller (Parte 2)

| Requisito | Estado |
|---|---|
| ≥4 clases | ✅ |
| ≥60 imgs/clase | ✅ |
| División 70/15/15 | ✅ |
| Pipeline preproc + segmentación | ✅ |
| HOG + Color + Gabor + LBP + **GLCM** | ✅ |
| SVM + k-NN + RF + MLP | ✅ |
| Validación cruzada 5-fold | ✅ |
| Experimentos E1..E5 separados | ✅ |
| Matriz confusión + ROC + curvas aprendizaje | ✅ |
| Galería FP/FN | ✅ |
| Importancia de características (RF) | ✅ |
| Preguntas 2.1–2.4 | ✅ (`docs/PREGUNTAS_CONCEPTUALES.md`) |
| README del banco | ✅ (`data/README.md`) |
| **Informe IEEE** (≤6 págs) | ✅ (`docs/informe_ieee.pdf`, **6 págs, TNR 12**) |
| Diapositivas (≤8) | ✅ (`docs/diapositivas.md`) |
| **Modelo .pkl serializado** | ✅ (`model/modelo_minerales.pkl`) |
| Fotos propias con celular | ⏳ **Pendiente tuyo** |
| ≥10 fotos de escena | ⏳ **Pendiente tuyo** |
| Interfaz gráfica | ✅ (`app.py`, Streamlit) |

---

## 📝 Pendientes (para ti)

1. **Comprar** los 4 minerales (cuarzo, pirita, malaquita, hematita)
2. **Tomar fotos** siguiendo `docs/TOMAR_FOTOS.md`
3. **Agregar fotos al test set:**
   ```bash
   python3 src/agregar_fotos_propias.py /ruta/a/tus/fotos/
   ```
4. **Re-entrenar:**
   ```bash
   ./run.sh
   python3 src/guardar_modelo.py
   ```
5. **Compartir el modelo** (`model/modelo_minerales.pkl`) con tu compañero para la GUI

## 🔄 Interfaz gráfica

Implementada en `app.py` con **Streamlit**:

```bash
streamlit run app.py
```

- Carga `model/modelo_minerales.pkl` (ver `docs/MODELO_README.md`)
- Permite subir una o varias imágenes (jpg, png, bmp)
- Selector de modelo: Voting Ensemble, Random Forest, MLP, SVM (los 3 últimos se extraen ya entrenados del ensemble) o comparación de los 4
- Muestra predicción, barras de probabilidad por clase, preprocesamiento (CLAHE/gris/máscara Otsu) y resumen en lote

---

## 🔧 Reproducibilidad

```bash
# Setup inicial
pip install -r requirements.txt

# Pipeline completo
./run.sh

# Verificar resultados
cat results/tables/resultados_final.csv
```

---

## 📚 Referencias

- González & Woods, *Digital Image Processing*, 4.ª ed.
- [tedqc/mineral-dataset (HuggingFace)](https://huggingface.co/datasets/tedqc/mineral-dataset)
- scikit-learn: https://scikit-learn.org/
- scikit-image: https://scikit-image.org/