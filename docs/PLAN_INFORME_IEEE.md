# 📋 Plan detallado — Informe IEEE (≤ 6 páginas)

**Tarea 2 — Procesamiento Digital de Imágenes — Universidad de Antioquia — 2026-II**

---

## Decisión de alcance

- El **informe IEEE** cubre la TAREA COMPLETA (Parte 1 + Parte 2), pero las **únicas preguntas obligatorias explícitas** son **2.1, 2.2, 2.3 y 2.4** (sección "Discusión"), según el enunciado literal:
  > *"Informe en Formato IEEE: artículo técnico de máximo 6 páginas con Resumen, Introducción, Marco Teórico (filtros y descriptores), Metodología, Resultados (tablas de experimentos, matrices de confusión), Discusión con las respuestas 2.1 a 2.4 y Conclusiones."*
- Las **preguntas 1.1–1.9** viven en el notebook de la Parte 1 (entregable aparte), **no** en el IEEE.

---

## 🎯 Especificaciones de formato (IEEE)

| Elemento | Regla |
|---|---|
| Plantilla | **IEEE Conference Paper** (formato doble columna) |
| Tamaño | Carta (8.5" × 11") |
| Fuente base | **Times New Roman 12 pt** |
| Título | TNR 24 pt, centrado, **negrita** |
| Autores | TNR 11 pt, centrado |
| Afiliación | TNR 10 pt, italic |
| **Títulos de sección (I., II., III.…)** | TNR 12 pt, **MAYÚSCULAS, negrita**, centrado |
| **Subtítulos (A., B., C.…)** | TNR 12 pt, **negrita**, cursiva, alineado izquierda |
| Cuerpo | TNR 12 pt, justificado, interlineado 1.0 |
| Figuras / tablas | TNR 10 pt en caption, centradas |
| Referencias | TNR 10 pt, formato IEEE `[1]`, `[2]` |
| **Límite** | **6 páginas** |

> **Tip de diagramación:** el IEEE tiene espacio vertical limitado. Combinar figuras en subplots 2×2 o 1×3 con `matplotlib.gridspec` para entrar en una columna o en la página.

---

## 📐 Estructura página por página

### **Página 1 — Portada + Resumen + Introducción**

#### Título (centrado, TNR 24 pt, negrita)

> **Clasificación Automática de Minerales Mediante Extracción de Características y Aprendizaje Clásico**

#### Autores (centrado)

> [Nombre 1], [Nombre 2], [Nombre 3]
> *Universidad de Antioquia — Facultad de Ingeniería — PDI, 2026-II*

#### **Resumen — Abstract** *(encabezado en **negrita**)*

Párrafo único 150–200 palabras que responda: **qué problema** (4 minerales), **qué método** (preproc → descriptores manuales → SVM/k-NN/RF/MLP/Ensemble), **qué resultado** (**83.3 % test / 92.8 % CV-5**), **qué hallazgo** (descriptor Color+GLCM de 38 dims supera a HOG de 1764 dims).

Palabras clave: *mineral recognition, handcrafted features, HOG, GLCM, voting ensemble*.

#### **I. Introducción**

3–4 párrafos cortos:

1. Contexto: la Tarea 1 trabajó píxeles; la Tarea 2 trabaja descriptores y clasificadores.
2. Motivación: automatizar el reconocimiento visual de minerales (geología, gemología, educación).
3. Objetivos: ≥4 clases, ≥60 imgs/clase, descriptores manuales, comparativa de 4+ clasificadores, app gráfica.
4. Estructura del artículo: secciones II (Marco Teórico), III (Metodología), IV (Resultados), V (Discusión con preguntas 2.1–2.4) y VI (Conclusiones).

---

### **Página 2 — Marco Teórico (filtros + descriptores)**

#### **II. Marco Teórico**

**A. Filtros digitales**

Cubre los filtros de la Parte 1 con cita a González & Woods §3-§4:

- Convolución 2D y modos de borde.
- Suavizado lineal (Gaussiano) y no lineal (mediana, bilateral).
- Detectores de borde (Roberts, Prewitt, Sobel, Laplaciano, LoG, Canny).
- Filtros frecuenciales (Ideal, Butterworth, Gaussiano; *ringing*).
- Banco de filtros direccionales (Gabor multi-orientación).

**B. Descriptores de imagen** — Tabla clave: 7 familias con sus dimensiones.

| Descriptor | Dims | Invarianzas | Uso |
|---|---|---|---|
| HOG | 1764 | Traslación, rotación parcial | E1, E2, E2b, E4 |
| Color HSV (μ, σ, skew) | 9 | Brillo | E3 |
| Color LAB (μ, σ, skew) | 9 | Perceptual | E3 |
| Gabor 4θ × 2λ | 16 | Rotación parcial | E2, E2b, E4 |
| LBP uniforme (P=8, R=1) | 10 | Iluminación | E2, E2b |
| GLCM (5 props × 4 dirs) | 20 | Dirección | E2, E2b, E3 |
| Hu Moments | 7 | Trasl., escala, rot. | E2b, E3b |
| **Total E2b** | **1835** | — | Pipeline completo |

**C. Clasificadores clásicos**

- SVM (RBF, C=10, γ=scale).
- k-NN (k=5, métrica euclidiana).
- Random Forest (n=100, max_depth=10 vía GridSearch).
- MLP ((128,64,32) con Adam, early stopping).
- **Voting ensemble (soft, RF+MLP+SVM)** — modelo final.

> **Mapa de modelos** *(figura/tabla resumen cruzada — ver bloque inferior)*.

> **Figura 1** *(subplot 2×2)*: original, Gauss σ=1, mediana, bilateral. Caption indica parámetros, PSNR/SSIM, y conclusión ("el Gaussiano preserva mejor bordes que la media, y el bilateral preserva más detalle que ambos a costa de velocidad").

---

### **Página 3 — Metodología (banco + preproc + extracción + experimentos)**

#### **III. Metodología**

**A. Banco de imágenes**

- **Dataset público:** `tedqc/mineral-dataset` (HuggingFace), 62,088 imgs, 781 etiquetas, CC0; se filtraron 4 clases (quartz→cuarzo, pyrite→pirita, malachite→malaquita, hematite→hematita). Resolución nativa 689–1024 px. **Cámara/equipo: no documentado por el curador; los JPEG no conservan EXIF** (verificado experimentalmente). Estilo catálogo mineralógico.
- **Fotos propias con celular:** protocoladas para 12 imgs/clase, cenital, cartulina blanca/negra según el mineral. **Estado: pendiente de captura por el grupo.**
- **Distribución final:** 60 originales por clase → 42 train + 9 val + 9 test; train augmentado a 90 imgs/clase con rotación, flip, brillo, contraste, ruido y JPEG-compression.

> **Figura 2** *(subplot 2×4)*: una imagen ejemplo por clase del dataset público + espacio reservado para fotos propias.

**B. Preprocesamiento**

Pipeline justificado por la Parte 1:

1. Resize bicúbico a 128×128.
2. **Gaussiano 5×5 σ=1.0** (justificado por PSNR/SSIM del taller).
3. **CLAHE** sobre L de LAB (clip=2.0, tile 8×8).
4. **Otsu** sobre gris → máscara binaria.
5. **Morfología** apertura + cierre con kernel elíptico 3×3.
6. ROI final + bounding box.

> **Figura 3** *(subplot 1×5)*: original → resize → Gauss → CLAHE → Otsu → ROI.

**C. Extracción de características (1835 dims)**

Recorrido por las 7 familias (mismas que Tabla 1), fórmula o referencia breve para cada una. Estandarización con `StandardScaler().fit(X_train)` — **sin fuga** (sello para la pregunta 2.3).

**D. Diseño experimental (E1–E5)**

| ID | Descriptores | Dims |
|---|---|---|
| E1 | HOG | 1764 |
| E2 | HOG + Color + Gabor + LBP + GLCM | 1828 |
| E2b | E2 + Hu | **1835** |
| E3 | Color + GLCM | **38** ⭐ |
| E3b | Color + GLCM + Hu | 45 |
| E4 | HOG + Gabor | 1780 |
| E5 | Hu solo | 7 |

Cada experimento × 4 clasificadores + ensemble. Justificación de E3: pocas dimensiones evitan sobreajuste con solo 60 imgs/clase.

**E. Métricas**

Accuracy, F1 macro, classification_report, matriz de confusión normalizada, ROC one-vs-rest, CV-5, GridSearchCV, curvas de aprendizaje, galería FP/FN.

---

### **Página 4 — Resultados + figuras + análisis de errores**

#### **IV. Resultados**

**A. Resultados principales**

> **Tabla 1 — Resultados por experimento y clasificador (test set)**
> Filas: E1, E2, E2b, E3, E3b, E4, E5. Columnas: SVM, k-NN, RF, MLP, Ensemble. Celdas: Accuracy / F1 macro. **Resaltar** la mejor combinación (E3 + Ensemble: 0.833 / 0.836).

> **Figura 4** *(subplot 1×2)*: (a) **Matriz de confusión normalizada** del ensemble final; (b) **Curvas ROC** one-vs-rest del ensemble.

**B. Hallazgo clave — E3 gana**

Párrafo argumentando por qué **Color + GLCM (38 dims)** supera a HOG-dominated: a 128×128 el HOG produce histogramas ruidosos por baja densidad de gradientes; los minerales se diferencian mejor por color medio y textura global. Cita experimentos: E3 (38d)=0.836 F1 vs E2b (1835d)=0.737.

**C. Análisis de errores — Galería FP/FN**

> **Figura 5 — Galería de errores**: 9 imágenes mal clasificadas con etiquetas real → predicha. Párrafo interpretando los pares de confusión (pirita↔cuarzo, hematita→cuarzo, etc.).

**D. Importancia de características + PCA**

> **Figura 6** *(subplot 1×2)*: (a) Top-20 features según `RandomForest.feature_importances_` (barplot horizontal); (b) **Proyección PCA 2-D** coloreada por clase (4 colores).

---

### **Página 5 — Discusión con preguntas 2.1–2.4**

#### **V. Discusión**

**A. Validación cruzada y ajuste de hiperparámetros**

> **Tabla 2 — Validación cruzada 5-fold** (E3, media ± std): SVM 0.922±0.040, k-NN 0.761±0.038, RF 0.900±0.030, MLP 0.900±0.032.

> **Tabla 3 — Mejores hiperparámetros (GridSearchCV)**: RF = {max_depth=10, n_estimators=100, max_features='sqrt'}; MLP = {(128,64,32), α=1e-4, lr=0.005}.

> **Figura 7 — Curvas de aprendizaje** *(train vs val)*: identifica leve sobreajuste del MLP y estabilidad del RF.

**B. Respuestas a las preguntas conceptuales 2.1–2.4**

Cada pregunta en **negrita** + párrafo de respuesta (los textos ya están en `docs/PREGUNTAS_CONCEPTUALES.md`, solo se resumen).

**Pregunta 2.1 — Invarianzas de los descriptores**

Tabla 5×5 (descriptor × {traslación, rotación, escala, iluminación}). Conclusión: en este problema la cámara cenital y el resize fijo dan cuasi-invariancia de traslación/escala; HOG+LBP cubren rotación; HSV+LBP cubren iluminación. (Esto explica por qué Color+GLCM funciona bien.)

**Pregunta 2.2 — Accuracy engañoso con clases desbalanceadas**

Párrafo: F1 macro y matriz normalizada por clase son más informativos. En nuestro caso las clases están balanceadas (90 train / 9 test) por lo que el accuracy sigue siendo válido, pero igualmente reportamos F1 macro.

**Pregunta 2.3 — Fuga de información (data leakage)**

Párrafo: tres causas (StandardScaler con todo, PCA con todo, augmentaciones cruzadas train-test). Indicar que en `src/clasificador.py` el `StandardScaler` se ajusta **solo con `X_train`** y se aplica igual a `X_val`/`X_test`; el augmentation de `src/aumentar_datos.py` solo escribe en `train/`. Verificación con `numpy`: no hay imágenes duplicadas entre splits.

**Pregunta 2.4 — Rasgos manuales vs CNN**

Tabla 9 filas (cantidad datos, interpretabilidad, costo, robustez, preproc, tamaño, explicabilidad, rendimiento con pocos datos, reproducibilidad). Conclusión: con 240 imgs totales, los descriptores manuales son la opción correcta; entrenar una CNN daría <70% por sobreajuste; transfer learning (VGG16 + fine-tuning) podría mejorar pero no era necesario.

---

### **Página 6 — Interfaz gráfica + Conclusiones + Referencias**

#### **VI. Aplicación con Interfaz Gráfica**

- App **Streamlit** (`app.py`): carga `model/modelo_minerales.pkl`, recibe 1 imagen, devuelve clase + barras de probabilidad + visualización del preproc (CLAHE / gris / máscara Otsu).
- Selector de modelo: Voting Ensemble, Random Forest, MLP, SVM o comparación de los 4.
- Procesamiento por lote con resumen de predicciones.
- **Detección múltiple**: tras segmentar la escena con Otsu + contornos, dibuja bounding boxes con clase y confianza.
- **Prueba de robustez**: aplicada sobre las ≥10 imágenes de escena *(pendiente de captura por el grupo; el protocolo y los criterios de evaluación están documentados)*.

> **Figura 8 — Captura de la app**: tres paneles lado a lado — (a) imagen cargada, (b) barras de probabilidad, (c) máscara Otsu + bounding box.

#### **VII. Conclusiones y Trabajo Futuro**

**Conclusiones** (5 viñetas en **negrita**):

1. El sistema clasifica correctamente 30/36 imágenes de test (**83.3 % accuracy, 92.8 % CV-5**).
2. La familia **Color + GLCM (38 dims)** supera a las configuraciones dominadas por HOG.
3. El **Voting ensemble (RF+MLP+SVM)** ofrece el mejor balance sesgo/varianza.
4. La augmentation del train mitiga parcialmente el *domain gap* dataset↔fotos caseras.
5. La arquitectura es **reproducible** (semilla 42), **interpretables** y desplegable en CPU.

**Trabajo futuro** (4 viñetas):

1. Capturar e integrar fotos propias con celular y las ≥10 imágenes de escena.
2. Ampliar a más minerales (galena, magnetita, calcita).
3. Incorporar SIFT/ORB + Bag-of-Visual-Words como descriptor adicional.
4. Comparar con una CNN liviana (MobileNet) por transfer-learning.

#### **Referencias** *(TNR 10 pt, formato IEEE)*

1. R. C. González & R. E. Woods, *Digital Image Processing*, 4.ª ed., Pearson, 2018.
2. R. M. Haralick, K. Shanmugam, I. Dinstein, "Textural Features for Image Classification", *IEEE Trans. SMC*, 1973.
3. T. Ojala, M. Pietikäinen, T. Mäenpää, "Multiresolution Gray-Scale and Rotation Invariant Texture Classification with LBP", *IEEE Trans. PAMI*, 2002.
4. M.-K. Hu, "Visual Pattern Recognition by Moment Invariants", *IRE Trans. IT*, 1962.
5. N. Dalal & B. Triggs, "Histograms of Oriented Gradients for Human Detection", *CVPR*, 2005.
6. C. Cortes & V. Vapnik, "Support-Vector Networks", *Machine Learning*, 1995.
7. L. Breiman, "Random Forests", *Machine Learning*, 2001.
8. tedqc, "Mineral Image Dataset", HuggingFace, 2025. [En línea]. https://huggingface.co/datasets/tedqc/mineral-dataset
9. F. Pedregosa *et al.*, "Scikit-learn: Machine Learning in Python", *JMLR*, 2011.
10. S. van der Walt *et al.*, "scikit-image: image processing in Python", *PeerJ*, 2014.

---

## 🗺️ MAPA DE MODELOS USADOS EN EL INFORME

*(tabla de referencia cruzada — clave para que el revisor ubique cada técnica)*

| # | Técnica | Aparece en | Fig./Tabla | Resultado |
|---|---|---|---|---|
| 1 | Convolución 2D + modos de borde | II-A | Fig. 1 | error ≤ 1e-12 vs cv2 |
| 2 | Filtro Gaussiano (σ=1.0) | II-A, III-B | Fig. 1 | mejor PSNR/SSIM |
| 3 | Filtro Mediana | II-A | Fig. 1 | mejor con S&P |
| 4 | Filtro Bilateral | II-A | — | preserva bordes |
| 5 | Sobel / Scharr | II-A | Fig. 1 | bordes finos |
| 6 | Laplaciano 8-vecinos | II-A | — | cruces por cero |
| 7 | Canny + histéresis | II-A | Fig. 1 | bordes continuos |
| 8 | FFT + fftshift + log | II-A | Fig. 1 | espectro visible |
| 9 | Butterworth LP/HP | II-A | — | sin *ringing* (n≥4) |
| 10 | Banco Gabor 4θ×2λ | II-A, II-B | Fig. 1 | vector 24-d |
| 11 | HOG (skimage) | II-B, III-C | Tabla 1, Fig. 4 | 1764 dims |
| 12 | Color HSV μ/σ/skew | II-B, III-C | Tabla 1 | 9 dims |
| 13 | Color LAB μ/σ/skew | II-B, III-C | Tabla 1 | 9 dims |
| 14 | Gabor multi-orient. | II-B | Tabla 1 | 16 dims |
| 15 | LBP uniforme P=8,R=1 | II-B | Tabla 1 | 10 dims |
| 16 | GLCM (5×4 dirs) | II-B | Tabla 1, Fig. 4 | 20 dims |
| 17 | Hu Moments (7 inv.) | II-B | Tabla 1 | 7 dims |
| 18 | **StandardScaler** fit_train only | III-C, Preg. 2.3 | — | sin leakage |
| 19 | SVM RBF (C=10, γ=scale) | II-C, IV-A | Tabla 1, 2 | 0.833 test (E3) |
| 20 | k-NN (k=5) | II-C | Tabla 1, 2 | 0.417 test E0 |
| 21 | Random Forest (n=100, d=10) | II-C, IV-A | Tabla 1, 2, 3 | 0.778 test (E3) |
| 22 | MLP (128,64,32) | II-C | Tabla 1, 2, 3 | 0.750 test (E3) |
| 23 | **Voting ensemble (soft)** | II-C, IV-A | Tabla 1, Fig. 4 | **0.833 / 0.928 CV** |
| 24 | GridSearchCV | V-A | Tabla 3 | mejores hp |
| 25 | Validación cruzada 5-fold | V-A | Tabla 2 | 0.928 ± 0.025 |
| 26 | PCA 2-D visualización | IV-D | Fig. 6 | embedding 2-D |
| 27 | Curvas de aprendizaje | V-A | Fig. 7 | diagnóstico sesgo/var |
| 28 | Streamlit app + detección múltiple | VI | Fig. 8 | GUI |

---

## ✅ CHECKLIST DE CUMPLIMIENTO DEL ENUNCIADO

Verifico contra lo que pide el enunciado **literalmente** ("Informe en Formato IEEE: artículo técnico de máximo 6 páginas con Resumen, Introducción, Marco Teórico (filtros y descriptores), Metodología, Resultados (tablas de experimentos, matrices de confusión), Discusión con las respuestas 2.1 a 2.4 y Conclusiones"):

| Requisito del enunciado | Sección del plan | Estado |
|---|---|---|
| Resumen | Resumen — Abstract | ✅ |
| Introducción | I. Introducción | ✅ |
| Marco Teórico (filtros) | II.A Filtros digitales | ✅ |
| Marco Teórico (descriptores) | II.B Descriptores de imagen | ✅ |
| Metodología | III.A–E (Banco, Preproc, Extracción, Experimentos, Métricas) | ✅ |
| Resultados: tablas de experimentos | IV.A Tabla 1 | ✅ |
| Resultados: matrices de confusión | IV.A Fig. 4(a) | ✅ |
| Discusión: respuesta 2.1 | V.B Pregunta 2.1 | ✅ |
| Discusión: respuesta 2.2 | V.B Pregunta 2.2 | ✅ |
| Discusión: respuesta 2.3 | V.B Pregunta 2.3 | ✅ |
| Discusión: respuesta 2.4 | V.B Pregunta 2.4 | ✅ |
| Conclusiones | VII. Conclusiones y Trabajo Futuro | ✅ |
| Máximo 6 páginas | — | ✅ (ajustable con subplots compactos) |
| **TNR 12 + negritas** | plantilla IEEE | ✅ |

---

## 📅 Cronograma de ejecución

### Paso 1 — Material disponible *(ya hecho)*

- ✅ Pipeline Parte 1 + Parte 2 ejecutado.
- ✅ Tablas CSV en `results/tables/`.
- ✅ Figuras en `results/figures/` + `results/figures_taller/`.
- ✅ Preguntas 2.1–2.4 redactadas en `docs/PREGUNTAS_CONCEPTUALES.md`.
- ✅ Informe previo `docs/informe_ieee.pdf` (6 págs) como base.

### Paso 2 — Reorganización

1. Crear subplots compactos con `matplotlib.gridspec` para entrar en 6 págs:
   - Filtros (4-6 paneles por figura).
   - Pipeline preproc (5 paneles en 1 fila).
   - Matriz confusión + ROC (1×2).
   - Importancia + PCA (1×2).
   - Curvas aprendizaje (1 figura).
   - App GUI (3 capturas).
2. Reemplazar el `informe_ieee.docx` actual por una versión alineada con este outline.
3. Compactar texto (eliminar redundancias).

### Paso 3 — Formato IEEE

1. Aplicar plantilla IEEE oficial (LaTeX `IEEEtran.cls` o Word "IEEE Conference Paper").
2. TNR 12 pt en todo el cuerpo.
3. **Negrita** en títulos (I., II., …) y subtítulos (A., B., …).
4. Generar `.docx` y `.pdf`.

### Paso 4 — Verificación final

- [ ] ≤6 páginas.
- [ ] Tablas y figuras todas referenciadas en el texto.
- [ ] Preguntas 2.1–2.4 cada una con **header en negrita** + respuesta de 1 párrafo mínimo.
- [ ] Mapa de modelos presente.
- [ ] Referencias en formato IEEE.