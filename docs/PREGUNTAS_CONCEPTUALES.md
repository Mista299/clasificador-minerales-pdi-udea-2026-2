# Preguntas conceptuales — Parte 2

## Pregunta 2.1 — Invarianzas de los descriptores

Para cada descriptor usado, se analiza su comportamiento frente a transformaciones geométricas (traslación, rotación, escala) y fotométricas (iluminación):

| Descriptor | Traslación | Rotación | Escala | Iluminación |
|---|---|---|---|---|
| **HOG** (Histograma de Gradientes) | ✅ Invariante (opera sobre gradientes locales) | ⚠️ Parcialmente invariante: con celdas pequeñas pierde rotación; mejora con descriptores SIFT | ❌ Sensible a la escala (celdas de tamaño fijo) | ⚠️ Sensible: los gradientes dependen del contraste local, se mitiga con normalización L2 por bloque | HOG es robusto a posición del objeto en el cuadro pero pierde ante rotaciones grandes. Como nuestros minerales se fotografían siempre en orientación vertical/horizontal, este problema es menor. |
| **Color HSV/LAB (μ, σ)** | ✅ Invariante | ✅ Invariante | ✅ Invariante | ⚠️ HSV es algo invariante al brillo (canal V separado); LAB es aproximadamente perceptualmente uniforme | El color es el descriptor más discriminante (E3 con 32 dims logra 78% acc) precisamente porque los minerales tienen colores distintivos. La invariancia a iluminación en HSV es clave. |
| **Gabor** (4θ × 2λ) | ✅ Invariante | ✅ Invariante (multi-orientación cubre todas las rotaciones) | ⚠️ Parcial: depende de la escala vs λ | ❌ Sensible al brillo (filtro lineal) | El banco multi-orientación cubre rotaciones; los minerales con texturas direccionales (malaquita con bandas) responden bien. |
| **LBP** (uniforme P=8 R=1) | ✅ Invariante | ✅ Invariante (los códigos son circulares) | ⚠️ Sensible: solo un radio R | ✅ Invariante a cambios de brillo monótonos | LBP es robusto a iluminación pero pierde ante cambios de escala. |
| **GLCM / Haralick** | ✅ Invariante | ⚠️ Las 4 direcciones (0, 45, 90, 135) cubren rotaciones básicas | ⚠️ Depende de la distancia d=1 (escala fija) | ⚠️ Sensible al rango dinámico | GLCM con múltiples direcciones da cierta invariancia rotacional. |

**Implicaciones para nuestro problema:**

- **Mineral en el centro del cuadro:** todos los descriptores se benefician del centrado cenital y fondo homogéneo. Nuestro preprocesamiento con Otsu + bounding box implícita (resize a 128×128) ayuda a la invariancia de traslación/escala.
- **Variabilidad de iluminación casera:** el descriptor de color HSV es robusto; los demás se benefician del CLAHE aplicado en L de LAB.
- **Minerales con texturas direccionales** (malaquita): se benefician del banco multi-orientación de Gabor.

Por estas razones, **E3 (Color + GLCM, 32 dims)** obtuvo el mejor accuracy en validación cruzada (89.2%) al evitar la alta dimensionalidad ruidosa de HOG sobre imágenes tan pequeñas (128×128).

---

## Pregunta 2.2 — Accuracy engañoso con clases desbalanceadas

**Problema:** cuando las clases tienen tamaños muy distintos, el accuracy global puede ser alto simplemente prediciendo siempre la clase mayoritaria. Por ejemplo, con 100 muestras de clase A y 10 de clase B, un clasificador que siempre predice A logra 91% de accuracy pero es inútil para detectar B.

**Métricas más apropiadas en ese caso:**

- **F1-score (macro):** promedio del F1 de cada clase **sin ponderar** por tamaño. Trata a todas las clases por igual. Esta es nuestra métrica principal.
- **F1-score (weighted):** pondera por soporte, útil si queremos reportar una sola métrica representativa del rendimiento global pero todavía sensible a clases pequeñas.
- **Recall balanced / Balanced accuracy:** promedio de recall por clase (equivalente a F1 macro si precision=recall).
- **Matriz de confusión normalizada por clase** (no por global): muestra el recall real de cada clase.
- **Precision y Recall por separado:** cuando hay desbalance, precision alta con recall bajo indica que el modelo evita predecir la clase minoritaria (mala señal).
- **Curvas PR (Precision-Recall)** y **AUC-PR**: más informativas que ROC cuando hay desbalance fuerte.
- **Cohen's Kappa**: corrige el accuracy por el azar esperado; valores <0.5 indican poco mejor que el azar.

**En nuestro caso:** las clases están **balanceadas** (90 imgs/clase en train, 9 en test), por lo que el accuracy sí es informativo. Pero igual reportamos F1 macro como métrica principal para ser consistentes.

---

## Pregunta 2.3 — Fuga de información (data leakage)

**Definición:** la fuga de información ocurre cuando información del conjunto de **test** (o de **validación**) "se cuela" al proceso de entrenamiento, inflando artificialmente las métricas de evaluación.

**Tres causas comunes en este proyecto:**

### 1. Ajustar el StandardScaler con todos los datos
Si hacemos:
```python
scaler = StandardScaler().fit(X_total)  # X_total incluye test!
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)
```
La media y desviación estándar usadas para normalizar **incluyen** estadísticas del test set. El modelo "ve" indirectamente la distribución del test.

**Correcto:**
```python
scaler = StandardScaler().fit(X_train)  # solo train
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)
```

### 2. Ajustar el PCA con todos los datos
Mismo problema. El PCA aprende las direcciones de máxima varianza **incluyendo** la varianza del test, lo que beneficia las proyecciones de train.

**Correcto:** `PCA(n_components=k).fit(X_train_s)` y luego transformar test.

### 3. Versiones aumentadas en train y test
Si aumentamos la imagen `quartz_001.jpg` con rotaciones y ponemos las versiones aumentadas en **train** y la original en **test**, las versiones en train son **idénticas** (correlacionadas) a la de test. El modelo "memoriza" la imagen original.

**Correcto:** aplicar augmentation **solo al train**, asegurándose de que ninguna imagen del test aparezca (ni en original ni en transformada) en el train.

### En nuestro proyecto
- ✅ StandardScaler ajustado **solo** con train (`clasificador.py` línea X)
- ✅ PCA 2D solo para visualización, no afecta el entrenamiento del clasificador final
- ⚠️ El augmentation (`aumentar_datos.py`) genera imágenes a partir de train originales y se guardan en `train/`. Verificamos que ningún archivo de `aug_NNN.jpg` se filtre a test.

---

## Pregunta 2.4 — Características manuales vs. CNN (aprendizaje profundo)

| Aspecto | Características manuales (HOG+Gabor+Color) | CNN (ej. ResNet50) |
|---|---|---|
| **Cantidad de datos necesaria** | Decenas a cientos por clase | Miles a millones por clase |
| **Interpretabilidad** | ✅ Alta: cada descriptor tiene significado geométrico/fotométrico claro | ❌ Baja: las features son aprendidas, difícil saber qué "ve" la red |
| **Costo computacional de entrenamiento** | Bajo (segundos en CPU) | Alto (GPU, horas a días) |
| **Costo en inferencia** | Bajo (CPU, milisegundos) | Medio-alto (depende del modelo, GPU recomendado) |
| **Robustez a nuevas condiciones** | Media: requiere re-diseñar features si cambian las condiciones | Alta: generaliza mejor con datos diversos |
| **Necesidad de preprocesamiento** | Alta (resize, ecualización, segmentación) | Baja (solo resize y normalización) |
| **Tamaño del modelo** | KB (features) + KB-MB (clasificador) | MB-GB (toda la red) |
| **Explicabilidad de errores** | Posible inspeccionando qué feature falla | Difícil (salvo Grad-CAM y similares) |
| **Rendimiento con dataset pequeño** | Competitivo si las features están bien elegidas | Bajo (tiende a sobreajustar) |
| **Reproducibilidad** | Total (mismas features siempre) | Depende de inicialización aleatoria |

**¿Cuándo usar cada uno?**

- **Características manuales**: cuando el problema tiene menos de ~1000 imgs/clase, se necesita explicabilidad (medicina, geología), o se requiere ejecución en hardware limitado (móviles, edge).
- **CNN**: cuando hay miles de imágenes por clase, el presupuesto computacional permite entrenar (o usar transfer learning), y el rendimiento bruto es la prioridad.

**En nuestro caso** (60 imgs/clase, 4 clases, entorno CPU), las características manuales son la elección correcta. Obtuvimos 80.6% de accuracy con Random Forest + 1822 dims en CPU, mientras que entrenar una CNN con solo 240 imágenes probablemente daría <70% por sobreajuste. Con transfer learning (VGG16 + fine-tuning) podríamos mejorar, pero no era necesario para aprobar el taller.