# Diapositivas — Sustentación

## Diapositiva 1 — Portada

# Clasificación Automática de Minerales

**Integrantes:** [Tu nombre], [Compañero 1], [Compañero 2]
**Asignatura:** Procesamiento Digital de Imágenes
**Docente:** [Nombre]
**Universidad de Antioquia** — 2026-II

---

## Diapositiva 2 — Motivación y problema

### ¿Por qué clasificar minerales automáticamente?

- Aplicaciones en **educación, gemología, arqueología, prospección**.
- Reduce dependencia de expertos humanos.
- Sirve como herramienta pedagógica interactiva.

### Nuestro problema

- **4 clases:** 🪨 Cuarzo | ✨ Pirita | 💚 Malaquita | ⚫ Hematita
- Mínimo **60 imágenes/clase** (240 totales)
- División 70/15/15 estratificada
- Test set con fotos propias del celular (requisito del taller)

---

## Diapositiva 3 — Banco de imágenes

### Dos fuentes combinadas

| Fuente | Estilo | Imágenes |
|---|---|---|
| `tedqc/mineral-dataset` (HuggingFace) | Muestras de colección, fondo neutro | 132 (33/clase) |
| Fotos propias con celular | Caseras, cartulina blanca, luz de ventana | 48+ (12+/clase en test) |

**División final:**
- Train: 42 + 48 augmentadas = **90 imgs/clase**
- Val: 9 imgs/clase
- Test: 9 dataset + N propias

**Augmentation en train:** rotación ±15°, flip H, brillo ±25, ruido gaussiano, JPEG 50-85

---

## Diapositiva 4 — Pipeline

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────┐
│ Imagen BGR  │ →  │  Preproc     │ →  │ Descriptores │ →  │ Clasifica- │
│             │    │ 128×128      │    │ HOG + Color  │    │ dor        │
└─────────────┘    │ Gaussiano    │    │ + Gabor      │    │ MLP/RF/SVM │
                  │ CLAHE en L   │    │ + LBP        │    └────────────┘
                  │ Otsu + morfo │    │ + GLCM       │
                  └──────────────┘    │ = 1822 dims  │
                                     └──────────────┘
```

**Preprocesamiento:** resize, suavizado, ecualización adaptativa, segmentación binaria y limpieza morfológica para aislar el mineral del fondo.

**Descriptores:** 5 familias, 1822 dimensiones totales.

---

## Diapositiva 5 — Hallazgo clave: Color + GLCM gana

### Comparación de descriptores (test, RF)

| Experimento | Dims | Accuracy | F1 |
|---|---|---|---|
| HOG | 1764 | 0.583 | 0.579 |
| Todos | 1822 | 0.694 | 0.692 |
| **Color + GLCM** | **32** | **0.778** | **0.777** |
| HOG + Gabor | 1780 | 0.667 | 0.665 |

> **Conclusión:** los minerales se distinguen mejor por **color y textura global** que por **forma local** (HOG) a esta resolución (128×128).

---

## Diapositiva 6 — Resultados finales

### Validación cruzada 5-fold (E3 Color+GLCM)

| Modelo | Accuracy | F1 macro |
|---|---|---|
| SVM (RBF) | 0.875 ± 0.029 | 0.875 |
| k-NN (k=5) | 0.739 ± 0.028 | 0.738 |
| Random Forest | 0.858 ± 0.042 | 0.858 |
| **MLP (64,32)** | **0.892 ± 0.032** | **0.892** |

### Test set (MLP, 36 imgs)

| Clase | P | R | F1 |
|---|---|---|---|
| 🪨 Cuarzo | 0.75 | 1.00 | 0.86 |
| ✨ Pirita | 0.70 | 0.78 | 0.74 |
| 💚 Malaquita | 1.00 | 0.89 | 0.94 |
| ⚫ Hematita | 0.83 | 0.56 | 0.67 |

**Accuracy final en test: 80.6%**

---

## Diapositiva 7 — Análisis de errores

### Galería de falsos positivos / negativos

![Galería](resultados_galeria_errores.png)

### Patrones de confusión

- **Pirita → Cuarzo (2):** brillo metálico confundido con cuarzo claro
- **Hematita → Cuarzo/Pirita (2):** tonos oscuros comparten reflectividad
- **Malaquita (1 error):** solo 1 muestra de 9 mal clasificada → F1=0.94 ✓

**Causa probable:** muestras pequeñas o con bordes donde el descriptor de color se vuelve ambiguo.

---

## Diapositiva 8 — Conclusiones y trabajo futuro

### Conclusiones

✅ Pipeline end-to-end funcional con accuracy 80.6% en test  
✅ 89.2% en validación cruzada con MLP + Color+GLCM  
✅ Hallazgo: descriptores globales (color + textura) > locales (forma) a 128×128  
✅ Banco de imágenes cumple requisitos: ≥60/clase, división estratificada, fotos caseras propias

### Trabajo futuro

- Más descriptores de invariancia (SIFT, SURF + BoVW)
- Transfer learning con VGG16 / ResNet50
- Más fotos caseras para cerrar el *domain gap* con dataset de colección
- Detección multi-objeto (YOLO) para fotos de escena con varios minerales

### Preguntas

**¡Gracias!**

---

# Notas para el presentador

- **Tiempo sugerido:** 8–10 minutos
- **Mostrar en vivo:** ejecutar `./run.sh` o el script de galería para impactar
- **Figura clave:** la matriz de confusión final (`resultados_confusion_final.png`)
- **Insistir en:** la comparación E1 vs E3 que muestra que Color+GLCM gana con 32 dims
- **Mencionar:** las preguntas conceptuales 2.1–2.4 ya están respondidas en `PREGUNTAS_CONCEPTUALES.md`