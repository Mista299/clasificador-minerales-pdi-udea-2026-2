# Parte 2: proyecto de detección y clasificación automática de minerales

> **Documento explicativo del pipeline completo de la Parte 2.** Describe cómo el sistema procesa una imagen, qué filtros y transformaciones aplica, cómo extrae características, cómo decide a qué mineral pertenece y cómo se evalúa y despliega el resultado.

---

## 1. Objetivo del proyecto

El sistema recibe la fotografía de un mineral y determina si corresponde a una de estas cuatro clases:

| Clase | Características visuales generales |
|---|---|
| **Cuarzo** | Blanco, gris o translúcido; relativamente uniforme. |
| **Pirita** | Amarillo o dorado, con aspecto metálico. |
| **Malaquita** | Verde intenso, a menudo con bandas o vetas. |
| **Hematita** | Gris oscuro o negro, con brillo metálico. |

La clasificación no se realiza mirando directamente los píxeles de la imagen. Primero se transforma cada fotografía en un **vector numérico de características** y luego se entrega ese vector a un clasificador de aprendizaje automático.

```text
Fotografía
   ↓
Preprocesamiento
   ↓
Segmentación
   ↓
Extracción de descriptores
   ↓
Vector de características
   ↓
Clasificador
   ↓
Mineral + probabilidades
```

---

## 2. Estructura general del repositorio

```text
trabajo2/
├── app.py                       Interfaz web Streamlit
├── ejecutar_todo.py             Orquestador de Parte 1 y Parte 2
├── run.sh                       Pipeline de la Parte 2
├── src/
│   ├── clasificador.py          Preprocesamiento y descriptores principales
│   ├── clasificador_v2.py       Versión mejorada con histograma de color
│   ├── extraer_imagenes.py      Descarga y organización del dataset
│   ├── aumentar_datos.py        Augmentation del train set
│   ├── agregar_fotos_propias.py Incorporación de fotos del celular
│   ├── experimentos.py          Comparación de descriptores E1–E5
│   ├── validacion_cruzada.py    Validación cruzada 5-fold
│   ├── gridsearch.py            Búsqueda de hiperparámetros
│   ├── ensemble.py              Voting ensembles
│   ├── pipeline_v3.py           Pipeline avanzado con Voting y Stacking
│   ├── guardar_modelo.py        Serialización del modelo final
│   ├── predecir.py              Predicción desde terminal
│   └── galeria_errores.py       Análisis visual de errores
├── data/dataset/                Imágenes y caché de características
├── model/                       Modelos serializados (.pkl)
├── results/                     Figuras y tablas
└── docs/                        Informes y documentación
```

---

# 3. El banco de imágenes

## 3.1 Origen de los datos

El banco combina dos fuentes:

1. **Dataset público `tedqc/mineral-dataset` de HuggingFace.**
   - Se descargan los archivos parquet del conjunto.
   - Se filtran las cuatro clases de interés:
     - `quartz` → cuarzo.
     - `pyrite` → pirita.
     - `malachite` → malaquita.
     - `hematite` → hematita.
   - Las imágenes se convierten a RGB y JPEG con calidad 92.

2. **Fotos propias tomadas con celular.**
   - Simulan el dominio real de uso.
   - Se recomienda un protocolo de captura:
     - Cámara cenital.
     - Mineral centrado.
     - Fondo blanco o negro mate.
     - Luz natural indirecta.
     - Distancia aproximada de 25–30 cm.
   - Se incorporan al conjunto de prueba mediante:

   ```bash
   python src/agregar_fotos_propias.py /ruta/a/mis/fotos/
   ```

   Las subcarpetas pueden llamarse en español o inglés:

   ```text
   fotos/
   ├── cuarzo/    o quartz/
   ├── pirita/    o pyrite/
   ├── malaquita/ o malachite/
   └── hematita/  o hematite/
   ```

## 3.2 División estratificada

Las imágenes se separan en tres conjuntos:

| Conjunto | Propósito | Distribución original |
|---|---|---:|
| **Train** | Entrenar el modelo | 42 por clase |
| **Val** | Desarrollo y comparación de experimentos | 9 por clase |
| **Test** | Evaluación final | 9 por clase |

Después de la ampliación mediante augmentation, el train queda con:

```text
90 imágenes por clase
```

Distribución actual verificada:

```text
Train: 360 imágenes
Val:    36 imágenes
Test:   36 imágenes
Total: 432 imágenes
```

### ¿Por qué es importante la separación?

El conjunto de prueba debe contener imágenes que el modelo **nunca haya visto**, ni siquiera como versiones aumentadas. Si una imagen aparece en train y test, el modelo puede memorizarla y las métricas resultan artificialmente altas.

## 3.3 Augmentation o aumento de datos

Para sobrevivir mejor a las diferencias entre fotografías de colección y fotos caseras, el script `src/aumentar_datos.py` genera versiones modificadas de las imágenes originales de train.

Las transformaciones son:

| Transformación | Detalle | Objetivo |
|---|---|---|
| Rotación | Aleatoria entre −15° y +15° | Simular distintas orientaciones. |
| Volteo horizontal | 50% de probabilidad | Simular el mineral colocado al revés. |
| Brillo y contraste | Contraste 0.8–1.2 y brillo ±25 | Simular diferentes luces. |
| Ruido gaussiano | `σ` entre 3 y 12, con 50% de probabilidad | Simular el sensor de un celular. |
| Desenfoque | Kernel 3×3 o 5×5, con 40% de probabilidad | Simular movimiento de la mano. |
| Compresión JPEG | Calidad 50–85, con 30% de probabilidad | Simular el almacenamiento de un celular. |

**Importante:** el augmentation se aplica únicamente al train. Validation y test conservan las imágenes originales.

---

# 4. Pipeline de procesamiento de una imagen

El preprocesamiento está implementado en `src/clasificador.py`, función `preprocess()`.

```text
Imagen BGR original
       ↓
Resize 128×128
       ↓
Suavizado gaussiano 5×5, σ=1.0
       ↓
Conversión a LAB + CLAHE sobre L
       ↓
Conversión a escala de grises
       ↓
Umbralización de Otsu
       ↓
Apertura morfológica
       ↓
Cierre morfológico
       ↓
Máscara binaria del mineral
```

## 4.1 Redimensionamiento

```python
cv2.resize(img_bgr, (128, 128), interpolation=cv2.INTER_AREA)
```

```text
Foto original          Resultado fijo
┌──────────────┐       ┌──────────────┐
│              │       │              │
│   Mineral    │  →    │   Mineral    │
│              │       │   128×128    │
└──────────────┘       └──────────────┘
   tamaño variable          tamaño fijo
```

Todas las imágenes terminan con un tamaño fijo de 128×128 píxeles.

**¿Por qué?**

- Todos los descriptores necesitan dimensiones coherentes.
- Reduce el costo computacional.
- Permite que HOG y Gabor se calculen con la misma configuración.

**¿Por qué `INTER_AREA`?**

Al reducir una imagen, `INTER_AREA` promedia las áreas que corresponden a cada píxel de salida. Produce menos aliasing que un remuestreo lineal simple, especialmente en imágenes con textura.

## 4.2 Suavizado gaussiano

```python
blur = cv2.GaussianBlur(img, (5, 5), 1.0)
```

```text
Kernel 5×5, σ = 1.0

1  4  7  4  1
4 14 20 14  4       Los vecinos centrales pesan más.
7 20 28 20  7  →    Se conservan los bordes y se borra el ruido.
4 14 20 14  4
1  4  7  4  1
```

Se aplica un filtro gaussiano 5×5 con desviación estándar `σ = 1.0`.

### ¿Qué hace?

- Atenúa el ruido de alta frecuencia.
- Elimina píxeles aislados que podrían afectar la segmentación.
- Reduce cambios bruscos y pequeños.

### ¿Por qué se eligió este valor?

La Parte 1 comparó filtros de suavizado mediante PSNR y SSIM. El gaussiano 5×5 con `σ=1.0` presentó el mejor equilibrio entre eliminación de ruido y conservación de detalle.

```text
Kernel muy pequeño        Kernel utilizado        Kernel muy grande
Mucho ruido restante  →   Equilibrio correcto  →  Bordes y textura borrosos
```

### Relación con la Parte 1

Este filtro es la aplicación directa del estudio del Módulo A:

- El gaussiano reduce ruido gaussiano.
- La interpolación `INTER_AREA` realiza un promediado local.
- La comparación PSNR/SSIM de la Parte 1 justifica el parámetro utilizado en la Parte 2.

## 4.3 Conversión a LAB y aplicación de CLAHE

Este es uno de los pasos más importantes del preprocesamiento. La idea general puede resumirse así:

```text
Imagen BGR con iluminación desigual
                 ↓
Separar brillo y color en LAB
                 ↓
Corregir solamente el brillo local con CLAHE
                 ↓
Volver a combinar luminancia corregida + a + b
                 ↓
Imagen BGR con más detalle y mejor equilibrio tonal
```

### 4.3.1 ¿Qué es el espacio LAB?

LAB representa el color usando tres componentes:

```text
L → luminancia: qué tan claro u oscuro es el píxel
a → posición entre verde y rojo
b → posición entre azul y amarillo
```

Una representación simplificada:

```text
                 +a: rojo
                    ↗
                   /
     b− (azul) ◄── L ──► a+ (rojo)
                   \
                    ↘
                 +b: amarillo
```

En la práctica:

| Canal | Qué representa | Ejemplo |
|---|---|---|
| **L** | Brillo o luminancia | Una piedra negra y una blanca tienen L muy diferente. |
| **a** | Eje verde–rojo | La malaquita tiene valores `a` negativos; la hematita puede tener valores positivos. |
| **b** | Eje azul–amarillo | La pirita tiene valores `b` más altos por su tono dorado. |

### ¿Por qué no se corrige directamente RGB?

En RGB, un cambio de iluminación modifica simultáneamente los tres canales. Si una zona está más oscura, cambian R, G y B al mismo tiempo.

En LAB se puede separar el problema:

```text
Cambio de iluminación → se concentra principalmente en L
Color de la sustancia   → se conserva principalmente en a y b
```

Por eso el proyecto:

1. Corrige `L` mediante CLAHE.
2. Mantiene `a` y `b` sin ecualizar.
3. Recombina los tres canales.

Así se mejora el detalle y el brillo sin alterar tanto el tono del mineral.

### 4.3.2 Conversión de BGR a LAB

```python
lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
```

```text
BGR                         LAB

┌─────┬─────┬─────┐          ┌─────┬─────┬─────┐
│  B  │  G  │  R  │   ───►   │  L  │  a  │  b  │
└─────┴─────┴─────┘          └─────┴─────┴─────┘
 color compacto                brillo + color perceptual
```

La imagen deja de organizarse como tres colores primarios y pasa a organizarse como **brillo, tendencia al rojo/verde y tendencia al amarillo/azul**.

### 4.3.3 Separación de los canales

```python
l, a, b = cv2.split(lab)
```

```text
LAB completo
     ↓ split
┌──────────┬──────────┬──────────┐
│    L     │    a     │    b     │
│ Brillo   │ Verde↔Rojo│Azul↔Amar.│
└──────────┴──────────┴──────────┘
     ↓
Solo L continúa hacia CLAHE
```

### 4.3.4 ¿Qué es CLAHE?

**CLAHE** significa **Contrast Limited Adaptive Histogram Equalization**, es decir, *ecualización adaptativa de histograma con límite de contraste*.

Es una operación que mejora el contraste local de una imagen. Funciona en cuatro etapas:

```text
1. DIVIDIR
   La imagen se divide en teselas de 8×8.

2. HISTOGRAMA LOCAL
   Cada tesela calcula su propio histograma de intensidades.

3. ECUALIZAR CON LÍMITE
   Se redistribuyen los niveles de cada tesela.
   clipLimit=2.0 impide que se amplifique demasiado el ruido.

4. INTERPOLAR
   Las teselas se unen suavemente para evitar bordes entre ellas.
```

### 4.3.5 Parámetros utilizados

```python
l_eq = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
).apply(l)
```

| Parámetro | Valor | Significado |
|---|---:|---|
| `clipLimit` | `2.0` | Limita cuánto contraste puede aumentar cada tesela. Evita amplificar demasiado el ruido y los artefactos. |
| `tileGridSize` | `(8, 8)` | Divide la imagen en 64 teselas de 8×8. Una menor cantidad produce correcciones más locales; una mayor produce un efecto más suave. |

> **Nota sobre la escala de OpenCV:** LAB en OpenCV utiliza valores de 8 bits. L representa la luminancia, mientras que a y b se almacenan desplazadas alrededor de 128 para poder representarlas con números sin signo. Esta representación no cambia el funcionamiento conceptual descrito.

### 4.3.6 Reconstrucción de la imagen corregida

```python
img_eq = cv2.cvtColor(
    cv2.merge([l_eq, a, b]),
    cv2.COLOR_LAB2BGR
)
```

```text
Canal L original ──► CLAHE ──► L corregido ┐
                                             ├──► merge ──► LAB ──► BGR corregido
Canal a sin modificar ─────────────────────┤
Canal b sin modificar ─────────────────────┘
```

### Resultado visual esperado

```text
ANTES                              DESPUÉS
┌──────────────────────────┐      ┌──────────────────────────┐
│ Fondo claro              │      │ Fondo con detalle       │
│ Mineral algo oscuro      │  →   │ Mineral con más textura  │
│-textura poco visible     │      │ visible                  │
└──────────────────────────┘      └──────────────────────────┘
```

### ¿Por qué se usa?

- Corrige iluminación desigual.
- Hace visibles detalles en zonas con poco contraste.
- No modifica tanto los canales que contienen el color.
- Evita que una ecualización global sobresature algunas zonas.
- Mejora la consistencia del descriptor de color entre fotos con distinta exposición.

### Relación con la Parte 1

CLAHE es una transformación de intensidad. Su fundamento conceptual se relaciona con:

- Las transformaciones logarítmica y gamma.
- El filtro homomórfico.
- La ecualización de histogramas.

La diferencia es que CLAHE es **local**: adapta el contraste región por región, en lugar de aplicar una curva global a toda la imagen.

## 4.4 Conversión a escala de grises

```python
gray = cv2.cvtColor(img_eq, cv2.COLOR_BGR2GRAY)
```

```text
Imagen BGR corregida
        ↓
  ┌─────┬─────┬─────┐
  │  B  │  G  │  R  │  →  una sola matriz de intensidades
  └─────┴─────┴─────┘
```

La imagen en color corregida se convierte a una sola matriz de intensidades.

Esta versión se utiliza para:

- HOG.
- Gabor.
- LBP.
- GLCM.
- Segmentación por Otsu.

El color original corregido se conserva para los descriptores de color.

## 4.5 Segmentación mediante Otsu

```python
_, mask = cv2.threshold(
    gray, 0, 255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)
```

```text
Gris                         Máscara binaria

┌────────────────────┐       ┌────────────────────┐
│░░░░░░░░░░░░░░░░░░░░│       │                    │
│░░░░░██████░░░░░░░░░│  →    │       ██████       │
│░░░░░██████░░░░░░░░░│       │       ██████       │
│░░░░░░░░░░░░░░░░░░░░│       │                    │
└────────────────────┘       └────────────────────┘
 0 = fondo                  0 = fondo
 255 = mineral              255 = mineral
```

### ¿Cómo funciona Otsu?

Otsu busca el nivel de intensidad `t` que divide mejor la imagen en dos grupos:

- **Primer plano:** píxeles claros, mineral.
- **Fondo:** píxeles oscuros.

El criterio minimiza la varianza dentro de cada grupo y maximiza la separación entre ambos.

Mathematically:

\[
\sigma_B^2(t)=w_0w_1(\mu_0-\mu_1)^2
\]

Otsu selecciona el `t` que maximiza esta varianza entre clases.

### ¿Por qué se usa?

- No requiere fijar manualmente un umbral.
- Funciona bien cuando el objeto y el fondo tienen intensidades bien separadas.
- Es el punto de partida de una segmentación automática.

### Limitación

Si el fondo tiene partes claras o el mineral es muy oscuro, Otsu puede segmentar incorrectamente. Por eso después se aplica limpieza morfológica.

## 4.6 Limpieza morfológica

```python
kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (3, 3)
)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
```

### Apertura

La apertura es una **erosión seguida de una dilatación**.

- Elimina puntos aislados del fondo.
- Conserva regiones grandes y conectadas.
- Reduce falsos positivos caused by noise.

### Cierre

El cierre es una **dilatación seguida de una erosión**.

- Cierra pequeños huecos dentro del mineral.
- Conecta regiones fragmentadas.
- Reduce falsos negativos.

### ¿Por qué un elemento elíptico?

Los elementos circulares o elípticos producen contornos más suaves que un kernel cuadrado, lo cual resulta más adecuado para objetos de forma mineral.

### Relación con la Parte 1

Aquí se aplican los filtros de **mínimo y máximo** estudiados en el Módulo A:

- Erosión = filtro de mínimo.
- Dilatación = filtro de máximo.

En el taller se concluye que el mínimo y el máximo no son buenos filtros de ruido. En morfología, en cambio, son herramientas útiles para modificar la forma y limpiar máscaras.

---

# 5. Cómo se manejan los bordes

El proyecto no utiliza Canny como paso obligatorio del pipeline de inferencia, aunque Canny sí se estudia en la Parte 1. El manejo de bordes se realiza en tres niveles.

## 5.1 Suavizado antes de segmentar

El gaussiano 5×5 reduce las transiciones bruscaspixel a píxel. Esto evita que un ruido aislado genere un borde falso durante la umbralización.

## 5.2 Máscara binaria y operaciones morfológicas

Los bordes de la máscara se modifican mediante apertura y cierre:

- La apertura recorta pequeños artefactos del borde.
- El cierre regulariza huecos y pequeñas interrupciones del contorno.

## 5.3 Momentos de Hu sobre el contorno

Los momentos de Hu se calculan a partir del contorno externo más grande de la máscara:

```python
contours, _ = cv2.findContours(
    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
cnt = max(contours, key=cv2.contourArea)
moments = cv2.moments(cnt)
hu = cv2.HuMoments(moments)
```

### Propiedades de los momentos de Hu

Los siete momentos de Hu describen la forma del contorno y son invariantes a:

- Traslación.
- Escala.
- Rotación.

Por eso son útiles para comparar formas aunque el mineral aparezca con diferente tamaño u orientación.

### Limitación

El momento de Hu no necesita una operación de gradiente, por lo que no necesita que los bordes sean muy detallados. La segmentación y la limpieza morfológica previous son suficientes.

---

# 6. Extracción de características

La función principal es `extract_features()` de `src/clasificador.py`.

## 6.1 Vector completo

```text
HOG 1764 | Color 18 | Gabor 16 | LBP 10 | GLCM 20 | Hu 7
                  Total: 1835 dimensiones
```

El orden es importante porque los experimentos utilizan slices de este vector.

| Posición | Familia | Dimensiones |
|---|---|---:|
| 0–1763 | HOG | 1764 |
| 1764–1781 | Color | 18 |
| 1782–1797 | Gabor | 16 |
| 1798–1807 | LBP | 10 |
| 1808–1827 | GLCM | 20 |
| 1828–1834 | Hu | 7 |

## 6.2 HOG — Histogram of Oriented Gradients

```python
hog(
    gray,
    orientations=9,
    pixels_per_cell=(16, 16),
    cells_per_block=(2, 2),
    feature_vector=True
)
```

### ¿Qué hace?

1. Calcula gradientes de la imagen en escalones de 16×16 píxeles.
2. Determina la orientación de cada gradiente.
3. Construye un histograma de 9 orientaciones por celda.
4. Agrupa celdas en bloques de 2×2 y los normaliza.

\[
\text{HOG} = \text{histograma de orientaciones de gradientes}
\]

### ¿Qué detecta?

- Forma.
- Contornos.
- Estructura local.

### Relación con la Parte 1

HOG utiliza los mismos gradientes que:

- Roberts.
- Prewitt.
- Sobel.
- Scharr.

La diferencia es que HOG no busca un único borde: acumula orientaciones en celdas para construir una descripción de forma.

## 6.3 Color en HSV y LAB

Para cada canal de HSV y LAB se calculan tres statistics:

```text
Media, desviación estándar y skewness
```

### Dimensiones

```text
HSV: 3 canales × 3 estadísticas = 9
LAB: 3 canales × 3 estadísticas = 9
Total: 18
```

### Media

Indica el tono o nivel típico.

### Desviación estándar

Indica cuánto contraste o variación tiene ese canal.

### Skewness

```python
((x - x.mean()) ** 3).mean() / (x.std() ** 3 + 1e-9)
```

Mide la asimetría de la distribución:

- Negativa: más valores bajos.
- Cercana a cero: distribución simétrica.
- Positiva: más valores altos.

Es útil para distinguir distribuciones de color que tienen la misma media.

### ¿Por qué HSV y LAB?

- **HSV** separa el tono H de la saturación S y el brillo V. El canal V separado hace que el sistema sea más tolerante a cambios de iluminación.
- **LAB** representa mejor la percepción humana y separa la luminancia de los componentes de color.

### Relación con la Parte 1

La ecualización CLAHE se aplica antes de extraer el color para reducir la influencia de la iluminación.

## 6.4 Gabor

```python
for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
    for lam in [8, 16]:
        k = cv2.getGaborKernel(
            (31, 31), 3.0, theta, lam, 0.5, 0,
            ktype=cv2.CV_32F
        )
        resp = cv2.filter2D(gray, cv2.CV_32F, k)
        f_gabor.extend([resp.mean(), resp.std()])
```

Cada filtro Gabor es una sinusoida modulada por una envolvente gaussiana:

\[
G(x,y)=\exp\left(-\frac{x^2+y^2}{2\sigma^2}\right)
\cos\left(2\pi\frac{x\cos\theta+y\sin\theta}{\lambda}+\phi\right)
\]

### Configuración

| Parámetro | Valor | Significado |
|---|---|---|
| `theta` | 0°, 45°, 90°, 135° | Orientaciones detectadas. |
| `lambda` | 8 y 16 | Frecuencias espaciales. |
| `sigma` | 3.0 | Ancho de la envolvente. |
| `gamma` | 0.5 | Relación entre anchura y longitud de onda. |
| Kernel | 31×31 | Tamaño de la ventana. |

### Respuesta

Para cada filtro se conservan:

- La media de la respuesta.
- La desviación estándar de la respuesta.

Son **16 dimensiones**: 4 orientaciones × 2 longitudes de onda × 2 estadísticas.

### ¿Qué detecta?

- Texturas direccionales.
- Rayas.
- Bandas.
- Estructuras periódicas.

Por ejemplo, la malaquita puede presentar bandas verdes que producen respuestas 강 in certain orientations.

### Relación con la Parte 1

Gabor fue uno de los módulos explorados en la Parte 1. Su respuesta se puede interpretar como la aplicación de un kernel sinusoidal a diferentes orientaciones y frecuencias.

## 6.5 LBP — Local Binary Pattern

```python
lbp = local_binary_pattern(
    gray, P=8, R=1, method="uniform"
)
hist, _ = np.histogram(
    lbp.ravel(), bins=10, range=(0, 10), density=True
)
```

### ¿Cómo funciona?

Para cada píxel:

1. Se observan sus 8 vecinos.
2. Se compara cada vecino con el píxel central.
3. Se genera un número binario de 8 bits.
4. Ese número se reemplaza por un código uniforme.
5. Se construye el histograma.

### Ventajas

- Invariante a cambios monotónicos de brillo.
- Robusto a pequeñas variaciones de iluminación.
- Describe microtexturas locales.

### Parámetros

- `P=8`: ocho vecinos.
- `R=1`: radio de uno píxel.
- `method="uniform"`: histograma de 10 bins.

## 6.6 GLCM / Haralick

```python
gray_q = (gray // 8).astype(np.uint8)
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
glcm = graycomatrix(
    gray_q,
    distances=[1],
    angles=angles,
    levels=32,
    symmetric=True,
    normed=True
)
```

### ¿Qué es una GLCM?

La **Gray-Level Co-occurrence Matrix** registra cuántas veces aparece un par de intensidades en una relación espacial concreta.

Para una imagen con niveles de gris `i` y `j`:

\[
P(i,j)=
\frac{\text{número de veces que }(i,j)\text{ aparece}}
{\text{total de pares evaluados}}
\]

### Propiedades extraídas

Para cada una de las cuatro direcciones se calculan cinco propiedades:

| Propiedad | Interpretación |
|---|---|
| Contrast | Variaciones de intensidad entre pares. |
| Dissimilarity | Distancia entre niveles de los pares. |
| Homogeneity | Concentración alrededor de la diagonal. |
| Energy | Concentración de la distribución. |
| Correlation | Relación lineal entre intensidades. |

### Dimensiones

```text
4 direcciones × 5 propiedades = 20
```

### Reducción de niveles

La imagen se cuantiza a 32 niveles:

```python
gray_q = (gray // 8).astype(np.uint8)
```

Esto reduce el costo y evita que pequeñas variaciones de ruido produzcan matrices demasiado sensibles.

### Relación con la Parte 1

Las propiedades de Haralick y las máscaras de energía de Laws están relacionadas: ambas buscan cuantificar textura. GLCM utiliza pares de intensidades; Laws utiliza respuestas a máscaras de bordes, manchas y ondulaciones.

## 6.7 Momentos de Hu

Los momentos de Hu son siete números que describen la **forma** de un objeto a partir de su contorno. En este proyecto se calculan sobre el contorno externo más grande de la máscara binaria obtenida mediante Otsu.

```text
Máscara binaria
      ↓
Contorno externo principal
      ↓
Momentos espaciales de la forma
      ↓
7 momentos de Hu: Hu1 … Hu7
```

### ¿Qué representa cada momento?

| Momento | Interpretación aproximada |
|---|---|
| **Hu1** | Momento general de la distribución de la forma. |
| **Hu2** | Simetría axial de la forma. |
| **Hu3** | Simetría respecto a ejes oblicuos. |
| **Hu4** | Combinaciones de simetría y elongación. |
| **Hu5** | Simetría de orden superior. |
| **Hu6** | Combinaciones más complejas de los momentos anteriores. |
| **Hu7** | Momento de mayor orden, muy sensible a detalles pequeños. |

En términos conceptuales:

- **Hu1–Hu3** describen las simetrías más generales del contorno.
- **Hu4–Hu6** combinan esas simetrías con características como la elongación y la distribución de la forma.
- **Hu7** integra información de orden superior y es el más sensible a perturbaciones pequeñas del contorno.

No es necesario memorizar las fórmulas para explicar el uso de Hu en el proyecto: la idea central es que los siete valores describen **simetrías y estructura geométrica** de la silueta del mineral.

### Propiedades de invariancia

Los momentos de Hu son invariantes a:

```text
Traslación:  mover la forma no cambia el descriptor.
Escala:      agrandar o reducir la forma no cambia el descriptor.
Rotación:    girar la forma no cambia el descriptor.
```

Por eso son útiles para comparar dos fotografías del mismo mineral aunque tengan distinto encuadre, tamaño u orientación.

### Transformación logarítmica aplicada

```python
hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
```

Los valores de Hu pueden ser muy pequeños o muy grandes. Se aplica una transformación logarítmica con signo para:

- Comprimir el rango.
- Conservar el signo.
- Evitar que Hu7 domine el entrenamiento.
- Facilitar el entrenamiento de los modelos lineales o neuronales.

---

# 7. ¿Qué descriptores utiliza realmente el modelo final?

Aunque se extraen 1835 características, el modelo principal utiliza solamente el slice **E3**:

```text
Color (18) + GLCM (20) = 38 dimensiones
```

El slice está definido por:

```python
SLICE = slice(1764, 1802)
```

## ¿Por qué se usan solo 38 dimensiones?

Los experimentos demostraron que los descriptores globales de color y textura son más discriminantes para este problema que HOG, Gabor y LBP.

| Experimento | Dimensiones | Accuracy | F1 macro |
|---|---:|---:|---:|
| E1: solo HOG | 1764 | 0.583 | 0.579 |
| E2: HOG + Color + Gabor + LBP + GLCM | 1828 | 0.750 | 0.750 |
| E2b: E2 + Hu | 1835 | 0.750 | 0.737 |
| **E3: Color + GLCM** | **38** | **0.806** | **0.807** |
| E3b: Color + GLCM + Hu | 45 | 0.778 | 0.779 |
| E4: HOG + Gabor | 1780 | 0.722 | 0.723 |
| E5: solo Hu | 7 | 0.333 | 0.235 |

### Interpretación

- HOG aporta muchas dimensiones, pero a 128×128 la información de gradientes locales es ruidosa.
- Los minerales de este conjunto se distinguen principalmente por **color y textura global**.
- Un vector de 38 dimensiones:
  - Reduce el riesgo de sobreajuste.
  - Reduce el costo de entrenamiento.
  - Permite usar SVM, MLP y Random Forest con menor riesgo de sobreajuste.

La conclusión no es que HOG sea inútil, sino que **para este dataset y esta resolución no agrega información útil después de color y GLCM**.

---

# 8. Estandarización

Antes de entrenar:

```python
sc = StandardScaler().fit(X_train)
X_train_s = sc.transform(X_train)
X_test_s = sc.transform(X_test)
```

El escalador transforma cada característica para que tenga:

\[
z=\frac{x-\mu}{\sigma}
\]

### ¿Por qué?

- HOG, GLCM, color y Gabor tienen escalas muy diferentes.
- Sin estandarización, una familia con valores grandes podría dominar el entrenamiento.
- SVM y MLP se benefician especialmente de variables con escala comparable.

### Regla crítica: ajustar solo con train

El `StandardScaler` **nunca debe aprenderse usando train + test**, porque eso constituye fuga de información. El test se transforma usando la media y desviación aprendidas exclusivamente del train.

---

# 9. Clasificadores

## 9.1 Modelos base

### SVM con kernel RBF

```python
SVC(kernel="rbf", C=10, gamma="scale", probability=True)
```

- `C=10`: penaliza los errores de clasificación.
- `gamma="scale"`: define la anchura de la influencia de cada ejemplo.
- `probability=True`: permite obtener probabilidades.

El kernel RBF permite aprender fronteras de decisión no lineales.

### k-Nearest Neighbors

```python
KNeighborsClassifier(n_neighbors=5, metric="euclidean")
```

- Clasifica una imagen según sus cinco vecinos más cercanos.
- No tiene fase de entrenamiento explícita.
- Funciona bien si las características están bien escaladas.

### Random Forest

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
```

- Construye 100 árboles de decisión.
- Cada árbol observa subconjuntos aleatorios de datos y características.
- La predicción se obtiene por votación de los árboles.
- Es resistente al sobreajuste si se limita la profundidad.

### MLP — perceptrón multicapa

```python
MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    max_iter=1000,
    alpha=0.0001,
    learning_rate_init=0.005,
    early_stopping=True,
    random_state=42
)
```

- Tres capas ocultas: 128, 64 y 32 neuronas.
- `early_stopping`: detiene el entrenamiento si la validación no mejora.
- `alpha`: regularización para reducir sobreajuste.

## 9.2 Voting Ensemble

```python
ensemble = VotingClassifier(
    estimators=[
        ("rf", rf),
        ("mlp", mlp),
        ("svm", svm),
    ],
    voting="soft",
    n_jobs=-1
)
```

### Soft voting

Cada modelo devuelve probabilidades. El ensemble calcula el promedio:

\[
P(c|x)=\frac{P_{RF}(c|x)+P_{MLP}(c|x)+P_{SVM}(c|x)}{3}
\]

La clase con mayor probabilidad promedio es la predicción.

### Hard voting

Cada modelo vota por una clase y gana la clase con más votos.

El proyecto compara ambas modalidades y diferentes combinaciones de modelos.

### ¿Por qué combinar modelos?

Los tres modelos tienen forma de observar los datos:

- RF: conjuntos de reglas y subespacios.
- MLP: combinaciones aprendidas de neuronas.
- SVM: fronteras de decisión con kernel.

Cuando coinciden, la decisión suele ser más confiable. El ensemble también reduce la sensibilidad a un modelo individualmente mal ajustado.

## 9.3 Stacking — pipeline avanzado

El script `src/pipeline_v3.py` también implementa:

```python
StackingClassifier(
    estimators=[("rf", rf), ("mlp", mlp), ("svm", svm)],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5,
    passthrough=False
)
```

### Diferencia con Voting

- Voting: promedia las probabilidades directamente.
- Stacking: entrega las predicciones de los modelos base a un meta-aprendiz, en este caso Logistic Regression, que aprende cómo combinar them.

Stacking puede encontrar combinaciones más complejas, pero también puede sobreajustar si no se valida correctamente.

## 9.4 Modelo principal actual

El archivo generado por `src/guardar_modelo.py` es:

```text
model/modelo_minerales.pkl
```

Su contenido principal es:

```python
{
    "modelo": ensemble,
    "scaler": sc,
    "slice": slice(1764, 1802),
    "slice_start": 1764,
    "slice_end": 1802,
    "clases": ["cuarzo", "pirita", "malaquita", "hematita"],
    "labels": [0, 1, 2, 3],
    "n_features": 38,
    "accuracy_test": 0.833,
    "accuracy_cv5": 0.928,
    "modelo_tipo": "VotingClassifier (Soft, RF+MLP+SVM)",
    "version": "2.0"
}
```

El bundle contiene todo lo necesario para inferencia:

1. Modelo entrenado.
2. Escalador.
3. Slice de características.
4. Correspondencia entre labels y clases.
5. Métricas de referencia.

---

# 10. ¿Cómo detecta el sistema un mineral?

El proceso de decisión es el siguiente.

## 10.1 Extraer características

```python
feats_all = clasificador.extract_features(img_bgr)
```

La imagen produce un vector de 1835 valores.

## 10.2 Seleccionar las 38 características relevantes

```python
feats_slice = feats_all[1764:1802]
```

Se conservan Color y GLCM.

## 10.3 Estandarizar

```python
feats_s = bundle["scaler"].transform(
    feats_slice.reshape(1, -1)
)
```

Se aplican la media y desviación estándar aprendidas durante el entrenamiento.

## 10.4 Predecir

```python
idx = bundle["modelo"].predict(feats_s)[0]
proba = bundle["modelo"].predict_proba(feats_s)[0]
```

## 10.5 Traducir el índice a nombre

```python
label = bundle["labels"][idx]
clase = bundle["clases"][label]
```

En la versión actual:

```text
0 → cuarzo
1 → pirita
2 → malaquita
3 → hematita
```

### Ejemplo conceptual

Supongamos que el ensemble devuelve:

```text
cuarzo:   0.82
pirita:   0.11
malaquita: 0.04
hematita:  0.03
```

La predicción es **cuarzo**, con 82% de confianza.

### ¿Qué significa que dos minerales compartan características?

El modelo no busca coincidencias exactas de píxeles. Compara el vector de la nueva imagen con patrones statistics aprendidos durante el entrenamiento.

Por ejemplo:

- La malaquita suele presentar alta saturación verde y una distribución de color particular.
- La pirita suele presentar tonos amarillos y reflejos metálicos.
- El hematita suele ser mucho más oscuro.
- El cuarzo suele ser más claro y uniforme.

GLCM añade la textura: si dos minerales tienen colores parecidos, la estructura espacial de sus tonos puede helping to distinguish them.

---

# 11. Evaluación del modelo

## 11.1 Validación cruzada 5-fold

```python
skf = StratifiedKFold(
    n_splits=5, shuffle=True, random_state=42
)
```

El train set se divide en cinco partes:

```text
Fold 1: se usa fold 1 para validar, folds 2–5 para entrenar
Fold 2: se usa fold 2 para validar, folds 1, 3–5 para entrenar
...
```

Se repite cinco veces y se calcula la media y desviación estándar.

### ¿Por qué es útil?

- Reduce la dependencia de una única partición.
- Mide cómo generaliza el modelo ante distintos subconjuntos.
- Permite comparar modelos y descriptores de forma más confiable.

## 11.2 GridSearchCV

Se buscan los mejores hiperparámetros de RF y MLP. Para RF, por ejemplo:

```text
n_estimators
max_depth
max_features
min_samples_split
```

El modelo se evalúa mediante validación cruzada y se selecciona la configuración con mejor F1.

## 11.3 Accuracy

\[
Accuracy=\frac{\text{predicciones correctas}}{\text{total de predicciones}}
\]

Útil porque las clases están balanceadas.

## 11.4 F1 macro

\[
F1=\frac{2PR}{P+R}
\]

Se calcula por clase y se promedia sin ponderar. Así, una clase con pocas amostras tiene el mismo peso que las demás.

## 11.5 Matriz de confusión

Muestra:

- En la diagonal: predicciones correctas.
- Fuera: confusiones entre clases.

Confusiones frecuentes:

```text
Pirita → Cuarzo
Hematita → Cuarzo/Pirita
```

Las causas probables son:

- Brillo metálico similar.
- Muestras pequeñas o con poco material.
- Colores ambiguos bajo una iluminación deficiente.

## 11.6 ROC y AUC

La curva ROC representa:

```text
TPR = Sensitivity = true positives
FPR = 1 − Specificity
```

El AUC mide el área bajo esa curva:

- `0.5`: igual que el azar.
- `1.0`: separación perfecta.

## 11.7 Galería de errores

Muestra las imágenes mal clasificadas, incluyendo:

- Clase real.
- Clase predicha.
- Confianza de la predicción.
- Probabilidad de la clase verdadera.

Esto permite explicar visualmente los errores.

---

# 12. Resultados actuales

## 12.1 Modelos con todas las características

| Modelo | Accuracy | F1 macro |
|---|---:|---:|
| SVM RBF | 0.611 | 0.602 |
| k-NN | 0.472 | 0.462 |
| Random Forest | 0.750 | 0.737 |
| MLP | 0.639 | 0.639 |

## 12.2 Modelos con E3: Color + GLCM

| Modelo | Accuracy | F1 macro |
|---|---:|---:|
| Random Forest | 0.778 | 0.781 |
| MLP | 0.750 | 0.745 |
| SVM | 0.833 | 0.836 |
| **Voting ensemble** | **0.833** | **0.836** |

## 12.3 Validación cruzada

Para Color + GLCM + Hu:

| Modelo | Accuracy CV-5 | F1 CV-5 |
|---|---:|---:|
| SVM RBF | 0.922 ± 0.040 | 0.922 ± 0.040 |
| k-NN | 0.761 ± 0.038 | 0.761 ± 0.038 |
| Random Forest | 0.900 ± 0.030 | 0.900 ± 0.030 |
| MLP | 0.900 ± 0.032 | 0.900 ± 0.033 |

El ensemble achieves un CV-5 reported of 0.928 ± 0.025.

## 12.4 Desglose del ensemble por clase

| Clase | Precisión | Recall | F1 |
|---|---:|---:|---:|
| Cuarzo | 0.800 | 0.889 | 0.842 |
| Pirita | 0.700 | 0.778 | 0.737 |
| Malaquita | 1.000 | 0.889 | 0.941 |
| Hematita | 0.875 | 0.778 | 0.824 |

La malaquita es la clase más fácil de identificar. Pirita y hematita presentan más confusiones, probablemente por su reflectividad metálica.

---

# 13. Interfaz gráfica

La aplicación se ejecuta con:

```bash
python -m streamlit run app.py
```

## 13.1 Funciones disponibles

- Cargar una o varias imágenes.
- Predecir la clase de cada mineral.
- Mostrar barras de probabilidad.
- Visualizar el preprocesamiento:
  - Imagen con CLAHE.
  - Escala de grises.
  - Máscara de Otsu.
- Comparar varios modelos.
- Procesar imágenes por lotes.

## 13.2 Modelo utilizado

La aplicación prioriza:

```text
model/modelo_minerales_v3.pkl
```

y utiliza como fallback:

```text
model/modelo_minerales.pkl
```

## 13.3 Predicción desde terminal

Para una imagen individual:

```bash
python src/predecir.py foto.jpg
```

Para una carpeta completa:

```bash
python src/predecir.py carpeta_de_fotos/
```

Para la demostración:

```bash
python src/predecir.py --demo
```

La demostración verificada clasificó correctamente las cuatro clases con confianzas superiores al 93%.

---

# 14. Cómo ejecutar todo

## Opción recomendada

```bash
pip install -r requirements.txt
python ejecutar_todo.py
```

Opciones:

```bash
python ejecutar_todo.py --solo-parte1
python ejecutar_todo.py --solo-parte2
python ejecutar_todo.py --solo-app
python ejecutar_todo.py --app
```

## Solo Parte 2

```bash
./run.sh
```

En Windows se recomienda:

```bash
python ejecutar_todo.py --solo-parte2
```

---

# 15. Correlación con la Parte 1

La Parte 1 no está desconectada de la Parte 2: sus resultados justifican decisiones concretas.

| Concepto de la Parte 1 | Aplicación en la Parte 2 |
|---|---|
| Filtro gaussiano y compromiso ruido–detalle | Se elige Gaussian blur 5×5, `σ=1.0`. |
| PSNR y SSIM | Justifican cuantitativamente el parámetro del suavizado. |
| Transformaciones de intensidad | CLAHE corrige iluminación local. |
| CLAHE y filtro homomórfico | Preprocesamiento robusto ante iluminación desigual. |
| Filtro de mínimo y máximo / morfología | Apertura y cierre de la máscara de Otsu. |
| Roberts, Prewitt, Sobel y Scharr | Fundamento de los gradientes utilizados por HOG. |
| Gabor | Banco direccional para medir textura. |
| Laws y GLCM | Representan textura mediante mecanismos diferentes. |
| Borde y Canny | Imágenes con bordes limpios producen mejores descriptores de forma. |

---

# 16. Limitaciones y trabajo futuro

## Limitaciones actuales

1. **Domain gap:** las fotos públicas y las caseras pueden tener iluminación y fondos diferentes.
2. **Tamaño reducido del conjunto:** 360 imágenes de entrenamiento limitan la capacidad de modelos complejos.
3. **Confusiones entre metales:** pirita, hematita y cuarzo pueden compartir brillo.
4. **Fotos propias y de escena pendientes:** deben añadirse para evaluar robustez real.
5. **Dependencia del preprocesamiento:** si Otsu falla con fondos complejos, los descriptores de forma se deterioran.
6. **Un solo modelo por tipo de mineral:** la aplicación actual está orientada a cuatro clases y a un mineral principal por imagen.

## Trabajo futuro

- Integrar fotos propias y escenas con múltiples minerales.
- Usar SIFT u ORB con Bag of Visual Words.
- Evaluar MobileNet u otra CNN ligera mediante transfer learning.
- Mejorar la detección de objetos en escenas mediante YOLO u otro detector.
- Probar modelos de segmentación más robustos que Otsu.
- Añadir más clases y fotos para reducir confusiones.

---

# 17. Resumen final

> El sistema convierte una fotografía de mineral en una predicción mediante cinco etapas: primero normaliza la imagen con resize y suavizado gaussiano; segundo, corrige la iluminación con CLAHE sobre el canal L de LAB; tercero, separa el mineral del fondo mediante Otsu y limpieza morfológica; cuarto, extrae un vector de 1835 descriptores que resumen forma, color, textura direccional, microtextura y momentos de forma; y quinto, normaliza las características y entrega el slice más informativo —Color y GLCM, 38 dimensiones— a un voting ensemble que combina Random Forest, MLP y SVM. El modelo final alcanza 83.3% de accuracy y 0.836 de F1 macro en el conjunto de prueba, y se despliega mediante una interfaz Streamlit que muestra la predicción, las probabilidades por clase y las etapas del preprocesamiento.
