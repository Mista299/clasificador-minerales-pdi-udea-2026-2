# Explicación resumida de la Parte 1: Módulos A y B

> **Objetivo de este documento:** explicar los conceptos importantes del taller de filtros de la Parte 1, sin entrar en los detalles de implementación. El contenido corresponde a las secciones **2.1–2.3**: convolución, suavizado, realce y detección de bordes.

---

## 1. La idea general

El procesamiento digital de imágenes modifica los píxeles mediante operaciones matemáticas. En este taller se estudian dos familias principales:

1. **Filtros espaciales (Módulo A):** modifican la imagen usando sus píxeles vecinos.
   - **Suavizado:** reducen el ruido, pero también pueden borrar detalles.
   - **Realce:** aumentan el contraste local para hacer la imagen más nítida.

2. **Detección de bordes (Módulo B):** buscan cambios bruscos de intensidad.
   - Se usan la **primera derivada** (gradiente) o la **segunda derivada** (Laplaciano).
   - Los bordes son importantes porque describen la forma y los límites de los objetos.

El proceso general es:

```text
Imagen original → suavizado opcional → detección o realce → resultado
```

Normalmente se suaviza antes de derivar, porque las derivadas amplifican el ruido.

---

# Módulo A. Convolución y filtros de suavizado

## 2. ¿Qué es una convolución?

Una convolución es una operación en la que el valor de cada píxel de salida se calcula como una suma ponderada de sus vecinos. Los pesos los proporciona una matriz llamada **kernel** o **máscara**.

```text
Original:      kernel 3×3:        Resultado:
a b c          1 1 1              nuevo valor en el centro
d e f          1 1 1
g h i          1 1 1
```

El kernel se desliza sobre la imagen. En cada posición, se multiplican los píxeles por los coeficientes del kernel y se suman.

### Conceptos importantes

- **Kernel:** define qué característica se extrae o modifica.
- **Tamaño del kernel:** determina el alcance del filtro. Un kernel grande opera sobre una región mayor.
- **Vecindario:** píxeles que rodean al píxel central.
- **Convolución frente a correlación:** matemáticamente, la convolución voltea el kernel. OpenCV implementa correlación. La diferencia solo importa con kernels asimétricos; en kernels simétricos, como el gaussiano o la media, ambos resultados coinciden.

### Tipos de borde

Al filtrar, los píxeles del borde no tienen vecinos completos. OpenCV debe inventar esos valores:

- **`BORDER_CONSTANT`:** rellena con ceros. Puede crear un marco oscuro.
- **`BORDER_REPLICATE`:** repite el píxel del borde.
- **`BORDER_REFLECT`:** refleja la imagen, como un espejo.

En general, `REFLECT` o `REPLICATE` producen menos artefactos que `CONSTANT`.

---

## 3. ¿Qué es un filtro pasa-bajo?

Un filtro **pasa-bajo** deja pasar las frecuencias bajas y atenúa las altas.

- **Frecuencias bajas:** zonas suaves, grandes cambios de brillo.
- **Frecuencias altas:** ruido, detalles muy finos y texturas.

Los filtros de suavizado son filtros pasa-bajo: reducen el ruido, pero también suavizan los bordes y las texturas.

> **Compromiso fundamental:** un filtro más fuerte elimina más ruido, pero borra más información útil.

---

## 4. Filtros de suavizado

### 4.1 Filtro de media o “box”

Calcula el promedio de los píxeles de una ventana.

```text
Kernel 3×3 de media:

1/9  1/9  1/9
1/9  1/9  1/9
1/9  1/9  1/9
```

- **Ventana pequeña:** conserva más detalle, pero elimina poco ruido.
- **Ventana grande:** reduce más ruido, pero produce más borrosidad.
- Es rápido, aunque puede crear artefactos quadrados.

### 4.2 Promedio ponderado

Usa coeficientes distintos para cada vecino.

```text
1  2  1
2  4  2      dividido entre 16
1  2  1
```

Da más peso al píxel central y a sus vecinos inmediatos. Es una aproximación discreta al filtro gaussiano.

### 4.3 Filtro gaussiano

Asigna pesos siguiendo una campana de Gauss. Los píxeles centrales tienen más peso que los lejanos.

- **σ pequeño:** suavizado suave.
- **σ grande:** más suavizado y más borrosidad.
- Es más natural que la media y produce menos artefactos.
- **Propiedad importante:** es **separable**. Un kernel gaussiano 2D puede calcularse como dos filtros 1D: uno horizontal y otro vertical. Esto reduce mucho el costo computacional.

### 4.4 Filtro de mediana

Ordena los valores de la ventana y escoge el valor central.

```text
Valores: 97, 98, 99, 100, 100, 101, 102, 103, 255
Mediana: 100
```

Es el filtro más adecuado para **ruido de sal y pimienta**:

- La media incluye el valor extremo 255 y produce una mancha.
- La mediana lo descarta porque queda en uno de los extremos.
- Conserva mejor los bordes porque no genera valores intermedios inexistentes.

### 4.5 Filtros de mínimo y máximo

- **Mínimo:** reemplaza el píxel por el menor valor de su vecindario.
- **Máximo:** reemplaza el píxel por el mayor valor.

No son filtros de ruido adecuados:

- El mínimo oscurece y expande regiones oscuras.
- El máximo aclara y expande regiones claras.

Son útiles principalmente para **morfología**, como erosión y dilatación.

### 4.6 Filtro bilateral

Suaviza preservando los bordes. Para cada vecino, pondera según:

1. Su cercanía espacial.
2. Su similitud de intensidad con el píxel central.

Por eso, los píxeles situados al otro lado de un borde tienen peso casi cero y no contaminan el resultado.

**Parámetros:**

- `sigmaColor`: tolerancia a cambios de intensidad. Si es demasiado grande, se comporta como un gaussiano y borra bordes.
- `sigmaSpace`: alcance espacial del suavizado.
- `d`: diámetro máximo del vecindario; limita el alcance de `sigmaSpace`.

Es útil para ruido gaussiano, pero no para sal y pimienta, porque un píxel corrupto es muy diferente de todos sus vecinos y conserva casi todo su peso.

### 4.7 Non-Local Means (NLM)

Compara el vecindario de cada píxel con parches similares en **toda la imagen** y los promedia.

- Preserva mejor las texturas que los filtros locales.
- Es más lento.
- El parámetro `h` controla la intensidad del filtrado: debe ser comparable a la desviación del ruido.
- Valores bajos dejan ruido; valores altos lo eliminan, pero pueden producir un aspecto “plastificado”.

---

## 5. Modelos de ruido estudiados

### Ruido gaussiano

Se suma ruido aleatorio con media cero y desviación estándar `σ`.

- Produce variaciones suaves repartidas por toda la imagen.
- Suele eliminarse bien con promediado: media, gaussiano, bilateral o NLM.
- A mayor `σ`, la imagen pierde más detalle.

### Ruido sal y pimienta

Algunos píxeles se sustituyen por 0 o 255.

- Produce puntos extremos aislados.
- La **mediana** es la mejor opción.
- El promedio solo reparte el error y crea manchas.

### Ruido speckle

Es multiplicativo: la perturbación depende del brillo.

- Las zonas claras se degradan más que las oscuras.
- Se elimina mediante filtros de promediado.

---

## 6. Métricas de evaluación

Se comparan las imágenes filtradas con la imagen original limpia.

### PSNR

*Máxima relación señal a pico*. Mide el error cuadrático entre ambas imágenes.

- **Más PSNR:** menor error.
- Como el error se eleva al cuadrado, los errores grandes penalizan mucho.
- Tiende a favorecer imágenes muy suavizadas.

### SSIM

*Índice de similitud estructural*. Compara luminancia, contraste y estructura.

- Generalmente se parece más a la percepción humana.
- Si una imagen tiene poco ruido pero estructura casi nula, su PSNR puede ser alto y su SSIM bajo.

### Idea central

Un filtro no es automáticamente bueno porque “suavice”. Debe compararse con la imagen sin filtrar y evaluarse considerando tanto PSNR como SSIM.

---

## 7. Concepto clave: normalización del kernel

En una región constante de intensidad `c`, un kernel `h` produce:

```text
Resultado = c × suma de los coeficientes de h
```

Por tanto:

- **Suma = 1:** conserva el brillo.
- **Suma < 1:** oscurece.
- **Suma > 1:** aclara y puede saturar.

Los filtros de suavizado se normalizan para no cambiar la exposición. En cambio, los kernels de derivada suman cero porque deben responder solo a cambios.

---

# Módulo B. Realce y detección de bordes

## 8. ¿Qué es un borde?

Un borde aparece donde la intensidad cambia rápidamente.

Dos formas de detectarlo:

- **Primera derivada:** gradiente.
- **Segunda derivada:** laplaciano.

### Gradiente

```text
Gx = variación horizontal
Gy = variación vertical

Magnitud:   |∇f| = sqrt(Gx² + Gy²)
Dirección:  θ = atan2(Gy, Gx)
```

- `Gx` responde principalmente a bordes **verticales**.
- `Gy` responde principalmente a bordes **horizontales**.
- La magnitud combina ambos.

> **Importante:** las derivadas deben calcularse en punto flotante (`float64`). Si se usan enteros de 8 bits, los valores negativos se convierten en cero y se pierde aproximadamente la mitad de los bordes.

---

## 9. Operadores de gradiente

### Roberts

Usa una ventana 2×2 y diferencias diagonales.

- Muy buena localización.
- Más sensible al ruido.
- Sus direcciones están rotadas 45° respecto a las de Sobel y Prewitt.

### Prewitt

Calcula diferencias horizontales y verticales,Vuega con el vecindario.

- Más robust al ruido que Roberts.
- Detecta bordes de manera sencilla.

### Sobel

Combina diferencia y promediado:

```text
Gx:

-1  0  1
-2  0  2
-1  0  1
```

- Da más peso al píxel central.
- Es más estable que Prewitt.
- Un kernel mayor produce bordes más gruesos y menos ruido, pero pierde detalle fino.

### Scharr

Es una variante optimizada de Sobel.

- Mejor respuesta isotrópica: responde de forma más uniforme a bordes de diferentes orientaciones.
- En OpenCV, equivale a `Sobel` con `ksize=-1`.

### Comparación rápida

| Operador | Ventaja | Desventaja |
|---|---|---|
| Roberts | Localización precisa | Muy sensible al ruido |
| Prewitt | Simple y más estable que Roberts | Menos uniforme que Sobel/Scharr |
| Sobel | Buen equilibrio | Depende del tamaño del kernel |
| Scharr | Mejor isotropía | Coeficientes de mayor escala |

---

## 10. Segunda derivada: Laplaciano, LoG y DoG

### Laplaciano

```text
4 vecinos:        8 vecinos:

 0  1  0          1  1  1
 1 -4  1          1 -8  1
 0  1  0          1  1  1
```

- Detecta cambios en cualquier dirección.
- Su respuesta tiene signo: positiva o negativa a cada lado de un borde.
- El borde se ubica en el **cruce por cero** de la respuesta.
- Es muy sensible al ruido, porque amplifica las altas frecuencias.

### LoG: Laplaciano del Gaussiano

Primero suaviza y después calcula el laplaciano:

```text
LoG = Laplaciano(Gaussiano)
```

- El suavizado elimina parte del ruido antes de derivar.
- `σ` grande: menos ruido, pero bordes menos precisos.
- `σ` pequeño: más detalle, pero más ruido.

### DoG: Diferencia de Gaussianas

Resta dos imágenes suavizadas con diferente `σ`:

```text
DoG = G(σ₁) − G(σ₂)
```

La relación `σ₂ ≈ 1.6σ₁` aproxima bien al LoG.

- Es más barato porque aprovecha la separabilidad del gaussiano.
- Es la base del detector SIFT.

### Idea fundamental

Las derivadas amplifican el ruido porque su ganancia aumenta con la frecuencia. Por eso, **suavizar antes de derivar** es una práctica habitual.

---

## 11. Detector Canny

Canny es un detector de bordes completo. Su implementación tiene cuatro etapas.

### 1. Suavizado gaussiano

Elimina parte del ruido. Importante: `cv2.Canny` no suaviza internamente; hay que hacerlo antes.

### 2. Cálculo del gradiente

Utiliza Sobel para obtener magnitud y dirección.

### 3. Supresión de no máximos

La magnitud del gradiente forma una franja ancha alrededor de cada borde. La NMS conserva únicamente el máximo local en la dirección del gradiente.

**Resultado:** bordes de un solo píxel de grosor y mejor localizados.

### 4. Doble umbral e histéresis

- Píxeles por encima del umbral alto: **bordes fuertes**.
- Píxeles entre el umbral bajo y el alto: **bordes débiles**.
- Un borde débil se conserva únicamente si está conectado a un borde fuerte.

**Resultado:** se eliminan fragmentos aislados de ruido y se cierran pequeños huecos de los contornos.

### Parámetros de Canny

- **σ del suavizado:** controla ruido frente a detalle.
- **Umbral bajo:** determina cuánto puede prolongarse un contorno.
- **Umbral alto:** determina dónde comienza un borde seguro.

Una relación aproximada de `umbral alto : umbral bajo = 2:1–3:1` funciona bien.

La regla de umbrales basados en la mediana permite adaptar Canny a imágenes con distinta exposición, evitando que un umbral fijo funcione mal en imágenes oscuras o claras.

---

## 12. Realce de nitidez: Unsharp masking y High-boost

La idea es separar la imagen en:

- Una versión suavizada.
- Una versión de detalle o alta frecuencia.

```text
Detalle = original − suavizada
Realce  = original + k × detalle
```

### Unsharp masking

- `k = 1`: realce clásico y moderado.
- `k < 1`: efecto sutil.
- `k > 1`: **high-boost**, más agresivo.

### Efectos secundarios

- Un valor grande de `k` amplifica también el ruido.
- Puede producir **halos**: una franja clara y otra oscura a ambos lados del borde.
- Un `σ` pequeño produce halos más estrechos.

---

# Filtros adicionales

## 13. Filtro homomórfico

Se utiliza para corregir iluminación no uniforme.

La imagen se modela como:

```text
Imagen = iluminación × reflectancia
```

La iluminación cambia lentamente y contiene principalmente frecuencias bajas. La reflectancia contiene el detalle y frecuencias altas.

Como el producto no puede separarse directamente con un filtro lineal, el proceso es:

```text
log → FFT → filtro H(u,v) → IFFT → exp
```

En la práctica:

- Se atenúan las frecuencias bajas para reducir la iluminación.
- Se amplifican las frecuencias altas para recuperar el detalle.

**Parámetros:**

- `gammaL`: cuánto se atenúa la iluminación.
- `gammaH`: cuánto se amplifica el detalle.
- `D0`: frecuencia a partir de la cual se considera que empieza el detalle.

Es un preprocesamiento útil antes de segmentar o extraer descriptores.

---

## 14. Máscaras de energía de Laws

Son filtros para analizar textura. Cada máscara se construye como producto exterior de dos vectores 1D:

| Vector | Significado |
|---|---|
| L5 | Nivel o brillo local |
| E5 | Bordes |
| S5 | Manchas |
| R5 | Ondulaciones |

El procedimiento general es:

1. Restar la media local para reducir la influencia de la iluminación.
2. Convolucionar con la máscara.
3. Calcular la energía local de la respuesta.
4. Combinar máscaras simétricas.

Los mapas resultantes distinguen regiones lisas de regiones texturizadas. Pueden combinarse con clustering para segmentar por textura, sin utilizar color.

---

# 15. Resumen de las ideas más importantes

1. **Un kernel es una plantilla que define la operación** sobre cada vecindario.
2. **Suavizar siempre implica un compromiso:** menos ruido, pero posible pérdida de detalle.
3. **El tipo de ruido determina el filtro adecuado.**
   - Gaussiano o speckle: promediado.
   - Sal y pimienta: mediana.
4. **Mediana y media no son equivalentes:** la mediana resiste valores extremos.
5. **Bilateral y NLM preservan bordes y textura**, pero no resuelven bien el ruido impulsivo.
6. **Los kernels de suavizado se normalizan** para conservar el brillo.
7. **Los bordes se detectan mediante derivadas.** Sobel y Scharr usan la primera; Laplaciano, LoG y DoG usan la segunda.
8. **Las derivadas amplifican el ruido.** Por eso se calcula en `float64` y se suaviza previamente.
9. **Canny combina:** suavizado + gradiente + NMS + histéresis.
10. **El realce de nitidez puede producir halos y amplificar el ruido.**
11. **El homomórfico corrige iluminación; Laws extrae textura.**
12. **PSNR mide error; SSIM mide similitud estructural.** Deben interpretarse conjuntamente.

---

## 16. Conexión con la Parte 2

- Los **gradientes** de Sobel y Scharr son la base del descriptor **HOG**.
- Los filtros de **denoising** preparan la imagen antes de extraer descriptors.
- **Canny** puede utilizarse para detectar o aislar contornos.
- **Laws** puede servir como descriptor de textura.
- El **homomórfico** mejora la consistencia cuando las fotos tienen iluminación irregular.
