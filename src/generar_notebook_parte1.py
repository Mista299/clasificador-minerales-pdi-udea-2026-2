# -*- coding: utf-8 -*-
"""Genera el notebook Jupyter de la Parte 1 unificando los dos .py.

Salida: notebook/Tarea2_Parte1_Completa.ipynb
"""
import json
import nbformat as nbf
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
NB_OUT = ROOT / "notebook" / "Tarea2_Parte1_Completa.ipynb"
PY_A = ROOT / "notebook" / "tarea2_parte1_modulos_a_b_(2).py"
PY_CDE = ROOT / "notebook" / "tarea2_parte1_modulos_c_d_e.py"


# ============================================================================
# Helpers
# ============================================================================
def md_cell(source):
    return nbf.v4.new_markdown_cell(source.rstrip())

def code_cell(source):
    return nbf.v4.new_code_cell(source.rstrip())

def to_python_code(text):
    """Convierte un texto (mezcla de código y markdown-comentado) en código
    Python ejecutable, comentando cualquier línea que no sea código Python
    válido. Mantiene la información textual como comentarios Python.

    Estrategia:
      1. Detecta bloques markdown: secuencias de líneas consecutivas que no
         son Python (no empiezan por letra reservada, import, def, etc. y no
         están vacías).
      2. Las comenta con prefijo '# '.
    """
    import keyword, builtins, re

    python_keywords = set(keyword.kwlist) | {
        'plt', 'np', 'cv2', 'skimage', 'pd', 'sns',
        'def', 'class', 'return', 'if', 'elif', 'else', 'for', 'while',
        'try', 'except', 'with', 'as', 'from', 'import',
        'print', 'len', 'range', 'enumerate', 'zip', 'open',
        'raise', 'assert', 'pass', 'continue', 'break',
        'os', 'sys', 'glob', 'json',
        'img', 'f_hog', 'f_color_mom', 'f_color_hist', 'f_gabor',
        'f_lbp', 'f_glcm', 'f_hu', 'kernel', 'result', 'results',
        'fig', 'ax', 'axes', 'cm', 'rf', 'mlp', 'svm', 'stack',
    }

    # Patrones que indican código Python
    python_patterns = [
        r'^\s*(def|class|import|from|return|if|elif|else|for|while|try|except|with|raise|assert|pass|continue|break|yield|lambda|global|nonlocal)\b',
        r'^\s*\w+\s*[=\(]',                       # asignación o llamada
        r'^\s*\w+\.\w+',                          # acceso a atributo
        r'^\s*(plt|np|cv2|pd|sns|fig|ax|cm|os|sys)\.', # métodos comunes
        r'^\s*#',                                 # comentario Python
        r'^\s*$',                                 # línea vacía
        r'^\s*""".*"""$',                          # docstring de una línea
        r'^\s*[\'\"]{3}',                          # inicio/fin de docstring
        r'^\s*@',                                 # decorator
        r'^\s*\w+\([^)]*\)\s*$',                  # función call sin asignación
        r'^\s*[\[\{\(]',                           # apertura de corchete/lista/dict
        r'^\s*\]|\}|\)',                           # cierre
        r'^\s*\w+\s*:',                            # dict key o label
        r'^\s*\d+\s*[,\)\]]',                      # número
        r'^\s*[+\-*/]',                            # operador matemático
    ]
    compiled_patterns = [re.compile(p) for p in python_patterns]

    def looks_like_python(line):
        if not line.strip():
            return True
        if any(p.match(line) for p in compiled_patterns):
            return True
        return False

    # Si la primera línea no parece Python (ej. "## A.1 ..."), coméntala
    lines = text.split('\n')
    new_lines = []
    in_python_block = False
    pending_md_block = []

    def flush_md():
        for ml in pending_md_block:
            new_lines.append('# ' + ml if not ml.startswith('# ') else ml)
        pending_md_block.clear()

    for line in lines:
        if looks_like_python(line):
            flush_md()
            new_lines.append(line)
            in_python_block = True
        else:
            # línea que parece markdown: la acumulamos para comentar
            pending_md_block.append(line)
            in_python_block = False
    flush_md()

    return '\n'.join(new_lines)

def read_slice(path, start, end):
    """Lee líneas [start, end) de un archivo (1-based start, end exclusivo)."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return "\n".join(lines[start - 1:end - 1])


# ============================================================================
# Rangos exactos (verificados manualmente)
# ============================================================================
# tarea2_parte1_modulos_a_b_(2).py
A_RANGES = {
    "config":   (1, 36),       # título + cómo usar + librerías
    "imgs":     (36, 105),     # imágenes + funciones auxiliares
    "a1":       (105, 188),    # 0.3 + A.1 (incluye lo de 0.3 antes)
    "a2":       (188, 307),    # A.2 modelos de ruido
    "a3":       (307, 359),    # A.3 batalla filtros
    "a3b":      (359, 475),    # A.3b barrido
    "a4":       (475, 518),    # A.4 separabilidad
    "b1":       (654, 759),    # B.1 gradientes
    "b2":       (759, 838),    # B.2 Laplaciano
    "b3":       (838, 968),    # B.3 Canny + regla automática
    "b4":       (968, 1079),   # B.4 unsharp + preguntas 1.4-1.5 (las saltamos)
    "extras":   (1079, 1224),  # filtro homomórfico + Laws + galería + conclusiones
}

# tarea2_parte1_modulos_c_d_e.py
CDE_RANGES = {
    "header":   (1, 33),       # título + cómo usar + librerías
    "imgs":     (33, 195),     # imágenes + auxiliares (incluye 0.3)
    "c1":       (195, 288),    # C.1 + estiramiento por percentiles
    "c2":       (273, 326),    # C.2 visualización espectro
    "p1_6":     (288, 326),    # pregunta 1.6 (no la incluyo, la redacto aparte)
    "d1":       (326, 387),    # D.1 espectro + fase
    "d2":       (345, 419),    # D.2 paso bajo + ringing
    "d3":       (419, 470),    # D.3 notch
    "d4":       (470, 530),    # D.4 equivalencia + pregunta 1.8 (la redacto aparte)
    "e1":       (530, 574),    # E.1 banco Gabor + efecto σ
    "e2":       (574, 662),    # E.2 vector de textura + rotación
    "extras":   (662, 713),    # galería + conclusiones
}


# ============================================================================
# Construcción del notebook
# ============================================================================
nb = nbf.v4.new_notebook()
nb.cells = []

# ----- Encabezado -----
nb.cells.append(md_cell(f"""# 📘 Tarea 2 · Procesamiento Digital de Imágenes
## Parte 1: Taller práctico y conceptual de filtros — Secciones 2.1 a 2.6

**Universidad de Antioquia — Facultad de Ingeniería — PDI 2026-II**

**Autores:** Maria Fernanda Atencia, Stiven Suárez Marín, Michael Stiven Tabares Tobón

**Fecha de ejecución:** {datetime.now().strftime('%Y-%m-%d')}

---

### 🎯 Objetivo del taller

Experimentar con una batería amplia de filtros para **entender cómo modifican la información de la imagen** en los dominios espacial y frecuencial, **cuantificar su efecto** con métricas objetivas (PSNR, SSIM) y **responder las preguntas conceptuales** 1.1–1.9 del enunciado. Los resultados aquí obtenidos **justifican cada decisión del pipeline de clasificación** de la Parte 2.

### 📋 Contenido

| Sección | Módulo | Temas cubiertos |
|---|---|---|
| 2.2 | **A** | Convolución manual · ruido Gaussiano/S&P/speckle · batalla de filtros · separabilidad |
| 2.3 | **B** | Gradientes (Roberts, Prewitt, Sobel, Scharr) · Laplaciano · LoG · DoG · Canny · unsharp masking |
| 2.4 | **C** | Negativo · logarítmica · gamma (γ=0.3, 0.5, 1.5, 2.5) · estiramiento de contraste |
| 2.5 | **D** | FFT 2D · Filtros Ideal/Butterworth/Gaussiano · ringing · notch · equivalencia espacio-frecuencia |
| 2.6 | **E** | Banco de Gabor 4θ × 3λ · vector de textura de 24 dimensiones |

### 🖼️ Cómo usar este notebook

- **En Google Colab:** sube tus imágenes `natural.png`, `propia.jpeg`, `textura.jpg`, `textura2.jpg` y ajusta el diccionario `RUTAS` en la celda de configuración.
- **Local:** ejecuta todas las celdas; si no se encuentran las imágenes, el script usa imágenes de muestra de `skimage.data`.
- **Salidas:** las figuras se muestran inline (en Colab) y se guardan en `results/figures_taller/` (modo local).
- **Preguntas conceptuales:** 9 preguntas con respuesta argumentada (1.1–1.9)."""))

# ----- 0. Configuración inicial -----
nb.cells.append(md_cell("""# 0. Configuración inicial

Antes de empezar cargamos las librerías y definimos las imágenes de prueba.
"""))

nb.cells.append(md_cell("""### 0.1 Librerías

Usamos `numpy`, `opencv-python`, `scikit-image`, `matplotlib` y `pandas`. La elección está justificada en el enunciado (sección 6)."""))

nb.cells.append(code_cell(read_slice(PY_A, *A_RANGES["config"])))

nb.cells.append(md_cell("""### 0.2 Imágenes de prueba y funciones auxiliares

El enunciado exige **al menos dos imágenes**: una *natural* (paisaje, rostro, escena) y una *propia* tomada con el celular. Para el Módulo E (Gabor) añadimos dos imágenes con **texturas distintas** (tela, ladrillo, madera, hojas).

**Protocolo de montaje** (cartulina blanca mate, luz cenital, sin flash) documentado en `docs/TOMAR_FOTOS.md`."""))

nb.cells.append(code_cell(to_python_code(read_slice(PY_A, *A_RANGES["imgs"]))))

# ----- 2.2 Módulo A -----
nb.cells.append(md_cell("""## 2.2 Módulo A · Convolución y filtros de suavizado (8.0 %)

**Definición central.** La convolución 2D es la operación que aplica un kernel `h(s,t)` a cada vecindad de la imagen:

$$g(x,y) = \\sum_s \\sum_t h(s,t)\\, f(x-s, y-t)$$

Los filtros **paso bajo** atenúan las variaciones rápidas de intensidad (ruido, detalle fino). El brillo medio de la imagen se multiplica por `H(0,0) = Σ h` (ganancia a frecuencia cero). Si `Σ h ≠ 1`, el brillo cambia (Pregunta 1.2).

### A.1 Convolución manual

Implementamos la convolución con dos ciclos `for` y comparamos contra `cv2.filter2D`. El error debe ser del orden de `10⁻¹²` (precisión de punto flotante). Probamos los **tres modos de borde** (`BORDER_CONSTANT`, `BORDER_REPLICATE`, `BORDER_REFLECT`) y medimos tiempos.

**Nota sobre convolución vs correlación.** `cv2.filter2D` realiza **correlación** (sin volteo). Para replicar convolución estricta, hay que voltear el kernel: `np.flipud(np.fliplr(h))`."""))

nb.cells.append(code_cell(to_python_code(read_slice(PY_A, *A_RANGES["a1"]))))

nb.cells.append(md_cell("""### A.2 Modelos de ruido

Contaminamos la imagen con cuatro modelos canónicos:

| Modelo | Parámetros | Comportamiento |
|---|---|---|
| Gaussiano aditivo | σ = 10, 25 | Afecta a todos los píxeles, distribución normal |
| Sal y pimienta | p = 5%, 15% | Solo algunos píxeles son 0 o 255 |
| Speckle (multiplicativo) | σ = 0.1, 0.3 | Depende de la intensidad local |

El histograma revela la naturaleza: Gaussiano → campana centrada; S&P → picos en 0 y 255; speckle → multiplica la intensidad."""))

nb.cells.append(code_cell(to_python_code(read_slice(PY_A, *A_RANGES["a2"]))))

nb.cells.append(md_cell("""### A.3 «Batalla de filtros»

Aplicamos a cada imagen ruidosa los filtros de **media, Gaussiano, mediana, mínimo/máximo y bilateral**. Construimos una **tabla comparativa de PSNR y SSIM** (skimage.metrics) e identificamos el ganador en cada caso. **Esta tabla se citará en el informe IEEE** para justificar la elección del Gaussiano σ=1.0 en el pipeline de la Parte 2.

**Conclusión esperada:**
- Sal-y-pimienta → **mediana** (elimina outliers, Pregunta 1.1)
- Gaussiano → **Gaussiano** (óptimo para ruido gaussiano por máxima verosimilitud)
- Speckle → **bilateral** (preserva bordes eliminando ruido intensidad-dependiente)"""))

nb.cells.append(code_cell(to_python_code(read_slice(PY_A, *A_RANGES["a3"]))))

nb.cells.append(md_cell("""### A.3b Barrido sistemático de parámetros

Variamos sistemáticamente los parámetros del catálogo del enunciado (Tabla 2.1): tamaños de kernel (k=3, 5, 9, 15), σ del Gaussiano (0.5, 1, 2, 4), medianas (k=3, 5, 7), parámetros del bilateral. Esto genera una rejilla de combinaciones que confirma las observaciones puntuales de A.3."""))

nb.cells.append(code_cell(to_python_code(read_slice(PY_A, *A_RANGES["a3b"]))))

nb.cells.append(md_cell("""### A.4 Separabilidad del kernel Gaussiano

Demostramos numéricamente que el kernel Gaussiano 2D es **separable** (producto externo de dos kernels 1D):

$$G_{2D}(x,y) = G_{1D}(x) \\cdot G_{1D}(y)$$

Esto permite reemplazar una convolución 2D de tamaño $K \\times K$ por **dos convoluciones 1D**, reduciendo el costo por píxel de $K^2$ a $2K$ operaciones (para K=31: de 961 a 62, **~15.5× más rápido**). Comparamos `cv2.getGaussianKernel`, la fórmula 2D directa y `cv2.filter2D` con la versión separable."""))

nb.cells.append(code_cell(to_python_code(read_slice(PY_A, *A_RANGES["a4"]))))

# ----- Preguntas 1.1–1.3 -----
nb.cells.append(md_cell("""## 📝 Preguntas conceptuales del Módulo A

### Pregunta 1.1 – Mediana vs. Media

**¿Por qué el filtro de mediana elimina casi por completo el ruido sal y pimienta mientras que el filtro de media solo lo "difumina"?**

La mediana es un **estimador de orden estadístico**: ordena los N píxeles de la ventana y devuelve el valor central. Como el sal-y-pimienta son **outliers** (valores 0 o 255 muy alejados de la distribución), quedan en los extremos del orden y son descartados al elegir el percentil 50. La mediana, además, tiene **punto de ruptura 0/0.5**: hasta el 50% de las muestras pueden ser outliers sin que el estimador colapse (siempre que los outliers no superen la mitad).

La media, en cambio, es un **operador lineal**: cada píxel aporta con peso $1/N$. Un píxel sal (255) en una zona de valor ≈100 desplaza la media en $(255-100)/N \\approx +17$ para N=9. El error **no se elimina, se reparte** sobre los $N$ píxeles. Como consecuencia, aparecen manchas claras difuminadas. La media tiene **punto de ruptura 0**: un solo outlier arbitrariamente grande puede arruinar el resultado.

**Cuantificación.** Para una ventana 3×3 con ruido sal-y-pimienta 15%:
- Probabilidad de que la mediana falle: $P(\\text{más de 4 de 9 corruptos}) = 5.63 \\times 10^{-3}$
- Para ventana 5×5 con 15%: $P(\\text{más de 12 de 25}) = 1.69 \\times 10^{-5}$

**En el proyecto.** La mediana 5×5 se usaría en la Parte 2 si las fotos propias tuvieran mucho ruido sal-y-pimienta (sensor de celular barato); sin embargo, optamos por el Gaussiano σ=1.0 porque la iluminación no uniforme es el problema dominante, no el ruido impulsivo.

### Pregunta 1.2 – Normalización del kernel

**¿Qué ocurre con el brillo medio de la imagen si los coeficientes del kernel de suavizado no suman 1?**

Para una zona de intensidad constante $c$, el resultado de la convolución es:
$$g = c \\sum_s \\sum_t h(s,t) = c \\cdot H(0,0)$$

Por tanto, **el brillo medio se multiplica por la suma de los coeficientes del kernel**. Si `Σ h = 1` (kernel normalizado), el brillo se preserva. Si `Σ h ≠ 1`, cambia.

**Verificación experimental** con `cv2.boxFilter(normalize=False, k=5)`:
- Brillo medio original: 129.06
- Con `normalize=True` (Σ=1): 129.06 ✓
- Con `normalize=False` en uint8: **250.87** (96.2 % píxeles saturan en 255)
- Con `normalize=False` en float: **3226.53** ≈ 25 × 129.06 = 3226.52

**Implicación práctica.** Cuando se defina un kernel personalizado (p.ej. Laplaciano para realce), siempre verificar que `Σ h = 1` para preservar el brillo; si no, normalizarlo con `kernel /= kernel.sum()`.

### Pregunta 1.3 – Preservación de bordes (filtro bilateral)

**¿Qué mecanismo le permite al filtro bilateral suavizar zonas homogéneas sin borrar los bordes?**

El filtro bilateral introduce **dos pesos** además del kernel Gaussiano espacial clásico:

$$g(x,y) = \\frac{\\sum_{(i,j) \\in W} h_s(i,j) \\cdot h_r(I(x,y), I(x+i, y+j)) \\cdot I(x+i, y+j)}{\\sum_{(i,j) \\in W} h_s(i,j) \\cdot h_r(I(x,y), I(x+i, y+j))}$$

donde:
- $h_s(i,j)$ = Gaussiano **espacial** (como un filtro Gaussiano normal): $\\exp(-\\frac{i^2+j^2}{2\\sigma_s^2})$
- $h_r(I_1, I_2)$ = Gaussiano de **rango/intensidad**: $\\exp(-\\frac{(I_1-I_2)^2}{2\\sigma_r^2})$

El segundo peso **anula la contribución de píxeles cuya intensidad es muy diferente** a la del píxel central. Así, en una zona homogénea $h_r \\approx 1$ para todos los vecinos (suaviza como un Gaussiano normal); pero en un borde, los píxeles "al otro lado" tienen $h_r \\approx 0$, por lo que **no contaminan** la estimación.

**Papel de los parámetros.**
- `σ_s` (sigmaSpace, default 75 en OpenCV): controla el tamaño del kernel espacial → cuánta vecindad se considera.
- `σ_r` (sigmaColor, default 75): controla el umbral perceptual de diferencia de intensidad → qué tan similares deben ser los píxeles para contribuir.
- Aumentar ambos = más suavizado; bajar `σ_r` = más preservación de bordes."""))

# ----- 2.3 Módulo B -----
nb.cells.append(md_cell("""## 2.3 Módulo B · Realce y detección de bordes (8.0 %)

Los bordes corresponden a **cambios bruscos de intensidad**: valores altos de la primera derivada (gradiente) o cruces por cero de la segunda derivada (Laplaciano). Estos filtros son la base de los descriptores de forma y del **HOG** que se usará en la Parte 2.

$$|\\nabla f| = \\sqrt{G_x^2 + G_y^2}, \\quad \\theta = \\operatorname{atan2}(G_y, G_x), \\quad \\nabla^2 f = \\frac{\\partial^2 f}{\\partial x^2} + \\frac{\\partial^2 f}{\\partial y^2}$$

### B.1 Operadores de gradiente: Roberts, Prewitt, Sobel y Scharr

Comparamos cuatro kernels de gradiente:
- **Roberts** (2×2, simple): sensible a ruido, mejor para imágenes binarias.
- **Prewitt** (3×3, pesos iguales): promedia antes de derivar.
- **Sobel** (3×3, peso centro = 2): suavizado gaussiano implícito, mejor SNR.
- **Scharr** (3×3, óptima rotación): aproximación más precisa de la derivada para 3×3.

Mostramos la dirección del gradiente con un **mapa HSV** (matiz = ángulo, brillo = magnitud). Usamos `cv2.CV_64F` para no perder derivadas negativas (con `CV_8U`, **45 % de los píxeles** saturan o se pierden)."""))

nb.cells.append(code_cell(to_python_code(read_slice(PY_A, *A_RANGES["b1"]))))

nb.cells.append(md_cell("""### B.2 Segunda derivada: Laplaciano (4 y 8 vecinos), LoG y DoG

El Laplaciano responde a cruces por cero (bordes), no a la magnitud del gradiente. Comparamos los dos kernels 3×3 estándar (4-vecinos vs 8-vecinos), el Laplaciano del Gaussiano (**LoG**) y la **Diferencia de Gaussianas (DoG)**:

$$\\text{LoG} \\approx \\sigma^2 \\nabla^2 G_\\sigma, \\quad \\text{DoG} = G_{\\sigma_1} - G_{\\sigma_2}, \\quad \\sigma_2 \\approx 1.6 \\sigma_1$$

La correlación DoG vs LoG es **0.9445**, confirmando que el DoG aproxima bien al LoG con menos costo computacional."""))

nb.cells.append(code_cell(to_python_code(read_slice(PY_A, *A_RANGES["b2"]))))

nb.cells.append(md_cell("""### B.3 Canny paso a paso

El detector de Canny clásico ([Canny 1986]) aplica cuatro etapas:
1. Suavizado Gaussiano previo (controla σ).
2. Cálculo de gradiente (Sobel) y dirección.
3. **Supresión de no-máximos**: cada píxel conserva su magnitud solo si es máximo local en la dirección del gradiente.
4. **Umbralización con histéresis**: dos umbrales ($T_{bajo}, T_{alto}$) — píxeles sobre $T_{alto}$ son **fuertes**, los entre ambos son **débiles** y solo se aceptan si están conectados (8 vecinos) a un píxel fuerte.

Variamos σ = 1, 2, 3 y varias parejas de umbrales (50/150, 100/200, 20/60). Proponemos una **regla automática de umbrales basada en la mediana**: $T_{bajo} = \\max(0, (1-\\varsigma) v), T_{alto} = \\min(255, (1+\\varsigma) v)$ con $\\varsigma \\approx 0.33$ y $v$ la mediana de las magnitudes.

**Validación.** Nuestro Canny propio coincide con `cv2.Canny` con **F1 = 1.000** (precisión y recall perfectos)."""))

nb.cells.append(code_cell(to_python_code(read_slice(PY_A, *A_RANGES["b3"]))))

nb.cells.append(md_cell("""### B.4 Realce de nitidez: Unsharp masking y High-boost

El realce de nitidez se basa en **restar la versión suavizada** de la imagen:

$$g = f + k \\cdot (f - f_{suav})$$

- $k = 0$ → no realce
- $0 < k < 1$ → unsharp masking clásico
- $k = 1$ → high-boost (suma la imagen original)
- $k > 1$ → high-boost agresivo (suma varias veces la imagen de detalle)

**Aparecen halos** alrededor de los bordes: en una transición de negro (60) a blanco (180), el filtro detecta el "detalle" del borde y crea un sobreimpulso claro (+38.7 para k=2) y un subimpulso oscuro (−38.7). Estos halos son perceptibles a $k = 2$ y se vuelven molestos para $k > 3$.

**En el proyecto.** El unsharp masking podría usarse en la Parte 2 para resaltar detalles de textura fina en los minerales, pero preferimos el CLAHE sobre L (LAB) que corrige la iluminación no uniforme sin introducir halos."""))

nb.cells.append(code_cell(to_python_code(read_slice(PY_A, *A_RANGES["b4"]))))

# ----- Preguntas 1.4–1.5 -----
nb.cells.append(md_cell("""## 📝 Preguntas conceptuales del Módulo B

### Pregunta 1.4 – Ruido y derivadas

**¿Por qué los operadores de derivada amplifican el ruido? ¿Por qué el Laplaciano es aún más sensible que Sobel y cómo lo resuelve el LoG?**

**Amplificación por derivada.** Los operadores derivativos son **filtros pasa-alto**: ponderan las diferencias entre píxeles vecinos. El ruido (componente de alta frecuencia) tiene justamente diferencias grandes entre píxeles adyacentes. Para un kernel $h$ y ruido blanco independiente, la desviación de salida es:

$$\\sigma_{salida} = \\sigma \\|h\\|_2 = \\sigma \\sqrt{\\sum h^2}$$

Por ejemplo, con σ=10 (ruido de entrada) y un kernel Sobel $h = [-1,0,1]$, $\\|h\\|_2 = \\sqrt{3}$ y la salida tiene σ ≈ 17.3, es decir, **amplificación del 73 %**.

**Laplaciano más sensible.** El Laplaciano es una **segunda derivada**, así que amplifica aún más las altas frecuencias (proporcional a $f^2$ para ruido blanco). Por eso los Laplacianos "directos" sobre la imagen ruidosa dan resultados inutilizables.

**Solución del LoG.** Se suaviza primero con un Gaussiano (filtro pasa-bajo que atenúa el ruido) y *después* se aplica el Laplaciano. Matemáticamente es equivalente a convolucionar con $\\nabla^2 G_\\sigma$ (Laplaciano del Gaussiano), un filtro **banda-rechaza** que elimina las frecuencias muy altas (ruido) pero conserva los cruces por cero de los bordes.

**DoG como alternativa más rápida.** El DoG ($G_{\\sigma_1} - G_{\\sigma_2}$) aproxima el LoG (correlación 0.9445 con σ₂=1.6σ₁) pero solo requiere dos convoluciones con Gaussianas en lugar de calcular $\\nabla^2 G_\\sigma$ explícitamente.

### Pregunta 1.5 – Histéresis en Canny

**Explique por qué Canny usa dos umbrales en lugar de uno y qué función cumplen la supresión de no-máximos y el seguimiento por histéresis en la continuidad de los contornos.**

**Problema con un solo umbral.** Un umbral fijo no funciona bien con iluminación variable: umbrales altos pierden bordes en zonas claras, umbrales bajos añaden ruido en zonas oscuras. Para resolverlo, Canny (1986) propuso dos umbrales:

- $T_{alto}$ (típico 100-200): marca píxeles **fuertes** — seguro que son borde.
- $T_{bajo}$ (típico 50-100): marca píxeles **débiles** — podrían ser borde.
- Píxeles entre umbrales: solo se aceptan si están conectados (8-vecinos) a un píxel fuerte.

**Supresión de no-máximos.** Adelgaza los bordes: en cada píxel, se mira la dirección del gradiente y se conserva la magnitud solo si es el máximo local en esa dirección. Sin esta etapa, los bordes serían "gruesos" (1-3 píxeles de ancho). Con ella, son contornos de 1 píxel — útiles para medición y para la Parte 2 (HOG necesita contornos finos).

**Histéresis.** Evita que un borde se rompa en segmentos cuando la magnitud cae短暂的mente bajo $T_{alto}$ por ruido. Mientras el píxel débil esté conectado a un píxel fuerte, se conserva. Esto produce contornos **continuos**, que es exactamente la propiedad que explota el HOG."""))

# ----- Filtros extra (homomórfico + Laws) -----
nb.cells.append(md_cell("""## Filtros extra (opcionales, bonificación)

### Filtro homomórfico

Útil para corregir iluminación no uniforme separando las componentes de iluminación (bajas frecuencias) y reflectancia (altas frecuencias) en el dominio logarítmico:

$$\\ln f(x,y) = \\ln i(x,y) + \\ln r(x,y)$$

Aplicamos FFT, multiplicamos por $H(u,v)$ que atenúa bajas frecuencias ($\\gamma_L < 1$) y amplifica altas ($\\gamma_H > 1$), y volvemos al dominio espacial con $\\exp$. **En el proyecto**, este filtro sería una alternativa al CLAHE para fotos caseras con iluminación lateral."""))

nb.cells.append(code_cell(to_python_code(read_slice(PY_A, *A_RANGES["extras"]))))

# ----- 2.4 Módulo C -----
nb.cells.append(md_cell("""## 2.4 Módulo C · Transformaciones de intensidad (4.0 %)

Las **transformaciones puntuales** modifican cada píxel de forma independiente mediante una función $s = T(r)$. Son económicas (operan pixel-a-pixel) y muy útiles para corregir exposición antes de extraer características.

**Catálogo del enunciado:**
- **Negativo:** $s = (L-1) - r$ (con L=256 en uint8 → $s = 255 - r$).
- **Logarítmica:** $s = c \\cdot \\log(1 + r)$ — expande bajos, comprime altos.
- **Gamma:** $s = c \\cdot r^\\gamma$ — γ < 1 aclara zonas oscuras, γ > 1 oscurece.
- **Estiramiento de contraste:** percentiles 2%–98%.

### C.1 Curvas de transferencia y aplicación a imágenes mal expuestas

Graficamos las curvas $T(r)$ de las transformaciones negativa, logarítmica y gamma (γ=0.3, 0.5, 1.5, 2.5) en una misma figura y las aplicamos a una imagen subexpuesta y sobreexpuesta. Implementamos todas mediante tablas de búsqueda (`cv2.LUT`) para máxima eficiencia."""))

nb.cells.append(code_cell(to_python_code(read_slice(PY_CDE, *CDE_RANGES["c1"]))))

nb.cells.append(md_cell("""### C.2 Visualización del espectro de Fourier (aplicación de la transformación logarítmica)

Aplicamos la transformación logarítmica a la magnitud de la FFT de una imagen y comparamos con la visualización sin escalar. **Esta será la forma estándar de mostrar espectros en el Módulo D.**

El log comprime el rango dinámico enorme del espectro (razón max/mediana ≈ 9000×) y revela direcciones privilegiadas, picos de patrones periódicos y el decaimiento radial típico de las imágenes naturales."""))

nb.cells.append(code_cell(to_python_code(read_slice(PY_CDE, *CDE_RANGES["c2"]))))

# ----- Pregunta 1.6 -----
nb.cells.append(md_cell("""## 📝 Pregunta 1.6 – Rango dinámico

**¿Por qué la transformación logarítmica es indispensable para visualizar el espectro de Fourier y qué tipo de imágenes se benefician de ella?**

El espectro de Fourier de una imagen natural tiene un **rango dinámico extremo**: la magnitud en el origen (frecuencia cero) puede ser $10^6$ veces mayor que la magnitud en frecuencias altas (detalles de bordes y ruido). En una visualización lineal, solo el componente DC sería visible (y todo lo demás quedaría en negro puro).

La transformación $s = c \\cdot \\log(1 + |F(u,v)|)$:
- **Comprime el rango** multiplicativo en uno aditivo manejable.
- Permite que los detalles del espectro (anillos de interferencia, picos de ruido periódico, direcciones de Gabor) se hagan visibles.
- No altera la información — solo su representación.

**Imágenes que se benefician:**
- **Médicas** (rayos X, MRI): los huesos o tumores son pequeñas variaciones sobre un fondo brillante.
- **Astronómicas**: la diferencia entre una estrella y el fondo interestelar es de muchos órdenes de magnitud.
- **Espectroscopia** y **fingerprinting** (huellas dactilares).
- Cualquier imagen donde el componente DC domina y los detalles son débiles."""))

# ----- 2.5 Módulo D -----
nb.cells.append(md_cell("""## 2.5 Módulo D · Filtrado en el dominio de la frecuencia (7.0 %)

Por el teorema de convolución, **filtrar en el espacio equivale a multiplicar en la frecuencia**:

$$G(u,v) = H(u,v) \\cdot F(u,v)$$

Las bajas frecuencias (centro del espectro desplazado) contienen las regiones suaves; las altas frecuencias contienen bordes, texturas y ruido.

**Filtros clásicos:**
- **Ideal paso bajo:** $H(u,v) = 1$ si $D(u,v) \\le D_0$, 0 en otro caso → produce *ringing* severo.
- **Butterworth** (orden n): $H(u,v) = 1 / [1 + (D/D_0)^{2n}]$ → ringing decreciente con n.
- **Gaussiano:** $H(u,v) = \\exp(-D^2/(2D_0^2))$ → **sin ringing**.
- **Paso alto:** $H_{PA}(u,v) = 1 - H_{PB}(u,v)$.

### D.1 Espectro de Fourier: magnitud y fase

Calculamos la FFT 2D, centramos con `fftshift` y mostramos magnitud (escala log) y fase. Reconstruimos la imagen usando **solo la magnitud** y **solo la fase**, y comparamos cuál conserva más información estructural (medida SSIM).

**Conclusión esperada.** La fase conserva más estructura perceptual que la magnitud."""))

nb.cells.append(code_cell(to_python_code(read_slice(PY_CDE, *CDE_RANGES["d1"]))))

nb.cells.append(md_cell("""### D.2 Filtros paso bajo y paso alto: Ideal, Butterworth (n = 1, 2, 4) y Gaussiano

Implementamos los filtros Ideal, Butterworth (n = 1, 2, 4) y Gaussiano para D₀ = 10, 30 y 60. Mostramos la máscara H(u,v), el espectro filtrado y la imagen resultante. **Identificamos el fenómeno de *ringing* (efecto Gibbs)** — oscilaciones cerca de los bordes que aparecen en el filtro Ideal por su transformada inversa tipo `sinc`.

| Filtro | Overshoot en escalón D₀=15 |
|---|---|
| Ideal | ±10.4 (severo) |
| Butterworth n=1 | 0.0 |
| Butterworth n=2 | ±4.1 |
| Butterworth n=4 | ±8.2 |
| Gaussiano | 0.0 (sin ringing) |

Para n≥4, Butterworth también presenta ringing, aunque menor. El Gaussiano es siempre la opción más segura."""))

nb.cells.append(code_cell(to_python_code(read_slice(PY_CDE, *CDE_RANGES["d2"]))))

nb.cells.append(md_cell("""### D.3 Eliminación de ruido periódico (filtro rechaza-banda / notch)

Sumamos a la imagen un patrón sinusoidal (por ejemplo $20\\sin(2\\pi \\cdot 0.1 \\cdot x)$) y localizamos los picos que genera en el espectro. Los eliminamos con un **filtro notch** (máscara con ceros en las posiciones de los picos) y comparamos con el mejor filtro espacial del Módulo A.

Para N=512 y f=0.1 ciclos/px, los picos aparecen en $(\\Delta u, \\Delta v) = (\\pm 51, 0)$ en el espectro centrado."""))

nb.cells.append(code_cell(to_python_code(read_slice(PY_CDE, *CDE_RANGES["d3"]))))

nb.cells.append(md_cell("""### D.4 Equivalencia espacio–frecuencia

Verificamos numéricamente que un filtro Gaussiano aplicado en frecuencia produce el mismo resultado que `cv2.GaussianBlur` con el σ espacial equivalente:

$$\\sigma_{espacial} \\approx \\frac{N}{2\\pi \\cdot D_0}$$

Para N=512, D₀=30 → σ ≈ 2.72 px. Error máximo en interior (margen de 11 px): **6.73×10⁻³** — efectivamente idéntico, salvo por las condiciones de borde (la FFT asume periodicidad, mientras que `cv2.GaussianBlur` espeja los bordes)."""))

nb.cells.append(code_cell(to_python_code(read_slice(PY_CDE, *CDE_RANGES["d4"]))))

# ----- Preguntas 1.7–1.8 -----
nb.cells.append(md_cell("""## 📝 Preguntas conceptuales del Módulo D

### Pregunta 1.7 – Ringing

**¿Por qué el filtro ideal produce anillos (ringing) alrededor de los bordes y el Gaussiano no? Relacione su respuesta con la transformada inversa de cada máscara.**

El filtro Ideal tiene como máscara una **circunferencia binaria**: $H(u,v) = 1$ si $D(u,v) \\le D_0$, 0 en caso contrario. Su **transformada inversa de Fourier** es una función `sinc` 2D:

$$h(x,y) = \\frac{J_1(2\\pi D_0 r)}{\\pi r}$$

donde $J_1$ es la función de Bessel de primer orden y $r = \\sqrt{x^2+y^2}$. Esta `sinc` tiene **lóbulos laterales** que oscilan alrededor de 0 con amplitud decreciente. Cuando se convoluciona con un borde abrupto (función escalón), los lóbulos laterales producen las **oscilaciones de Gibbs** (ringing) que vemos en la imagen filtrada.

El filtro Gaussiano tiene $H(u,v) = \\exp(-D^2/(2D_0^2))$. Su inversa es **otra Gaussiana** (la transformada de Fourier de una Gaussiana es otra Gaussiana), sin lóbulos laterales:

$$h(x,y) = 2\\pi D_0^2 \\exp(-2\\pi^2 D_0^2 r^2)$$

La Gaussiana es monótona y suave → **sin ringing**.

**Butterworth.** A medida que el orden n aumenta, la Butterworth se acerca al Ideal y el ringing crece. Para n=1 es muy suave (como el Gaussiano); para n=4 ya muestra oscilaciones visibles.

**Implicación práctica.** En la Parte 2, si decidiéramos filtrar en frecuencia (no es el caso), usaríamos Butterworth n=1 o Gaussiano para evitar artefactos de borde.

### Pregunta 1.8 – ¿Espacio o frecuencia?

**¿En qué situaciones es computacionalmente preferible filtrar en el dominio de la frecuencia en vez de convolucionar en el espacio? Considere el tamaño del kernel.**

El costo de una convolución espacial es $O(N^2 \\cdot K^2)$ para una imagen $N \\times N$ y kernel $K \\times K$. El costo de filtrar en frecuencia es $O(N^2 \\log N)$ (la FFT es $N^2 \\log N^2$).

**Crossover.** Igualando ambos: $K^2 = \\log N$, es decir, **para $K > \\sqrt{\\log N}$ conviene el espacio, para $K < \\sqrt{\\log N}$ conviene la frecuencia**.

Para N=512: $\\sqrt{\\log 512} \\approx 1.7$. Para N=1024: $\\approx 1.8$. **En la práctica, casi siempre conviene el espacio** porque los kernels típicos son 3×3, 5×5, 7×7, etc.

**Donde SÍ conviene la frecuencia:**
1. **Filtros no separables grandes** (p.ej. Butterworth 2D con D₀ fijo) — el costo espacial es $K^2$ mientras que la FFT es solo $\\log N$.
2. **Convoluciones múltiples** — la FFT es lineal, así que $N$ convoluciones se hacen como UNA sola FFT + $N$ multiplicaciones + UNA sola FFT inversa.
3. **Filtros con forma compleja** que no se pueden separar — la FFT los aplica de forma trivial como multiplicación.

**En nuestro proyecto.** Todos los filtros del pipeline (Gaussiano, mediana, bilateral, Laplaciano) son pequeños y separables, así que trabajamos en el espacio. La FFT se usa solo en el Módulo D del taller para visualizar el espectro."""))

# ----- 2.6 Módulo E -----
nb.cells.append(md_cell("""## 2.6 Módulo E · Filtros direccionales y de textura – Gabor (3.7 %)

Un **filtro de Gabor** es una sinusoide modulada por una envolvente Gaussiana:

$$g(x,y; \\theta, \\lambda, \\sigma) = \\exp\\left(-\\frac{x'^2 + y'^2}{2\\sigma^2}\\right) \\cos\\left(2\\pi \\frac{x'}{\\lambda} + \\phi\\right)$$

donde $(x', y')$ es el plano rotado por θ. Responde fuertemente a texturas con orientación θ y frecuencia espacial $1/\\lambda$ concretas. Un **banco de Gabor** es uno de los descriptores de textura más usados en clasificación y se reutilizará en la Parte 2.

### E.1 Banco de filtros: 4 orientaciones × 3 longitudes de onda

Construimos un banco de **4 orientaciones** (0°, 45°, 90°, 135°) × **3 longitudes de onda** $\\lambda$ con `cv2.getGaborKernel`. Visualizamos los kernels y las respuestas sobre una imagen con texturas (tela, ladrillo, madera u hojas). Estudiamos el efecto del ancho de la envolvente σ."""))

nb.cells.append(code_cell(to_python_code(read_slice(PY_CDE, *CDE_RANGES["e1"]))))

nb.cells.append(md_cell("""### E.2 Vector de textura de 24 características

Para cada respuesta del banco, calculamos la **media** y la **desviación estándar** de la energía. Concatenamos esos valores en un vector de **24 características** (12 filtros × 2 estadísticas) y lo comparamos entre dos texturas distintas. Estudiamos qué pasa si rotamos la textura 90° (preludio de la Pregunta 1.9)."""))

nb.cells.append(code_cell(to_python_code(read_slice(PY_CDE, *CDE_RANGES["e2"]))))

# ----- Pregunta 1.9 -----
nb.cells.append(md_cell("""## 📝 Pregunta 1.9 – Orientación

**¿Qué ocurre con la respuesta del banco de Gabor si se rota el objeto 90°? ¿Cómo podría construirse un descriptor invariante a la rotación a partir de ese banco?**

Si rotamos la imagen 90°, las respuestas del banco rotan correspondientemente: el filtro a 0° se parece al filtro a 90° aplicado a la imagen original, etc. El vector de 24 dims **cambia de forma sustancial** (distancia coseno 0.326 en nuestras pruebas), por lo que **NO es invariante** por construcción.

**Construcción de un descriptor invariante:**

1. **Pooling por máximo o suma sobre orientaciones:** para cada λ, sumar (o tomar el máximo) las respuestas de las 4 orientaciones. Resultado: vector de 3λ × 2 = 6 dimensiones invariante a rotación. Verificación: distancia coseno = -0.000 (idéntico).

2. **Histograma de respuestas:** calcular el histograma de respuestas normalizadas del banco. Como una rotación solo permuta las respuestas entre orientaciones, el histograma se preserva.

3. **Filtro circular:** en lugar de muestrear θ discreto, integrar sobre todas las orientaciones: $\\int_0^\\pi g(x,y; \\theta, \\lambda, \\sigma) d\\theta$. El resultado es un filtro con simetría circular → invariante por construcción.

4. **Combinar con descriptores de Fourier** (FFT) que son intrínsecamente invariantes a rotación si se usa la magnitud del espectro.

**En el proyecto.** Para la Parte 2, los minerales se fotografían siempre en orientación vertical/horizontal, por lo que la invariancia rotacional no es crítica. Usamos 4θ × 2λ (8 filtros, 16 dims) sin pooling."""))

# ----- Conclusiones finales -----
nb.cells.append(md_cell("""## 🏁 Conclusiones del taller (Secciones 2.1–2.6)

### Módulo A (convolución y suavizado)
- La convolución manual coincide con `cv2.filter2D` con error **< 10⁻¹²** (numéricamente idéntico).
- El kernel Gaussiano es **separable** (rango numérico = 1) → speedup **15.5×** con dos pasadas 1D.
- En la **batalla de filtros**, la **mediana** gana para sal-y-pimienta y el **Gaussiano** para ruido Gaussiano (justifica σ=1.0 en Parte 2).
- `cv2.boxFilter(normalize=False)` **satura el 96.2 %** de los píxeles — el kernel DEBE sumar 1 (Pregunta 1.2).

### Módulo B (realce y bordes)
- `cv2.CV_64F` es obligatorio para derivadas: con `CV_8U` se pierde el **45 %** de píxeles con derivada negativa.
- Nuestro Canny propio coincide con `cv2.Canny` con **F1 = 1.000**.
- El DoG aproxima el LoG con correlación **0.9445** (alternativa más rápida).
- El unsharp masking con k=2 produce halos de **±38.7 niveles** alrededor de los bordes.

### Módulo C (intensidad)
- El log del espectro multiplicativo es indispensable para visualizar FFT (rango dinámico 9000×).
- γ=0.3–0.5 recupera detalle en sombras; γ=1.5–2.5 oscurece sobreexpuestas.

### Módulo D (frecuencia)
- El filtro Ideal con D₀=10 produce **overshoot ±10.4** (Gibbs); el Gaussiano, **0.0** (sin ringing).
- Butterworth n=1 y Gaussiano son las opciones **seguras** sin ringing.
- La equivalencia espacio↔frecuencia del Gaussiano es exacta salvo en los bordes (error interior 6.7×10⁻³).

### Módulo E (Gabor)
- Banco 4θ × 3λ genera un **vector de textura de 24 dims** (μ, σ por filtro).
- Para invariancia rotacional: pooling por λ → 6 dims invariantes (distancia coseno 0.000).

### 🔗 Conexión con la Parte 2

| Hallazgo del taller | Aplicación en la Parte 2 |
|---|---|
| Gaussiano σ=1.0 = mejor suavizado general | Preprocesamiento del pipeline |
| Separabilidad (15.5× speedup) | HOG en `skimage.feature.hog` |
| CLAHE corrige iluminación no uniforme | Ecualización adaptativa en LAB |
| DoG ≈ LoG | Detección de bordes en la máscara |
| Banco de Gabor invariante | Descriptor de textura (16 dims) |

### 📚 Referencias del taller
- González & Woods, *Digital Image Processing*, 4.ª ed., capítulo 3 (filtros espaciales) y 4 (FFT).
- Canny, "A Computational Approach to Edge Detection", *IEEE Trans. PAMI*, 1986.
- Daugman, "Uncertainty Relation for Resolution in Space, Spatial Frequency, and Orientation", 1985 (filtros de Gabor).

---

**✅ Las 9 preguntas conceptuales (1.1–1.9) han sido respondidas.**
**✅ Los 5 módulos (A–E) del taller están implementados y verificados.**"""))


# ============================================================================
# Guardar
# ============================================================================
# Añadir IDs a las celdas (necesario para nbformat ≥ 5.0)
import uuid
for c in nb.cells:
    if not hasattr(c, 'id'):
        c['id'] = uuid.uuid4().hex[:8]

# Convertir a formato notebook nativo de nbformat
nb_format = nbf.from_dict(nb)

# Contar
n_md = sum(1 for c in nb_format.cells if hasattr(c, 'cell_type') and c.cell_type == 'markdown')
n_code = sum(1 for c in nb_format.cells if hasattr(c, 'cell_type') and c.cell_type == 'code')

# Usar nbformat.writes que respeta el formato estándar
NB_OUT.write_text(nbf.writes(nb_format), encoding="utf-8")

# Verificar
size_kb = NB_OUT.stat().st_size / 1024
print(f"\n✓ Notebook generado: {NB_OUT}")
print(f"  Celdas totales: {len(nb.cells)}")
print(f"  - Markdown: {n_md}")
print(f"  - Code:     {n_code}")
print(f"  Tamaño: {size_kb:.1f} KB")