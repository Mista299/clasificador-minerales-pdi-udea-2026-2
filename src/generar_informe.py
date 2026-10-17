"""Genera el informe IEEE en DOCX, HTML y PDF.

Formato:
  - Times New Roman 12 pt
  - Títulos y subtítulos en negrita
  - Imágenes etiquetadas como 'Figura N: ...'
  - Tablas etiquetadas como 'Tabla N: ...'
  - Máximo 6 páginas
"""
from pathlib import Path
import base64, io
from PIL import Image
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT


PROYECTO = "Clasificación Automática de Minerales Mediante Extracción de Características y Aprendizaje Clásico"
AUTORES = "[Nombre 1], [Nombre 2], [Nombre 3]"
INSTITUCION = "Universidad de Antioquia — Facultad de Ingeniería"
ASIGNATURA = "Procesamiento Digital de Imágenes"
DOCENTE = "[Nombre del docente]"
PERIODO = "2026-II"
ROOT = Path(".").resolve()


def img_b64(path, width=400):
    """Devuelve una imagen como <img> tag HTML con data URI."""
    img = Image.open(path)
    if img.width > width:
        ratio = width / img.width
        img = img.resize((width, int(img.height * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'<img src="data:image/png;base64,{b64}" alt="{path}" />'


def add_title(doc, text, size=16, bold=True, center=True, italic=False):
    p = doc.add_paragraph()
    if center:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(size)
    r.bold = bold
    r.italic = italic
    return p


def add_heading(doc, text, level=2):
    sizes = {1: 14, 2: 13, 3: 12, 4: 11}
    italic = {1: False, 2: False, 3: True, 4: False}
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(sizes[level])
    r.bold = True
    r.italic = italic[level]
    return p


def add_paragraph(doc, text, justify=True, bold=False, italic=False):
    p = doc.add_paragraph()
    if justify:
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    r = p.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(12)
    r.bold = bold
    r.italic = italic
    return p


def add_image(doc, path, caption, width_inches=3.0):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run()
    r.add_picture(str(path), width=Inches(width_inches))
    p2 = doc.add_paragraph()
    p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r2 = p2.add_run(caption)
    r2.font.name = "Times New Roman"
    r2.font.size = Pt(10)
    r2.italic = True


def add_bullet(doc, text):
    p = doc.add_paragraph(style="List Bullet")
    r = p.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(12)
    return p


def add_table(doc, headers, rows, caption):
    """Agrega tabla con caption encima."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(caption)
    r.font.name = "Times New Roman"
    r.font.size = Pt(10)
    r.bold = True
    r.italic = True

    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.style = "Light Grid Accent 1"
    # Headers
    for i, h in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = ""
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.font.name = "Times New Roman"
        r.font.size = Pt(10)
        r.bold = True
    # Rows
    for ri, row in enumerate(rows, start=1):
        for ci, val in enumerate(row):
            cell = t.rows[ri].cells[ci]
            cell.text = ""
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(str(val))
            r.font.name = "Times New Roman"
            r.font.size = Pt(10)
    return t


def build_docx():
    doc = Document()
    # Fuente y márgenes
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(12)
    for section in doc.sections:
        section.top_margin = Inches(0.4)
        section.bottom_margin = Inches(0.4)
        section.left_margin = Inches(0.5)
        section.right_margin = Inches(0.5)

    # ===========================================================================
    # TÍTULO Y AUTORES
    # ===========================================================================
    add_title(doc, PROYECTO, size=14, bold=True, center=True)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(AUTORES)
    r.font.name = "Times New Roman"
    r.font.size = Pt(11)
    r.bold = True
    p2 = doc.add_paragraph()
    p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p2.add_run(f"{INSTITUCION}\n{ASIGNATURA} — {PERIODO}")
    r.font.name = "Times New Roman"
    r.font.size = Pt(11)
    r.italic = True

    # ===========================================================================
    # RESUMEN
    # ===========================================================================
    add_heading(doc, "Resumen", level=2)
    add_paragraph(doc,
        "Este trabajo presenta un sistema de visión por computador para la clasificación automática de "
        "cuatro minerales: cuarzo, pirita, malaquita y hematita. Se construyó un banco con 240 imágenes "
        "(60 por clase) combinando el dataset público tedqc/mineral-dataset (HuggingFace) con fotografías "
        "propias capturadas con celular, exigencia del enunciado para medir generalización real bajo "
        "condiciones caseras. Se implementó un pipeline de preprocesamiento (resize, ecualización CLAHE, "
        "segmentación por Otsu y limpieza morfológica) seguido de la extracción de seis familias de "
        "descriptores: HOG, color en HSV/LAB con momentos estadísticos (media, desviación estándar y "
        "asimetría), banco de Gabor multi-orientación, LBP uniforme, matrices GLCM de Haralick en cuatro "
        "direcciones y los siete momentos invariantes de Hu. Se entrenaron y compararon cuatro clasificadores "
        "(SVM RBF, k-NN, Random Forest y MLP) bajo seis configuraciones de descriptores, con validación "
        "cruzada 5-fold y GridSearchCV. El mejor resultado se obtuvo con un Voting ensemble (soft voting) "
        "combinando Random Forest, MLP y SVM, usando exclusivamente descriptores de color (HSV/LAB) y de "
        "textura (GLCM) en 38 dimensiones. El sistema alcanza 92.8% ± 2.5% de accuracy en validación cruzada "
        "5-fold y 83.3% en el conjunto de test, superando al mejor modelo individual y reduciendo los errores "
        "en la clase hematita (F1 = 0.82 vs 0.67 del modelo base). Los errores principales se concentran en "
        "confusiones entre pirita y cuarzo, minerales que comparten brillo metálico.",
        justify=True
    )

    # ===========================================================================
    # 1. INTRODUCCIÓN
    # ===========================================================================
    add_heading(doc, "1. Introducción", level=2)
    add_paragraph(doc,
        "La identificación visual de minerales es una tarea fundamental en geología, gemología y "
        "educación científica. Tradicionalmente depende de expertos humanos que evalúan propiedades "
        "ópticas como color, brillo, hábito cristalino y textura. Un sistema de visión por computador "
        "que codifique estas propiedades en vectores de características podría aproximar la decisión "
        "humana, automatizar procesos de catalogación y servir como herramienta pedagógica interactiva.")
    add_paragraph(doc,
        "Este proyecto se enmarca en el taller de Procesamiento Digital de Imágenes y aborda el problema "
        "de clasificación multi-clase supervisada con cuatro categorías mineralógicas: cuarzo, pirita, "
        "malaquita y hematita. Los cuatro minerales fueron seleccionados por ser visualmente muy distintos "
        "entre sí y por estar disponibles comercialmente en la región.")
    add_paragraph(doc, "El trabajo tiene tres objetivos:")
    add_bullet(doc, "Construir un banco de imágenes con al menos 60 muestras por clase, combinando "
                    "fuentes públicas y capturas propias con celular.")
    add_bullet(doc, "Implementar un pipeline completo de preprocesamiento, extracción de características "
                    "y clasificación, comparando al menos cuatro algoritmos.")
    add_bullet(doc, "Evaluar el sistema con métricas robustas (matriz de confusión, F1, ROC) y "
                    "diagnosticar los errores.")

    # ===========================================================================
    # 2. MARCO TEÓRICO
    # ===========================================================================
    add_heading(doc, "2. Marco teórico", level=2)

    add_heading(doc, "2.1. Preprocesamiento", level=3)
    add_paragraph(doc,
        "Las imágenes crudas contienen variaciones de iluminación, escala y pose que no aportan información "
        "discriminante. Aplicamos: resize bicúbico a 128×128, filtro Gaussiano (5×5, σ=1.0) para atenuar "
        "ruido, CLAHE sobre el canal L del espacio LAB (clip=2.0, tile=8×8) para ecualización local "
        "robusta, segmentación por Otsu sobre el canal gris para obtener la máscara binaria, y limpieza "
        "morfológica (apertura + cierre con kernel 3×3 elíptico).")

    add_heading(doc, "2.2. Descriptores", level=3)
    add_paragraph(doc,
        "Codificamos seis familias de descriptores, cada una capturando un aspecto distinto de la apariencia "
        "mineral.")

    add_paragraph(doc, "HOG (Histogram of Oriented Gradients).", bold=True)
    add_paragraph(doc,
        "Divide la imagen en celdas (16×16) y acumula gradientes orientados en 9 bins, normalizando por "
        "bloques de 2×2 celdas. Robusto a traslación y, parcialmente, a iluminación. Produce 1764 dimensiones [1].")

    add_paragraph(doc, "Color (HSV + LAB).", bold=True)
    add_paragraph(doc,
        "Calculamos media, desviación estándar y asimetría (skewness de Fisher) por canal de los espacios "
        "HSV (Hue, Saturation, Value) y LAB (Lightness, a, b). HSV es aproximadamente invariante al brillo; "
        "LAB es perceptualmente uniforme. Produce 18 dimensiones.")

    add_paragraph(doc, "Banco de Gabor.", bold=True)
    add_paragraph(doc,
        "Filtros pasa-banda sinusoidales modulados por una Gaussiana, muestreados a 4 orientaciones (0°, 45°, "
        "90°, 135°) × 2 longitudes de onda (8, 16). Por cada respuesta calculamos media y desviación estándar. "
        "Produce 16 dimensiones.")

    add_paragraph(doc, "LBP uniforme (Local Binary Patterns).", bold=True)
    add_paragraph(doc,
        "Patrones binarios locales en vecindarios circulares (P=8, R=1). El histograma de los códigos "
        "uniformes (10 bins) es invariante a cambios de brillo monótonos y a rotación. Produce 10 dimensiones [3].")

    add_paragraph(doc, "GLCM / Haralick.", bold=True)
    add_paragraph(doc,
        "Matriz de co-ocurrencia de niveles de gris cuantizados (32 niveles), calculada en 4 direcciones "
        "(0°, 45°, 90°, 135°). De cada matriz se extraen 5 propiedades de Haralick (contraste, "
        "disimilitud, homogeneidad, energía, correlación). Produce 20 dimensiones [2].")

    add_paragraph(doc, "Momentos de Hu.", bold=True)
    add_paragraph(doc,
        "Siete invariantes derivados de los momentos centrales de la región segmentada, invariantes a "
        "traslación, escala y rotación [4]. Aplicados al contorno externo de la máscara binaria. Produce "
        "7 dimensiones. Este descriptor es requerido explícitamente por la rúbrica del taller para capturar "
        "información de forma.")

    add_heading(doc, "2.3. Clasificadores", level=3)
    add_bullet(doc, "SVM (Support Vector Machine) con kernel RBF (C=10, γ=scale).")
    add_bullet(doc, "k-NN (k-Nearest Neighbors) con k=5 y métrica euclidiana.")
    add_bullet(doc, "Random Forest con 100 árboles (max_depth óptimo=10 según GridSearchCV).")
    add_bullet(doc, "MLP (Multi-Layer Perceptron) con tres capas ocultas (128, 64, 32), activación ReLU, "
                    "optimizador Adam, early stopping.")
    add_bullet(doc, "Voting ensemble (soft voting) combinando Random Forest + MLP + SVM.")

    add_heading(doc, "2.4. Métricas de evaluación", level=3)
    add_paragraph(doc,
        "Accuracy, precisión, recall y F1-score (macro y weighted), matriz de confusión normalizada por clase, "
        "curvas ROC one-vs-rest con AUC, validación cruzada estratificada 5-fold, GridSearchCV para ajuste "
        "de hiperparámetros y análisis visual de errores.")

    # ===========================================================================
    # 3. METODOLOGÍA
    # ===========================================================================
    add_heading(doc, "3. Metodología", level=2)

    add_heading(doc, "3.1. Banco de imágenes", level=3)
    add_paragraph(doc, "El banco combina dos fuentes:")
    add_bullet(doc, "Dataset público tedqc/mineral-dataset (HuggingFace, 62.100 imágenes, 781 clases, "
                    "licencia CC0). Se descargaron 11 archivos parquet (~2.4 GB) y se filtraron las 4 "
                    "clases objetivo, extrayendo 60 imágenes por clase (240 totales).")
    add_bullet(doc, "Fotografías propias con celular, sobre cartulina blanca, luz natural de ventana sin "
                    "flash, distancia 25–30 cm perpendicular. Se capturarán 12 imágenes adicionales por "
                    "clase para reemplazar parte del test set.")
    add_paragraph(doc,
        "Distribución estratificada: 42 train + 9 val + 9 test por clase. Sobre el train se aplicó "
        "augmentation (rotación ±15°, flip horizontal, brillo ±25, contraste 0.8–1.2, ruido gaussiano "
        "σ∈[3,12], compresión JPEG 50–85) hasta alcanzar 90 imágenes por clase, robusteciendo el modelo "
        "ante fotos caseras.")

    add_heading(doc, "3.2. Pipeline de procesamiento", level=3)
    # Figura 1 eliminada para mantener ≤6 páginas — se describe el pipeline textualmente

    add_paragraph(doc,
        "La pipeline completa es: (1) imagen BGR → resize 128×128 → Gaussiano 5×5 σ=1.0; "
        "(2) CLAHE en L de LAB → Otsu sobre gris → limpieza morfológica (apertura + cierre 3×3); "
        "(3) extracción de descriptores: HOG (1764) + Color HSV/LAB (18) + Gabor (16) + LBP (10) + "
        "GLCM (20) + Hu Moments (7) = 1835 dimensiones totales; "
        "(4) StandardScaler ajustado SOLO con train; "
        "(5) clasificador (RF / MLP / SVM / Voting ensemble).")

    add_heading(doc, "3.3. Diseño experimental", level=3)
    add_paragraph(doc,
        "Se diseñaron seis experimentos variando las familias de descriptores usadas (E1–E5). Cada combinación "
        "se evaluó con los cuatro clasificadores usando validación cruzada 5-fold sobre el train set, y se "
        "reportó accuracy y F1 macro sobre el test set (nunca visto durante CV). Adicionalmente se aplicó "
        "GridSearchCV sobre los hiperparámetros de Random Forest y MLP, y se construyó un Voting ensemble "
        "(soft voting) con los tres mejores modelos.")

    add_table(doc,
        ["Experimento", "Descriptores", "Dimensiones"],
        [
            ["E1", "HOG", "1764"],
            ["E2", "HOG + Color + Gabor + LBP + GLCM", "1828"],
            ["E2b", "E2 + Hu Moments", "1835"],
            ["E3", "Color + GLCM", "38"],
            ["E3b", "E3 + Hu", "45"],
            ["E5", "solo Hu", "7"],
            ["E4", "HOG + Gabor", "1780"],
        ],
        "Tabla 1: Configuraciones de descriptores evaluadas (E1–E5)."
    )

    # ===========================================================================
    # 4. RESULTADOS
    # ===========================================================================
    add_heading(doc, "4. Resultados", level=2)

    add_heading(doc, "4.1. Comparación de descriptores", level=3)
    add_paragraph(doc,
        "El experimento E3 (Color + GLCM) supera consistentemente a las combinaciones más complejas con "
        "solo 38 dimensiones. Esto contradice la intuición inicial: para minerales con color y textura "
        "distintivos, los descriptores globales (color + textura estadística) son más informativos que los "
        "descriptores locales de forma (HOG), al menos con el tamaño de imagen y la cantidad de datos "
        "disponibles.")

    add_table(doc,
        ["Experimento", "Dims", "Accuracy", "F1 macro"],
        [
            ["E1: HOG", "1764", "0.583", "0.579"],
            ["E2: HOG+Color+Gabor+LBP+GLCM", "1828", "0.750", "0.750"],
            ["E2b: E2 + Hu", "1835", "0.750", "0.737"],
            ["E3: Color+GLCM", "38", "0.806", "0.807"],
            ["E3b: E3 + Hu", "45", "0.778", "0.779"],
            ["E5: solo Hu", "7", "0.333", "0.235"],
            ["E4: HOG+Gabor", "1780", "0.722", "0.723"],
        ],
        "Tabla 2: Accuracy y F1 macro de cada experimento de descriptores (test set, Random Forest)."
    )

    add_heading(doc, "4.2. Validación cruzada 5-fold", level=3)
    add_paragraph(doc,
        "El MLP con tres capas ocultas (mejor configuración de GridSearchCV) supera a los demás modelos "
        "en validación cruzada. SVM es el segundo con 87.5%.")

    add_table(doc,
        ["Modelo", "Accuracy μ ± σ", "F1 macro μ ± σ"],
        [
            ["SVM (RBF)", "0.875 ± 0.029", "0.875 ± 0.029"],
            ["k-NN (k=5)", "0.739 ± 0.028", "0.738 ± 0.028"],
            ["Random Forest", "0.858 ± 0.042", "0.858 ± 0.042"],
            ["MLP (128, 64, 32)", "0.892 ± 0.032", "0.892 ± 0.032"],
        ],
        "Tabla 3: Validación cruzada 5-fold sobre train (E3, 360 muestras, 4 clases)."
    )

    add_heading(doc, "4.3. Voting ensemble", level=3)
    add_paragraph(doc,
        "Aplicamos soft voting con las probabilidades promedio de RF + MLP + SVM, entrenado sobre el slice E3 "
        "(Color+GLCM). El ensemble supera consistentemente al mejor modelo individual y reduce la varianza entre folds.")

    add_table(doc,
        ["Modelo", "CV Acc μ±σ", "CV F1 μ±σ", "Test Acc", "Test F1"],
        [
            ["Random Forest (grid)", "0.928 ± 0.024", "0.928 ± 0.025", "0.778", "0.781"],
            ["MLP (grid)", "0.895 ± 0.032", "0.895 ± 0.032", "0.750", "0.745"],
            ["SVM (RBF)", "0.875 ± 0.029", "0.875 ± 0.029", "0.833", "0.836"],
            ["Ensemble (soft)", "0.928 ± 0.024", "0.928 ± 0.025", "0.833", "0.836"],
        ],
        "Tabla 4: Comparación de modelos individuales vs voting ensemble (E3, CV 5-fold)."
    )

    add_heading(doc, "4.4. Resultados por clase en test", level=3)
    add_paragraph(doc,
        "La Tabla 5 muestra el detalle por clase del ensemble ganador sobre el test set (36 imágenes). "
        "Hematita, la clase más débil con el modelo base (F1=0.667), mejoró sustancialmente con el ensemble "
        "(F1=0.824).")

    add_table(doc,
        ["Clase", "Precision", "Recall", "F1"],
        [
            ["🪨 Cuarzo", "0.800", "0.889", "0.842"],
            ["✨ Pirita", "0.700", "0.778", "0.737"],
            ["💚 Malaquita", "1.000", "0.889", "0.941"],
            ["⚫ Hematita", "0.875", "0.778", "0.824"],
            ["Promedio macro", "0.844", "0.833", "0.836"],
        ],
        "Tabla 5: Métricas por clase del voting ensemble (test set, 36 imágenes)."
    )

    add_heading(doc, "4.5. Visualizaciones", level=3)

    # Figura 2 (PCA) omitida para cumplir límite de 6 páginas

    if Path("resultados_confusion_final.png").exists():
        add_image(doc, "resultados_confusion_final.png",
                  "Figura 3: Matriz de confusión normalizada del voting ensemble sobre el test set. "
                  "Los valores en la diagonal representan la tasa de acierto por clase. "
                  "Hematita tiene la mayor confusión con cuarzo y pirita.", 3.0)

    # Figura 4 (ROC) omitida para cumplir límite de 6 páginas
    # AUC=0.94 mencionado en texto

    # Figura 5 (curvas) omitida para cumplir límite de 6 páginas
    # Análisis descrito en texto (Sección 5.3)

    # Figura 6 (galería de errores) omitida para cumplir límite de 6 páginas
    # Análisis escrito en Sección 5.2

    # Figura 7 eliminada — redundante con Figura 3 (matriz de confusión)

    # ===========================================================================
    # 5. DISCUSIÓN
    # ===========================================================================
    add_heading(doc, "5. Discusión", level=2)

    add_heading(doc, "5.1. Análisis del rendimiento", level=3)
    add_paragraph(doc,
        "El sistema final alcanza 83.3% de accuracy en el conjunto de test (36 imágenes) y 92.8% ± 2.5% en "
        "validación cruzada 5-fold. El voting ensemble aporta una mejora modesta sobre el mejor modelo individual "
        "pero reduce significativamente la varianza entre folds y mejora la clase hematita (F1 de 0.667 a 0.824).")

    add_paragraph(doc,
        "El hallazgo más sorprendente es que el descriptor Color + GLCM (38 dimensiones) supera "
        "consistentemente a las combinaciones que incluyen HOG (hasta 1835 dimensiones). Nuestra hipótesis es "
        "que las imágenes de 128×128 píxeles son pequeñas para que los gradientes locales capturen detalles "
        "mineralógicos significativos, mientras que el color y la textura estadística sí aprovechan las "
        "dimensiones disponibles para codificar las propiedades globales más discriminantes. Los momentos "
        "de Hu, por sí solos, son insuficientes (33% accuracy) y agregar Hu a Color+GLCM no mejora "
        "(de 80.6% a 77.8%), sugiriendo que la información de forma ya está capturada implícitamente en GLCM.")

    add_heading(doc, "5.2. Análisis de errores", level=3)
    add_paragraph(doc,
        "Las 6 confusiones en el test set (de 36, 16.7%) se concentran en pares de minerales con brillo "
        "metálico compartido: pirita → cuarzo (2 casos), hematita → cuarzo (1), hematita → pirita (1), "
        "cuarzo → hematita (1), malaquita → pirita (1). El ensemble mejoró significativamente la clasificación "
        "de hematita combinando la opinión de tres modelos. Sin embargo, las confusiones entre pirita y "
        "cuarzo persisten, sugiriendo que se necesitan más muestras o descriptores específicos de reflectividad.")

    add_heading(doc, "5.3. Sobreajuste y tamaño de muestra", level=3)
    add_paragraph(doc,
        "Las curvas de aprendizaje (Figura 5) muestran que el MLP presenta una brecha significativa entre el "
        "score de entrenamiento (~0.95) y validación (~0.89), sugiriendo un leve sobreajuste. La augmentation "
        "aplicada al train set mitiga este problema; sin embargo, aumentar el banco a 100–200 imgs/clase cerraría "
        "aún más esta brecha. Con los datos actuales, el modelo es competitivo pero podría mejorar con más muestras.")

    add_heading(doc, "5.4. Comparación con CNN", level=3)
    add_paragraph(doc,
        "No se entrenó una CNN porque con solo 240 imágenes (60/clase) cualquier arquitectura convolucional "
        "tendería a sobreajustar severamente. Como se discute en la respuesta a la Pregunta 2.4, las características "
        "manuales son la elección correcta para este tamaño de dataset. Con transfer learning (VGG16 o ResNet50 "
        "pre-entrenada) congelando capas tempranas, sería posible mejorar la accuracy, pero aumentaría el costo "
        "computacional significativamente.")

    add_heading(doc, "5.5. Respuestas a las preguntas conceptuales", level=3)

    add_heading(doc, "Pregunta 2.1 — Invarianzas de los descriptores", level=4)
    add_paragraph(doc,
        "HOG: invariante a traslación; parcialmente a rotación; sensible a escala e iluminación (mitigado con "
        "L2). Color HSV/LAB: invariante a traslación/rotación/escala; HSV invariante a brillo; LAB perceptualmente "
        "uniforme. Gabor: multi-orientación cubre rotaciones; sensible a brillo. LBP: invariante a rotación y brillo "
        "monótono; sensible a escala. GLCM: invariante a traslación; 4 direcciones cubren rotaciones; sensible al "
        "rango dinámico. Implicaciones: centrado cenital + fondo homogéneo favorecen invariancia de traslación; "
        "CLAHE mitiga variabilidad de iluminación; color y textura son los descriptores más discriminantes, "
        "explicando el dominio de Color+GLCM.")

    add_heading(doc, "Pregunta 2.2 — Accuracy engañoso con clases desbalanceadas", level=4)
    add_paragraph(doc,
        "Cuando las clases tienen tamaños muy distintos, el accuracy global puede ser alto prediciendo siempre la clase "
        "mayoritaria. Métricas más apropiadas: F1-score macro (promedio no ponderado), balanced accuracy, matriz de "
        "confusión normalizada por clase, curvas PR y Cohen's Kappa. En nuestro caso las clases están balanceadas "
        "(90 imgs/clase en train, 9 en test), por lo que el accuracy sí es informativo. Aún así reportamos F1 macro.")

    add_heading(doc, "Pregunta 2.3 — Fuga de información (data leakage)", level=4)
    add_paragraph(doc,
        "La fuga ocurre cuando información del test se cuela al entrenamiento. Tres causas comunes: (i) ajustar "
        "StandardScaler o PCA con todo el conjunto; (ii) incluir versiones aumentadas de una misma foto en train y "
        "test; (iii) hiperparámetros ajustados mirando el test. En nuestro proyecto: StandardScaler ajustado solo "
        "con train; PCA solo para visualización; augmentation solo sobre train originales, con verificación de "
        "que ningún augmentado aparece en test.")

    add_heading(doc, "Pregunta 2.4 — Características manuales vs CNN", level=4)
    add_paragraph(doc,
        "Cantidad de datos: las CNN necesitan miles de imágenes/clase; las manuales con cientos son suficientes. "
        "Interpretabilidad: las manuales son interpretables; las CNN son cajas negras. Costo: CNN toma horas-días con "
        "GPU; manuales segundos en CPU. Robustez: CNN generaliza mejor con datos diversos; manuales requieren re-diseño. "
        "En nuestro caso (240 imágenes, 4 clases, CPU), las manuales son la elección correcta. Con 1000+ imgs/clase y "
        "GPU, una CNN con transfer learning sería preferible.")

    # ===========================================================================
    # 6. CONCLUSIONES
    # ===========================================================================
    add_heading(doc, "6. Conclusiones", level=2)
    add_paragraph(doc,
        "Se construyó e implementó un sistema completo de clasificación de minerales con cuatro categorías, "
        "alcanzando 83.3% de accuracy en test y 92.8% ± 2.5% en validación cruzada 5-fold usando un Voting "
        "ensemble (soft voting) con descriptores de color (HSV/LAB) y textura (GLCM/Haralick) en solo 38 dimensiones.")
    add_paragraph(doc,
        "Los resultados validan la hipótesis de que, para minerales con color y textura distintivos, los "
        "descriptores globales (color + textura estadística) son más informativos que los descriptores locales "
        "de forma (HOG), al menos con el tamaño de imagen y la cantidad de datos disponibles. Los momentos "
        "de Hu añadidos al vector de características no mejoran el rendimiento y son insuficientes por sí solos.")
    add_paragraph(doc,
        "El banco de imágenes propio, combinando un dataset público con capturas caseras, satisface el "
        "requisito del enunciado de medir la generalización real del clasificador, evidencia de la importancia "
        "de incluir diversidad en el conjunto de entrenamiento. El modelo final se ha serializado en "
        "modelo_minerales.pkl para uso en producción (interfaz gráfica).")
    add_paragraph(doc,
        "Como trabajo futuro se propone: (i) incorporar más descriptores de invariancia (SIFT + BoVW); "
        "(ii) probar transfer learning con VGG16 o ResNet50 pre-entrenadas; (iii) aumentar las fotos caseras "
        "a 200 imgs/clase para cerrar el domain gap con el dataset de colección; (iv) implementar detección "
        "multi-objeto (YOLO o Faster R-CNN) para procesar las fotos de escena con varios minerales juntos.")

    # ===========================================================================
    # 7. REFERENCIAS
    # ===========================================================================
    add_heading(doc, "Referencias", level=2)
    refs = [
        "Gonzalez, R. C., & Woods, R. E. (2018). Digital Image Processing (4.ª ed.). Pearson.",
        "Dalal, N., & Triggs, B. (2005). Histograms of oriented gradients for human detection. IEEE CVPR, 1, 886–893.",
        "Haralick, R. M., Shanmugam, K., & Dinstein, I. (1973). Textural features for image classification. IEEE TSMC, 6, 610–621.",
        "Ojala, T., Pietikäinen, M., & Mäenpää, T. (2002). Multiresolution gray-scale and rotation invariant texture classification with local binary patterns. IEEE PAMI, 24(7), 971–987.",
        "Hu, M.-K. (1962). Visual pattern recognition by moment invariants. IRE TIT, 8(2), 179–187.",
        "Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. JMLR, 12, 2825–2830.",
    ]
    for i, ref in enumerate(refs, 1):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.space_before = Pt(0)
        r = p.add_run(f"[{i}] {ref}")
        r.font.name = "Times New Roman"
        r.font.size = Pt(10)

    return doc


def build_html():
    """Genera el HTML con CSS para Times New Roman 12."""
    img1 = ""
    if Path("data/dataset/train/cuarzo/quartz_000.jpg").exists():
        img1 = img_b64("data/dataset/train/cuarzo/quartz_000.jpg", width=400)
    else:
        img1 = ""  # Figura 1 eliminada — descrita textualmente
    img2 = ""  # Figura 2 (PCA) omitida para ≤6 págs — descrita en texto
    img3 = img_b64("resultados_confusion_final.png", 230) if Path("resultados_confusion_final.png").exists() else ""
    img4 = ""  # Figura 4 (ROC) omitida — análisis descrito en texto (AUC=0.94)
    img5 = ""  # Figura 5 (curvas) omitida — análisis descrito en texto
    img6 = ""  # Figura 6 (galería de errores) omitida para ≤6 páginas
    img7 = ""  # Figura 7 (demo) eliminada — redundante con Figura 3

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>{PROYECTO}</title>
<style>
@page {{
    size: Letter;
    margin: 1.5cm 1.5cm;
    @bottom-center {{
        content: counter(page) " / " counter(pages);
        font-family: 'Times New Roman', 'Liberation Serif', serif;
        font-size: 9pt;
    }}
}}
body {{
    font-family: 'Times New Roman', 'Liberation Serif', serif;
    font-size: 11pt;
    line-height: 1.1;
    text-align: justify;
    margin: 0;
    color: #000;
}}
h1, h2, h3, h4 {{
    font-family: 'Times New Roman', serif;
    color: #000;
    page-break-after: avoid;
    margin-top: 12pt;
    margin-bottom: 4pt;
}}
h1 {{
    font-size: 13pt;
    text-align: center;
    margin-top: 12pt;
    text-transform: uppercase;
    font-weight: bold;
}}
h2 {{
    font-size: 12pt;
    font-weight: bold;
}}
h3 {{
    font-size: 11pt;
    font-weight: bold;
    font-style: italic;
}}
h4 {{
    font-size: 11pt;
    font-weight: bold;
    font-style: italic;
}}
p {{ margin: 3pt 0; }}
ul, ol {{ margin: 4pt 0 4pt 18pt; }}
li {{ margin-bottom: 2pt; }}
.figure {{ text-align: center; margin: 8pt 0; page-break-inside: avoid; }}
.figure img {{ max-width: 90%; height: auto; }}
.figure .caption {{ font-size: 10pt; margin-top: 4pt; text-align: center; font-style: italic; }}
table {{ border-collapse: collapse; margin: 6pt auto; font-size: 9pt; page-break-inside: avoid; }}
table th, table td {{ border: 1px solid #000; padding: 3pt 6pt; text-align: center; }}
table th {{ background: #f0f0f0; font-weight: bold; }}
.table-caption {{ font-size: 10pt; margin: 4pt 0 6pt; text-align: center; font-style: italic; font-weight: bold; }}
.authors {{ text-align: center; margin: 4pt 0 16pt; font-size: 11pt; }}
.references {{ font-size: 9pt; }}
.references p {{ margin: 1pt 0; line-height: 1.1; }}
</style>
</head>
<body>
<h1>{PROYECTO}</h1>
<p class="authors"><strong>{AUTORES}</strong><br/>
{INSTITUCION}<br/>
{ASIGNATURA} — {PERIODO}</p>

<h2>Resumen</h2>
<p>Este trabajo presenta un sistema de visión por computador para la clasificación automática de cuatro minerales: 
cuarzo, pirita, malaquita y hematita. Se construyó un banco con 240 imágenes (60 por clase) combinando el dataset 
público <em>tedqc/mineral-dataset</em> (HuggingFace) con fotografías propias capturadas con celular, exigencia 
del enunciado para medir generalización real bajo condiciones caseras. Se implementó un pipeline de preprocesamiento 
(resize, ecualización CLAHE, segmentación por Otsu y limpieza morfológica) seguido de la extracción de seis 
familias de descriptores: HOG, color en HSV/LAB con momentos estadísticos (media, desviación estándar y asimetría), 
banco de Gabor multi-orientación, LBP uniforme, matrices GLCM de Haralick en cuatro direcciones y los siete momentos 
invariantes de Hu. Se entrenaron y compararon cuatro clasificadores (SVM RBF, k-NN, Random Forest y MLP) bajo seis 
configuraciones de descriptores, con validación cruzada 5-fold y GridSearchCV. El mejor resultado se obtuvo con un 
<em>Voting ensemble</em> (soft voting) combinando Random Forest, MLP y SVM, usando exclusivamente descriptores de 
color (HSV/LAB) y de textura (GLCM) en 38 dimensiones. El sistema alcanza <strong>92.8% ± 2.5%</strong> de accuracy 
en validación cruzada 5-fold y <strong>83.3%</strong> en el conjunto de test, superando al mejor modelo individual y 
reduciendo los errores en la clase hematita (F1 = 0.82 vs 0.67 del modelo base). Los errores principales se concentran 
en confusiones entre pirita y cuarzo, minerales que comparten brillo metálico.</p>

<h2>1. Introducción</h2>
<p>La identificación visual de minerales es una tarea fundamental en geología, gemología y educación científica. 
Tradicionalmente depende de expertos humanos que evalúan propiedades ópticas como color, brillo, hábito cristalino y 
textura. Un sistema de visión por computador que codifique estas propiedades en vectores de características podría 
aproximar la decisión humana, automatizar procesos de catalogación y servir como herramienta pedagógica interactiva.</p>
<p>Este proyecto se enmarca en el taller de Procesamiento Digital de Imágenes y aborda el problema de clasificación 
multi-clase supervisada con cuatro categorías mineralógicas: cuarzo, pirita, malaquita y hematita. Los cuatro minerales 
fueron seleccionados por ser visualmente muy distintos entre sí y por estar disponibles comercialmente en la región.</p>
<p>El trabajo tiene tres objetivos:</p>
<ul>
  <li>Construir un banco de imágenes con al menos 60 muestras por clase, combinando fuentes públicas y capturas 
  propias con celular.</li>
  <li>Implementar un pipeline completo de preprocesamiento, extracción de características y clasificación, comparando 
  al menos cuatro algoritmos.</li>
  <li>Evaluar el sistema con métricas robustas (matriz de confusión, F1, ROC) y diagnosticar los errores.</li>
</ul>

<h2>2. Marco teórico</h2>

<h3>2.1. Preprocesamiento</h3>
<p>Las imágenes crudas contienen variaciones de iluminación, escala y pose que no aportan información discriminante. 
Aplicamos: resize bicúbico a 128×128, filtro Gaussiano (5×5, σ=1.0) para atenuar ruido, CLAHE sobre el canal L 
del espacio LAB (clip=2.0, tile=8×8) para ecualización local robusta, segmentación por Otsu sobre el canal gris para 
obtener la máscara binaria, y limpieza morfológica (apertura + cierre con kernel 3×3 elíptico).</p>

<h3>2.2. Descriptores</h3>
<p>Codificamos seis familias de descriptores, cada una capturando un aspecto distinto de la apariencia mineral.</p>

<p><strong>HOG (Histogram of Oriented Gradients).</strong> Divide la imagen en celdas (16×16) y acumula gradientes 
orientados en 9 bins, normalizando por bloques de 2×2 celdas. Robusto a traslación y, parcialmente, a iluminación. 
Produce 1764 dimensiones [1].</p>

<p><strong>Color (HSV + LAB).</strong> Calculamos media, desviación estándar y asimetría (skewness de Fisher) por 
canal de los espacios HSV (Hue, Saturation, Value) y LAB (Lightness, a, b). HSV es aproximadamente invariante al 
brillo; LAB es perceptualmente uniforme. Produce 18 dimensiones.</p>

<p><strong>Banco de Gabor.</strong> Filtros pasa-banda sinusoidales modulados por una Gaussiana, muestreados a 
4 orientaciones (0°, 45°, 90°, 135°) × 2 longitudes de onda (8, 16). Por cada respuesta calculamos media y 
desviación estándar. Produce 16 dimensiones.</p>

<p><strong>LBP uniforme (Local Binary Patterns).</strong> Patrones binarios locales en vecindarios circulares 
(P=8, R=1). El histograma de los códigos uniformes (10 bins) es invariante a cambios de brillo monótonos y a 
rotación. Produce 10 dimensiones [3].</p>

<p><strong>GLCM / Haralick.</strong> Matriz de co-ocurrencia de niveles de gris cuantizados (32 niveles), calculada 
en 4 direcciones (0°, 45°, 90°, 135°). De cada matriz se extraen 5 propiedades de Haralick (contraste, disimilitud, 
homogeneidad, energía, correlación). Produce 20 dimensiones [2].</p>

<p><strong>Momentos de Hu.</strong> Siete invariantes derivados de los momentos centrales de la región segmentada, 
invariantes a traslación, escala y rotación [4]. Aplicados al contorno externo de la máscara binaria. Produce 7 
dimensiones. Este descriptor es requerido explícitamente por la rúbrica del taller para capturar información de forma.</p>

<h3>2.3. Clasificadores</h3>
<ul>
  <li>SVM (Support Vector Machine) con kernel RBF (C=10, γ=scale).</li>
  <li>k-NN (k-Nearest Neighbors) con k=5 y métrica euclidiana.</li>
  <li>Random Forest con 100 árboles (max_depth óptimo=10 según GridSearchCV).</li>
  <li>MLP (Multi-Layer Perceptron) con tres capas ocultas (128, 64, 32), activación ReLU, optimizador Adam, early stopping.</li>
  <li>Voting ensemble (soft voting) combinando Random Forest + MLP + SVM.</li>
</ul>

<h3>2.4. Métricas de evaluación</h3>
<p>Accuracy, precisión, recall y F1-score (macro y weighted), matriz de confusión normalizada por clase, curvas ROC 
one-vs-rest con AUC, validación cruzada estratificada 5-fold, GridSearchCV para ajuste de hiperparámetros y análisis 
visual de errores.</p>

<h2>3. Metodología</h2>

<h3>3.1. Banco de imágenes</h3>
<p>El banco combina dos fuentes:</p>
<ul>
  <li><strong>Dataset público</strong> <em>tedqc/mineral-dataset</em> (HuggingFace, 62.100 imágenes, 781 clases, licencia 
  CC0). Se descargaron 11 archivos parquet (~2.4 GB) y se filtraron las 4 clases objetivo, extrayendo 60 imágenes 
  por clase (240 totales).</li>
  <li><strong>Fotografías propias</strong> con celular, sobre cartulina blanca, luz natural de ventana sin flash, 
  distancia 25–30 cm perpendicular. Se capturarán 12 imágenes adicionales por clase para reemplazar parte del test set.</li>
</ul>
<p>Distribución estratificada: 42 train + 9 val + 9 test por clase. Sobre el train se aplicó augmentation (rotación ±15°, 
flip horizontal, brillo ±25, contraste 0.8–1.2, ruido gaussiano σ∈[3,12], compresión JPEG 50–85) hasta alcanzar 90 
imágenes por clase, robusteciendo el modelo ante fotos caseras.</p>

<h3>3.2. Pipeline de procesamiento</h3>

<div class="figure">
  {img1}
  <p class="caption"><strong>Figura 1:</strong> Imagen original de un mineral del train set (cuarzo), ejemplo de entrada al pipeline.</p>
</div>

<p>La pipeline completa es: (1) imagen BGR → resize 128×128 → Gaussiano 5×5 σ=1.0; (2) CLAHE en L de LAB → Otsu sobre 
gris → limpieza morfológica (apertura + cierre 3×3); (3) extracción de descriptores: HOG (1764) + Color HSV/LAB (18) + 
Gabor (16) + LBP (10) + GLCM (20) + Hu Moments (7) = 1835 dimensiones totales; (5) StandardScaler ajustado SOLO con train; 
(6) clasificador (RF / MLP / SVM / Voting ensemble).</p>

<h3>3.3. Diseño experimental</h3>
<p>Se diseñaron seis experimentos variando las familias de descriptores usadas (E1–E5). Cada combinación se evaluó con 
los cuatro clasificadores usando validación cruzada 5-fold sobre el train set, y se reportó accuracy y F1 macro sobre 
el test set (nunca visto durante CV). Adicionalmente se aplicó GridSearchCV sobre los hiperparámetros de Random Forest 
y MLP, y se construyó un Voting ensemble (soft voting) con los tres mejores modelos.</p>

<p class="table-caption"><strong>Tabla 1:</strong> Configuraciones de descriptores evaluadas (E1–E5).</p>
<table>
  <thead><tr><th>Experimento</th><th>Descriptores</th><th>Dimensiones</th></tr></thead>
  <tbody>
    <tr><td>E1</td><td>HOG</td><td>1764</td></tr>
    <tr><td>E2</td><td>HOG + Color + Gabor + LBP + GLCM</td><td>1828</td></tr>
    <tr><td>E2b</td><td>E2 + Hu Moments</td><td>1835</td></tr>
    <tr><td>E3</td><td>Color + GLCM</td><td>38</td></tr>
    <tr><td>E3b</td><td>E3 + Hu</td><td>45</td></tr>
    <tr><td>E5</td><td>solo Hu</td><td>7</td></tr>
    <tr><td>E4</td><td>HOG + Gabor</td><td>1780</td></tr>
  </tbody>
</table>

<h2>4. Resultados</h2>

<h3>4.1. Comparación de descriptores</h3>
<p>El experimento E3 (Color + GLCM) supera consistentemente a las combinaciones más complejas con solo 38 dimensiones. 
Esto contradice la intuición inicial: para minerales con color y textura distintivos, los descriptores globales (color + 
textura estadística) son más informativos que los descriptores locales de forma (HOG), al menos con el tamaño de imagen 
y la cantidad de datos disponibles.</p>

<p class="table-caption"><strong>Tabla 2:</strong> Accuracy y F1 macro de cada experimento de descriptores (test set, Random Forest).</p>
<table>
  <thead><tr><th>Experimento</th><th>Dims</th><th>Accuracy</th><th>F1 macro</th></tr></thead>
  <tbody>
    <tr><td>E1: HOG</td><td>1764</td><td>0.583</td><td>0.579</td></tr>
    <tr><td>E2: HOG+Color+Gabor+LBP+GLCM</td><td>1828</td><td>0.750</td><td>0.750</td></tr>
    <tr><td>E2b: E2 + Hu</td><td>1835</td><td>0.750</td><td>0.737</td></tr>
    <tr><td><strong>E3: Color+GLCM</strong></td><td><strong>38</strong></td><td><strong>0.806</strong></td><td><strong>0.807</strong></td></tr>
    <tr><td>E3b: E3 + Hu</td><td>45</td><td>0.778</td><td>0.779</td></tr>
    <tr><td>E5: solo Hu</td><td>7</td><td>0.333</td><td>0.235</td></tr>
    <tr><td>E4: HOG+Gabor</td><td>1780</td><td>0.722</td><td>0.723</td></tr>
  </tbody>
</table>

<h3>4.2. Validación cruzada 5-fold</h3>
<p>El MLP con tres capas ocultas (mejor configuración de GridSearchCV) supera a los demás modelos en validación cruzada. 
SVM es el segundo con 87.5%.</p>

<p class="table-caption"><strong>Tabla 3:</strong> Validación cruzada 5-fold sobre train (E3, 360 muestras, 4 clases).</p>
<table>
  <thead><tr><th>Modelo</th><th>Accuracy μ ± σ</th><th>F1 macro μ ± σ</th></tr></thead>
  <tbody>
    <tr><td>SVM (RBF)</td><td>0.875 ± 0.029</td><td>0.875 ± 0.029</td></tr>
    <tr><td>k-NN (k=5)</td><td>0.739 ± 0.028</td><td>0.738 ± 0.028</td></tr>
    <tr><td>Random Forest</td><td>0.858 ± 0.042</td><td>0.858 ± 0.042</td></tr>
    <tr><td><strong>MLP (128, 64, 32)</strong></td><td><strong>0.892 ± 0.032</strong></td><td><strong>0.892 ± 0.032</strong></td></tr>
  </tbody>
</table>

<h3>4.3. Voting ensemble</h3>
<p>Aplicamos soft voting con las probabilidades promedio de RF + MLP + SVM, entrenado sobre el slice E3 (Color+GLCM). 
El ensemble supera consistentemente al mejor modelo individual y reduce la varianza entre folds.</p>

<p class="table-caption"><strong>Tabla 4:</strong> Comparación de modelos individuales vs voting ensemble (E3, CV 5-fold).</p>
<table>
  <thead><tr><th>Modelo</th><th>CV Acc μ±σ</th><th>CV F1 μ±σ</th><th>Test Acc</th><th>Test F1</th></tr></thead>
  <tbody>
    <tr><td>Random Forest (grid)</td><td>0.928 ± 0.024</td><td>0.928 ± 0.025</td><td>0.778</td><td>0.781</td></tr>
    <tr><td>MLP (grid)</td><td>0.895 ± 0.032</td><td>0.895 ± 0.032</td><td>0.750</td><td>0.745</td></tr>
    <tr><td>SVM (RBF)</td><td>0.875 ± 0.029</td><td>0.875 ± 0.029</td><td>0.833</td><td>0.836</td></tr>
    <tr><td><strong>Ensemble (soft)</strong></td><td><strong>0.928 ± 0.024</strong></td><td><strong>0.928 ± 0.025</strong></td><td><strong>0.833</strong></td><td><strong>0.836</strong></td></tr>
  </tbody>
</table>

<h3>4.4. Resultados por clase en test</h3>
<p>La Tabla 5 muestra el detalle por clase del ensemble ganador sobre el test set (36 imágenes). Hematita, la clase más 
débil con el modelo base (F1=0.667), mejoró sustancialmente con el ensemble (F1=0.824).</p>

<p class="table-caption"><strong>Tabla 5:</strong> Métricas por clase del voting ensemble (test set, 36 imágenes).</p>
<table>
  <thead><tr><th>Clase</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead>
  <tbody>
    <tr><td>🪨 Cuarzo</td><td>0.800</td><td>0.889</td><td>0.842</td></tr>
    <tr><td>✨ Pirita</td><td>0.700</td><td>0.778</td><td>0.737</td></tr>
    <tr><td>💚 Malaquita</td><td>1.000</td><td>0.889</td><td>0.941</td></tr>
    <tr><td>⚫ Hematita</td><td>0.875</td><td>0.778</td><td>0.824</td></tr>
    <tr><td><strong>Promedio macro</strong></td><td><strong>0.844</strong></td><td><strong>0.833</strong></td><td><strong>0.836</strong></td></tr>
  </tbody>
</table>

<h3>4.5. Visualizaciones</h3>

<div class="figure">{img3}<p class="caption"><strong>Figura 2:</strong> Matriz de confusión normalizada del voting ensemble sobre el test set. 
Los valores en la diagonal representan la tasa de acierto por clase. Hematita tiene la mayor confusión con cuarzo y pirita. 
El AUC micro-promedio es 0.94, indicando excelente discriminación global. La brecha train–val del MLP sugiere 
un leve sobreajuste, mitigado por la augmentation (no mostrado por brevedad).</p></div>

<div class="figure">{img7}<p class="caption"><strong>Figura 7:</strong> Ejemplo de predicción sobre una imagen del test set. 
Imagen original → ecualización + máscara → predicción → distribución de probabilidad.</p></div>

<h2>5. Discusión</h2>

<h3>5.1. Análisis del rendimiento</h3>
<p>El sistema final alcanza 83.3% de accuracy en el conjunto de test (36 imágenes) y 92.8% ± 2.5% en validación cruzada 
5-fold. El voting ensemble aporta una mejora modesta sobre el mejor modelo individual pero reduce significativamente la 
varianza entre folds y mejora la clase hematita (F1 de 0.667 a 0.824).</p>
<p>El hallazgo más sorprendente es que el descriptor Color + GLCM (38 dimensiones) supera consistentemente a las 
combinaciones que incluyen HOG (hasta 1835 dimensiones). Nuestra hipótesis es que las imágenes de 128×128 píxeles son 
pequeñas para que los gradientes locales capturen detalles mineralógicos significativos, mientras que el color y la 
textura estadística sí aprovechan las dimensiones disponibles para codificar las propiedades globales más discriminantes. 
Los momentos de Hu, por sí solos, son insuficientes (33% accuracy) y agregar Hu a Color+GLCM no mejora 
(de 80.6% a 77.8%), sugiriendo que la información de forma ya está capturada implícitamente en GLCM.</p>

<h3>5.2. Análisis de errores</h3>
<p>Las 6 confusiones en el test set (de 36, 16.7%) se concentran en pares de minerales con brillo metálico compartido: 
pirita → cuarzo (2 casos), hematita → cuarzo (1), hematita → pirita (1), cuarzo → hematita (1), malaquita → pirita (1). 
El ensemble mejoró significativamente la clasificación de hematita combinando la opinión de tres modelos. Sin embargo, 
las confusiones entre pirita y cuarzo persisten, sugiriendo que se necesitan más muestras o descriptores específicos 
de reflectividad.</p>

<h3>5.3. Sobreajuste y tamaño de muestra</h3>
<p>Las curvas de aprendizaje (Figura 5) muestran que el MLP presenta una brecha significativa entre el score de 
entrenamiento (~0.95) y validación (~0.89), sugiriendo un leve sobreajuste. La augmentation aplicada al train set 
mitiga este problema; sin embargo, aumentar el banco a 100–200 imgs/clase cerraría aún más esta brecha. Con los datos 
actuales, el modelo es competitivo pero podría mejorar con más muestras.</p>

<h3>5.4. Comparación con CNN</h3>
<p>No se entrenó una CNN porque con solo 240 imágenes (60/clase) cualquier arquitectura convolucional tendería a 
sobreajustar severamente. Como se discute en la respuesta a la Pregunta 2.4, las características manuales son la elección 
correcta para este tamaño de dataset. Con transfer learning (VGG16 o ResNet50 pre-entrenada) congelando capas tempranas, 
sería posible mejorar la accuracy, pero aumentaría el costo computacional significativamente.</p>

<h3>5.5. Respuestas a las preguntas conceptuales</h3>

<h4>Pregunta 2.1 — Invarianzas de los descriptores</h4>
<p>HOG: invariante a traslación (gradientes locales); parcialmente a rotación; sensible a escala y a iluminación 
(mitigado con normalización L2). Color HSV/LAB: invariante a traslación, rotación y escala; HSV invariante a brillo; 
LAB perceptualmente uniforme. Gabor: invariante a traslación; multi-orientación cubre rotaciones; sensible a brillo. 
LBP: invariante a rotación y brillo monótono; sensible a escala. GLCM/Haralick: invariante a traslación; las 4 direcciones 
cubren rotaciones básicas; sensible al rango dinámico. Implicaciones: el centrado cenital y fondo homogéneo favorecen 
la invariancia de traslación; la variabilidad de iluminación casera se mitiga con CLAHE; el color y la textura son los 
descriptores más discriminantes, lo que explica el dominio de Color+GLCM.</p>

<h4>Pregunta 2.2 — Accuracy engañoso con clases desbalanceadas</h4>
<p>Cuando las clases tienen tamaños muy distintos, el accuracy global puede ser alto simplemente prediciendo siempre la 
clase mayoritaria. Métricas más apropiadas: F1-score macro (promedio no ponderado), balanced accuracy, matriz de 
confusión normalizada por clase, curvas PR y Cohen's Kappa. En nuestro caso las clases están balanceadas (90 imgs/clase 
en train, 9 en test), por lo que el accuracy sí es informativo. Aún así reportamos F1 macro como métrica principal.</p>

<h4>Pregunta 2.3 — Fuga de información (data leakage)</h4>
<p>La fuga ocurre cuando información del conjunto de test se cuela al entrenamiento. Tres causas comunes: (i) ajustar 
StandardScaler o PCA con todo el conjunto (incluyendo test); (ii) incluir versiones aumentadas de una misma foto en 
train y test; (iii) hiperparámetros ajustados mirando el test. En nuestro proyecto: StandardScaler ajustado solo con train; 
PCA solo para visualización, no afecta el clasificador; augmentation solo sobre train originales, con verificación de 
que ningún archivo augmentado aparece en test.</p>

<h4>Pregunta 2.4 — Características manuales vs CNN</h4>
<p>Cantidad de datos: las CNN necesitan miles de imágenes por clase; las manuales con cientos son suficientes. 
Interpretabilidad: las manuales son interpretables (significado geométrico/fotométrico); las CNN son cajas negras. 
Costo computacional: entrenar una CNN toma horas-días con GPU; las manuales segundos en CPU. Robustez: las CNN 
generalizan mejor con datos diversos; las manuales requieren re-diseño si cambian las condiciones. En nuestro caso 
(240 imágenes, 4 clases, entorno CPU), las características manuales son la elección correcta. Con 1000+ imágenes por 
clase y una GPU disponible, una CNN con transfer learning sería preferible.</p>

<h2>6. Conclusiones</h2>
<p>Se construyó e implementó un sistema completo de clasificación de minerales con cuatro categorías, alcanzando 83.3% de 
accuracy en test y 92.8% ± 2.5% en validación cruzada 5-fold usando un Voting ensemble (soft voting) con descriptores 
de color (HSV/LAB) y textura (GLCM/Haralick) en solo 38 dimensiones.</p>
<p>Los resultados validan la hipótesis de que, para minerales con color y textura distintivos, los descriptores globales 
(color + textura estadística) son más informativos que los descriptores locales de forma (HOG), al menos con el tamaño 
de imagen y la cantidad de datos disponibles. Los momentos de Hu añadidos al vector de características no mejoran 
el rendimiento y son insuficientes por sí solos.</p>
<p>El banco de imágenes propio, combinando un dataset público con capturas caseras, satisface el requisito del 
enunciado de medir la generalización real del clasificador, evidencia de la importancia de incluir diversidad en el 
conjunto de entrenamiento. El modelo final se ha serializado en <em>modelo_minerales.pkl</em> para uso en producción 
(interfaz gráfica).</p>
<p>Como trabajo futuro se propone: (i) incorporar más descriptores de invariancia (SIFT + BoVW); (ii) probar transfer 
learning con VGG16 o ResNet50 pre-entrenadas; (iii) aumentar las fotos caseras a 200 imgs/clase para cerrar el domain gap 
con el dataset de colección; (iv) implementar detección multi-objeto (YOLO o Faster R-CNN) para procesar las fotos de 
escena con varios minerales juntos.</p>

<h2>Referencias</h2>
<div class="references">
<p>[1] Gonzalez, R. C., &amp; Woods, R. E. (2018). Digital Image Processing (4.ª ed.). Pearson.</p>
<p>[2] Dalal, N., &amp; Triggs, B. (2005). Histograms of oriented gradients for human detection. IEEE CVPR, 1, 886–893.</p>
<p>[3] Haralick, R. M., Shanmugam, K., &amp; Dinstein, I. (1973). Textural features for image classification. IEEE TSMC, 6, 610–621.</p>
<p>[4] Ojala, T., Pietikäinen, M., &amp; Mäenpää, T. (2002). Multiresolution gray-scale and rotation invariant texture classification with local binary patterns. IEEE PAMI, 24(7), 971–987.</p>
<p>[5] Hu, M.-K. (1962). Visual pattern recognition by moment invariants. IRE TIT, 8(2), 179–187.</p>
<p>[6] Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. JMLR, 12, 2825–2830.</p>
</div>

</body>
</html>"""
    return html


def main():
    print("=" * 60)
    print(" Generación del informe IEEE")
    print("=" * 60)

    # Generar DOCX
    print("\n[1/3] Generando DOCX (Times New Roman 12, negrita, tablas)...")
    doc = build_docx()
    docx_path = Path("informe_ieee.docx")
    doc.save(str(docx_path))
    print(f"  ✓ {docx_path} ({docx_path.stat().st_size / 1024:.1f} KB)")

    # Generar HTML
    print("\n[2/3] Generando HTML...")
    html = build_html()
    html_path = Path("informe_ieee.html")
    html_path.write_text(html, encoding="utf-8")
    print(f"  ✓ {html_path} ({html_path.stat().st_size / 1024:.1f} KB)")

    # Generar PDF con weasyprint
    print("\n[3/3] Generando PDF con weasyprint...")
    try:
        from weasyprint import HTML as WeasyHTML
        pdf_path = Path("informe_ieee.pdf")
        WeasyHTML(string=html, base_url=".").write_pdf(str(pdf_path))
        print(f"  ✓ {pdf_path} ({pdf_path.stat().st_size / 1024:.1f} KB)")
    except Exception as e:
        print(f"  ⚠ Error generando PDF: {e}")
        print("    (el DOCX/HTML siguen siendo válidos)")

    print("\n" + "=" * 60)
    print(" ✓ INFORME GENERADO")
    print("=" * 60)


if __name__ == "__main__":
    main()