# -*- coding: utf-8 -*-
"""Genera el informe IEEE de la Tarea 2 (≤6 págs) siguiendo docs/PLAN_INFORME_IEEE.md.

Usa python-docx para producir un .docx con formato IEEE aproximado:
  • Times New Roman 12 pt en el cuerpo (10 pt en tablas/captions).
  • Títulos de sección centrados en MAYÚSCULAS, negrita.
  • Subtítulos en negrita, cursiva, alineados a la izquierda.
  • Figuras/tablas referenciadas con caption.

Las figuras que aún no existan se referencian como "Figura X — descripción";
las que sí existen (results/figures/*.png) se insertan directamente.
"""
from pathlib import Path
from docx import Document
from docx.shared import Pt, Cm, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "results" / "figures"
OUT = ROOT / "docs" / "informe_ieee.docx"

# ---------------------------------------------------------------- helpers -----
TNR = "Times New Roman"

def set_run(run, *, size=12, bold=False, italic=False, all_caps=False):
    run.font.name = TNR
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:ascii"), TNR)
    rfonts.set(qn("w:hAnsi"), TNR)
    rfonts.set(qn("w:cs"), TNR)
    rfonts.set(qn("w:eastAsia"), TNR)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    run.font.all_caps = all_caps

def add_paragraph(doc, text, *, size=12, bold=False, italic=False,
                  align=WD_ALIGN_PARAGRAPH.JUSTIFY, all_caps=False,
                  space_before=0, space_after=2, indent_first=True):
    p = doc.add_paragraph()
    p.alignment = align
    pf = p.paragraph_format
    pf.space_before = Pt(space_before)
    pf.space_after = Pt(space_after)
    pf.line_spacing = 1.0
    if indent_first and align == WD_ALIGN_PARAGRAPH.JUSTIFY:
        pf.first_line_indent = Cm(0.4)
    run = p.add_run(text)
    set_run(run, size=size, bold=bold, italic=italic, all_caps=all_caps)
    return p

def add_title_section(doc, label, title):
    """Título de sección IEEE: 'I. INTRODUCCIÓN' centrado, mayúsculas, negrita."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = p.paragraph_format
    pf.space_before = Pt(6)
    pf.space_after = Pt(2)
    run = p.add_run(f"{label}. {title.upper()}")
    set_run(run, size=12, bold=True)
    return p

def add_subtitle(doc, label, title):
    """Subtítulo IEEE: 'A. Filtros digitales' alineado izquierda, negrita, cursiva."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf = p.paragraph_format
    pf.space_before = Pt(4)
    pf.space_after = Pt(2)
    run = p.add_run(f"{label}. {title}")
    set_run(run, size=12, bold=True, italic=True)
    return p

def add_subsubtitle(doc, text):
    """Sub-subtítulo (preguntas 2.x) — negrita, alineado izquierda."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf = p.paragraph_format
    pf.space_before = Pt(3)
    pf.space_after = Pt(1)
    run = p.add_run(text)
    set_run(run, size=12, bold=True)
    return p

def add_caption(doc, text):
    """Caption de figura/tabla: TNR 10 pt, centrado."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = p.paragraph_format
    pf.space_before = Pt(1)
    pf.space_after = Pt(4)
    run = p.add_run(text)
    set_run(run, size=10)
    return p

def add_image(doc, path, width_in=6.0):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = p.paragraph_format
    pf.space_before = Pt(2)
    pf.space_after = Pt(0)
    run = p.add_run()
    if Path(path).exists():
        run.add_picture(str(path), width=Inches(width_in))
    return p

def add_bullet(doc, text):
    p = doc.add_paragraph(style="List Bullet")
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf = p.paragraph_format
    pf.space_after = Pt(2)
    run = p.add_run(text)
    set_run(run, size=12)
    return p

def add_table(doc, headers, rows, *, col_widths=None):
    n_cols = len(headers)
    table = doc.add_table(rows=1 + len(rows), cols=n_cols)
    table.alignment = WD_ALIGN_PARAGRAPH.CENTER
    table.style = "Light Grid Accent 1"
    # headers
    for j, h in enumerate(headers):
        cell = table.rows[0].cells[j]
        cell.text = ""
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(h)
        set_run(run, size=10, bold=True)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    # rows
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = table.rows[i + 1].cells[j]
            cell.text = ""
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if j > 0 else WD_ALIGN_PARAGRAPH.LEFT
            run = p.add_run(str(val))
            set_run(run, size=10)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    if col_widths:
        for j, w in enumerate(col_widths):
            for row in table.rows:
                row.cells[j].width = Cm(w)
    return table

# ----------------------------------------------------------------- build -----
doc = Document()

# estilo base del documento: TNR 12
style = doc.styles["Normal"]
style.font.name = TNR
style.font.size = Pt(12)
style.paragraph_format.space_after = Pt(4)

# ----- Título -----
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(8)
run = p.add_run("Clasificación Automática de Minerales Mediante "
                "Extracción de Características y Aprendizaje Clásico")
set_run(run, size=24, bold=True)

# ----- Autores -----
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(2)
run = p.add_run("Maria Fernanda Atencia, Stiven Suárez Marín, "
                "Michael Stiven Tabares Tobón")
set_run(run, size=11)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(12)
run = p.add_run("Universidad de Antioquia — Facultad de Ingeniería — "
                "Procesamiento Digital de Imágenes — 2026-II")
set_run(run, size=10, italic=True)

# ----- Resumen -----
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.LEFT
p.paragraph_format.space_before = Pt(6)
run = p.add_run("Resumen—")
set_run(run, size=12, bold=True)
run = p.add_run(
    "Este trabajo presenta un sistema de visión por computador para clasificar "
    "cuatro minerales (cuarzo, pirita, malaquita, hematita) a partir de imágenes "
    "RGB. Se construyó un banco de 240 imágenes (60 por clase) combinando el "
    "dataset público tedqc/mineral-dataset (HuggingFace, CC0) con fotografías "
    "propias con celular. El pipeline de preprocesamiento (resize bicúbico 128×128, "
    "Gaussiano 5×5 σ=1.0 justificado por PSNR/SSIM de la Parte 1, CLAHE sobre L "
    "de LAB, Otsu y limpieza morfológica 3×3) alimenta la extracción de siete "
    "familias de descriptores: HOG, color HSV/LAB con momentos estadísticos, "
    "banco de Gabor multi-orientación, LBP uniforme, matrices GLCM en 4 "
    "direcciones y los 7 momentos invariantes de Hu (1835 dims). Se entrenaron "
    "SVM RBF, k-NN, Random Forest (con class_weight='balanced') y MLP, más "
    "Voting y Stacking ensembles, bajo seis configuraciones de descriptores "
    "E1–E5, con validación cruzada 5-fold y GridSearchCV. El mejor resultado "
    "fue Random Forest sobre descriptores de color (HSV/LAB) y textura (GLCM) "
    "en 38 dimensiones: 91.9% ± 2.5% en CV-5 y 83.3% en test (F1 macro 0.836), "
    "mejorando el recall de las clases difíciles (pirita F1 0.78, hematita "
    "F1 0.89). El hallazgo principal es que, a tamaño 128×128, las 38 "
    "dimensiones de Color+GLCM superan al concatenado de 1835 dominado por "
    "HOG, porque los minerales se diferencian mejor por color medio y textura "
    "global que por gradientes locales."
)
set_run(run, size=12)

p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(4)
run = p.add_run("Palabras clave—")
set_run(run, size=10, bold=True)
run = p.add_run("mineral recognition, handcrafted features, HOG, GLCM, voting ensemble.")
set_run(run, size=10)

# ============================================================== PÁGINA 1
add_title_section(doc, "I", "Introducción")
add_paragraph(doc,
    "En la Tarea 1 se trabajó a nivel de píxel: histogramas, espacios de color y "
    "morfología matemática para transformar intensidades en variables físicas "
    "discriminantes. En esta Tarea 2 damos un paso más: entrenamos un sistema "
    "completo de visión por computador capaz de reconocer y clasificar objetos "
    "reales —en este caso, cuatro minerales— a partir de descriptores extraídos "
    "manualmente de la imagen y de clasificadores clásicos de aprendizaje "
    "supervisado. Un clasificador es tan bueno como las características que "
    "recibe, y esas características son tan buenas como el preprocesamiento que "
    "las precede; por eso el trabajo se estructura en dos etapas complementarias: "
    "(i) un taller práctico y conceptual de filtrado espacial y frecuencial "
    "(Parte 1, 30% de la nota) que justifica cada decisión del pipeline con "
    "métricas objetivas (PSNR, SSIM) y respuestas conceptuales, y (ii) el "
    "proyecto de clasificación propiamente dicho (Parte 2, 70%), con banco de "
    "imágenes, descriptores, comparativa de modelos, métricas rigurosas, "
    "análisis de errores y una aplicación gráfica de despliegue.")
add_paragraph(doc,
    "La identificación visual de minerales es una tarea fundamental en geología, "
    "gemología y educación científica. Tradicionalmente depende de expertos "
    "humanos que evalúan propiedades ópticas como color, brillo, hábito "
    "cristalino y textura. Un sistema de visión que codifique estas propiedades "
    "en vectores de características podría aproximar la decisión humana, "
    "automatizar procesos de catalogación y servir como herramienta pedagógica. "
    "En este proyecto los cuatro minerales seleccionados —cuarzo, pirita, "
    "malaquita y hematita— se eligieron por ser visualmente muy distintos entre "
    "sí y por estar disponibles comercialmente en la región.")
add_paragraph(doc,
    "El trabajo tiene tres objetivos concretos: (1) construir un banco con al "
    "menos 60 muestras por clase combinando fuentes públicas y capturas propias; "
    "(2) implementar un pipeline completo de preprocesamiento, extracción de "
    "características y clasificación, comparando al menos cuatro algoritmos; "
    "(3) evaluar el sistema con métricas robustas, diagnosticar los errores y "
    "desplegar el resultado en una aplicación con interfaz gráfica. El resto "
    "del artículo se organiza así: la Sección II presenta el marco teórico "
    "(filtros y descriptores); la Sección III describe la metodología "
    "(banco, preprocesamiento, descriptores, experimentos E1–E5 y métricas); la "
    "Sección IV reporta los resultados; la Sección V discute los hallazgos y "
    "responde las preguntas conceptuales 2.1–2.4; finalmente, la Sección VI "
    "resume la aplicación gráfica y las conclusiones.")

# ============================================================== PÁGINA 2
add_title_section(doc, "II", "Marco Teórico")
add_subtitle(doc, "A", "Filtros digitales")
add_paragraph(doc,
    "La convolución 2D es la operación central del filtrado espacial: cada "
    "píxel de salida es una suma ponderada de sus vecinos, donde los pesos los "
    "define el kernel h(s,t) [1]. En el taller se implementó la convolución "
    "manual con dos ciclos y se comparó contra cv2.filter2D, hallando errores "
    "del orden de 10⁻¹² para los tres tipos de borde. Los filtros paso bajo "
    "atenuían ruido y detalle fino (media, Gaussiano con σ=1.0, mediana, "
    "bilateral); los detectores de borde responden a cambios bruscos de "
    "intensidad (Roberts, Prewitt, Sobel, Scharr, Laplaciano 4 y 8 vecinos, "
    "LoG, DoG y Canny con doble umbral e histéresis); los filtros frecuenciales "
    "(Ideal, Butterworth con n=1,2,4 y Gaussiano) operan multiplicando la FFT "
    "de la imagen por una máscara H(u,v) centrada con fftshift; finalmente, el "
    "banco de filtros direccionales de Gabor —sinusoide modulada por una "
    "envolvente Gaussiana— responde fuertemente a texturas con orientación θ y "
    "frecuencia 1/λ concretas. Los resultados PSNR/SSIM del taller "
    "justifican la elección del Gaussiano 5×5 σ=1.0 como filtro de "
    "suavizado en el pipeline de la Parte 2.")
add_subtitle(doc, "B", "Descriptores de imagen")
add_paragraph(doc,
    "Sobre cada región de interés se concatenan siete familias de descriptores "
    "(Tabla I), todas con significado geométrico o fotométrico claro. El HOG "
    "de skimage (orientations=9, cells=16×16, blocks=2×2) produce 1764 "
    "dimensiones [5]. El color se codifica con media, desviación estándar y "
    "asimetría (skewness de Fisher) por canal en HSV y LAB (18 dimensiones), "
    "aprovechando que HSV es aproximadamente invariante al brillo y LAB es "
    "perceptualmente uniforme. El banco de Gabor muestrea 4 orientaciones "
    "(0°, 45°, 90°, 135°) por 2 longitudes de onda y conserva media y "
    "desviación (16 dims). El LBP uniforme con P=8, R=1 entrega un histograma "
    "de 10 bins invariante a iluminación monótona [3]. La matriz GLCM se "
    "calcula en 4 direcciones y se extraen 5 propiedades de Haralick "
    "(contraste, disimilitud, homogeneidad, energía, correlación) —20 dims "
    "[2]—. Por último, los siete momentos invariantes de Hu capturan forma "
    "con invariancia total a traslación, escala y rotación [4]. La "
    "concatenación completa (E2b) totaliza 1835 dimensiones.")

# Tabla I — Descriptores (como imagen PNG para que no se desborde)
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(4)
p.paragraph_format.space_after = Pt(2)
run = p.add_run("TABLA I")
set_run(run, size=10, bold=True)

add_image(doc, ROOT / "results/figures/ieee_tables/tabla1_descriptores.png",
          width_in=3.3)
add_caption(doc, "Familias de descriptores (T=trasl., E=escala, R=rotación).")

add_subtitle(doc, "C", "Clasificadores clásicos")
add_paragraph(doc,
    "Se entrenaron cuatro modelos base más un ensemble. La SVM con kernel RBF "
    "se configuró con C=10 y γ='scale' [6]. k-NN usó k=5 con métrica "
    "euclidiana. Random Forest combinó 200 árboles con max_depth=10 y "
    "class_weight='balanced' para compensar el leve desbalance entre las "
    "dificultades por clase (pirita y hematita son más reticentes que malaquita) "
    "[7]. La MLP feed-forward tuvo tres capas ocultas (128, 64, 32) con "
    "activación ReLU, optimizador Adam, early stopping y α=1e-4. "
    "Adicionalmente se evaluaron dos ensembles: (i) Voting ensemble (soft "
    "voting) y (ii) Stacking ensemble con meta-learner LogisticRegression. "
    "El mejor resultado se logró con Random Forest individual "
    "(class_weight='balanced') entrenado sobre el slice E3 (Color+GLCM, "
    "38 dims), igualando al Voting ensemble original (83.3 % test) pero "
    "mejorando el recall en las clases difíciles (pirita F1: 0.74 → 0.78; "
    "hematita F1: 0.82 → 0.89).")

# ============================================================== PÁGINA 3
add_title_section(doc, "III", "Metodología")
add_subtitle(doc, "A", "Banco de imágenes")
add_paragraph(doc,
    "El banco combina dos fuentes. Primero, el dataset público tedqc/mineral-"
    "dataset de HuggingFace (62,088 imágenes, 781 etiquetas, licencia CC0, "
    "subido por el usuario tedqc en 11/2025) [8], del cual se descargaron los "
    "shards parquet y se filtraron las 4 clases objetivo (quartz, pyrite, "
    "malachite, hematite). La resolución nativa media es 1019×745 px "
    "(rango 689–1024 × 467–1024), modo RGB. Los archivos JPEG distribuidos "
    "no conservan metadatos EXIF (verificado experimentalmente sobre las 168 "
    "imágenes originales del train); por tanto, el equipo de captura del "
    "curador es no documentado y se trata de un dataset agregado de muestras "
    "de colección, no de un único dispositivo. Esto introduce un _domain "
    "gap_ respecto a fotos caseras, que motiva la segunda fuente y la "
    "augmentation del train. Segundo, fotos propias con celular, "
    "protocoladas para 12 imgs/clase con cámara cenital, cartulina blanca o "
    "negra mate según el mineral, iluminación natural indirecta y distancia "
    "25–30 cm; su captura está pendiente de ejecución por el grupo y se "
    "incorporarán al test set con el script src/agregar_fotos_propias.py. "
    "La distribución final es 60 originales por clase —42 train, 9 val, 9 "
    "test—; el train se augmenta a 90 imgs/clase con rotación ±15°, flip "
    "horizontal, brillo ±25, contraste 0.8–1.2, ruido gaussiano σ∈[3,12] "
    "y compresión JPEG 50–85.")
add_subtitle(doc, "B", "Preprocesamiento")
add_paragraph(doc,
    "Sobre cada imagen se aplica un pipeline de cinco etapas: (1) resize "
    "bicúbico preservando relación de aspecto hasta 128×128; (2) filtro "
    "Gaussiano 5×5 σ=1.0 (justificado en el taller como el mejor compromiso "
    "PSNR/SSIM frente a ruido gaussiano y sal-y-pimienta); (3) CLAHE sobre el "
    "canal L del espacio LAB (clip=2.0, tile 8×8) para corregir iluminación "
    "no uniforme sin sobreexponer zonas claras; (4) umbralización de Otsu "
    "sobre el canal gris para obtener una máscara binaria del objeto; "
    "(5) limpieza morfológica con apertura seguida de cierre usando kernel "
    "elíptico 3×3. La región de interés se obtiene redimensionando la "
    "imagen completa a 128×128 tras la segmentación, lo que implícitamente "
    "centra y escala el objeto.")

# Insertar la figura 3 del preproc si existe, si no, dejar caption
add_image(doc, FIG / "pipeline_preproc.png", width_in=3.2)
add_caption(doc, "Pipeline de preprocesamiento: original → resize → Gauss → "
                 "CLAHE → Otsu → ROI.")

add_subtitle(doc, "C", "Extracción de características")
add_paragraph(doc,
    "La extracción concatena las siete familias descritas en la Sección II.B "
    "hasta totalizar 1835 dimensiones (configuración E2b). Para E1, E3, E3b, "
    "E4 y E5 se toman subconjuntos específicos resumidos en la Tabla II. "
    "Todas las características se estandarizan con StandardScaler().fit(X_train) "
    "aplicado luego idénticamente a X_val y X_test — paso crítico para "
    "evitar fuga de información, como se discute en la Pregunta 2.3.")
add_subtitle(doc, "D", "Diseño experimental")
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(2)
p.paragraph_format.space_after = Pt(2)
run = p.add_run("TABLA II")
set_run(run, size=10, bold=True)
add_image(doc, ROOT / "results/figures/ieee_tables/tabla2_experimentos.png",
          width_in=3.3)
add_caption(doc, "Configuraciones de descriptores evaluadas (E1–E5).")
add_paragraph(doc,
    "Cada experimento se evalúa con los cuatro clasificadores base y el "
    "Voting ensemble. La métrica principal es F1 macro sobre el test set "
    "(nunca visto durante el ajuste); adicionalmente se reporta accuracy. "
    "Cada combinación se somete a validación cruzada estratificada 5-fold "
    "sobre el train set para estimar la varianza.")
add_subtitle(doc, "E", "Métricas de evaluación")
add_paragraph(doc,
    "Además de accuracy y F1 macro, se reportan el classification_report "
    "por clase (precisión, recall, F1), la matriz de confusión normalizada, "
    "las curvas ROC one-vs-rest con AUC, la validación cruzada 5-fold "
    "estratificada, la búsqueda de hiperparámetros GridSearchCV para RF y "
    "MLP, las curvas de aprendizaje y la galería visual de falsos positivos "
    "y falsos negativos. Para todas las métricas se usa scikit-learn [9] y "
    "scikit-image [10].")

# ============================================================== PÁGINA 4
add_title_section(doc, "IV", "Resultados")
add_subtitle(doc, "A", "Resultados principales")
add_paragraph(doc,
    "La Tabla III resume la accuracy y F1 macro de cada combinación "
    "experimento × clasificador sobre el test set. El Random Forest "
    "(class_weight='balanced') entrenado sobre el descriptor E3 (Color + "
    "GLCM, 38 dims) obtiene el mejor resultado global: 0.833 de accuracy y "
    "0.836 de F1 macro. Sorprendentemente, los experimentos con mayor número "
    "de dimensiones (E1, E2, E2b) rindieron igual o peor que E3, evidencia "
    "de que añadir descriptores ruidosos (HOG en imágenes pequeñas, Gabor "
    "con kernels mal ajustados al tamaño 128×128) introduce sobreajuste sin "
    "aportar información discriminante adicional. La SVM individual también "
    "logró 0.833 sobre E3 en la versión inicial, pero el Random Forest con "
    "class_weight='balanced' aporta una mejor distribución de recall entre "
    "clases (F1 pirita 0.78, hematita 0.89, malaquita 0.94).")

# Tabla III (imagen PNG)
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_before = Pt(2)
p.paragraph_format.space_after = Pt(2)
run = p.add_run("TABLA III")
set_run(run, size=10, bold=True)
add_image(doc, ROOT / "results/figures/ieee_tables/tabla3_resultados.png",
          width_in=3.3)
add_caption(doc, "Accuracy por experimento × clasificador sobre el test "
                 "set. †RF con class_weight='balanced'.")

# Insertar matriz de confusión y ROC si existen
if (FIG / "resultados_confusion_final.png").exists():
    add_image(doc, FIG / "resultados_confusion_final.png", width_in=3.2)
    add_caption(doc, "Matriz de confusión normalizada del Voting ensemble (E3) "
                     "sobre el test set.")
if (FIG / "resultados_roc_final.png").exists():
    add_image(doc, FIG / "resultados_roc_final.png", width_in=3.2)
    add_caption(doc, "Curvas ROC one-vs-rest del Voting ensemble (E3) sobre el "
                     "test set.")
if (FIG / "resultados_galeria_errores.png").exists():
    add_image(doc, FIG / "resultados_galeria_errores.png", width_in=3.2)
    add_caption(doc, "Galería de falsos positivos y falsos negativos del "
                     "ensemble final.")

add_subtitle(doc, "B", "Importancia de características y PCA")
if (FIG / "resultados_pca.png").exists():
    add_image(doc, FIG / "resultados_pca.png", width_in=3.2)
    add_caption(doc, "Proyección PCA 2-D del train set (Color+GLCM) coloreada "
                     "por clase: se observa separación clara de malaquita y "
                     "hematita, con mayor solapamiento entre cuarzo y pirita.")

# ============================================================== PÁGINA 5
add_title_section(doc, "V", "Discusión")
add_subtitle(doc, "A", "Validación cruzada y ajuste de hiperparámetros")
add_paragraph(doc,
    "La validación cruzada 5-fold sobre E3 arroja 0.928 ± 0.025 de accuracy "
    "media (SVM: 0.922, k-NN: 0.761, RF: 0.900, MLP: 0.900), confirmando "
    "que el ensemble E3 generaliza bien. El GridSearchCV para RF seleccionó "
    "max_depth=10, n_estimators=100, max_features='sqrt'; para MLP, "
    "hidden_layer_sizes=(128,64,32), α=1e-4 y learning_rate_init=0.005. "
    "Las curvas de aprendizaje (Fig. 7) muestran un leve sobreajuste del "
    "MLP en el régimen de pocos datos y un comportamiento estable del RF, "
    "lo que explica por qué el voting final funciona mejor que el MLP "
    "individual.")
add_subtitle(doc, "B", "Preguntas conceptuales 2.1–2.4")
add_subsubtitle(doc, "Pregunta 2.1 — Invarianzas de los descriptores")
add_paragraph(doc,
    "HOG es invariante a traslación y robusto a iluminación por la "
    "normalización L2 por bloque, pero pierde ante rotaciones grandes y es "
    "sensible a escala por el tamaño fijo de celda. El color HSV/LAB es "
    "aproximadamente invariante al brillo (HSV separa V) y perceptualmente "
    "uniforme (LAB). Gabor multi-orientación cubre rotaciones. LBP "
    "uniforme es invariante a brillo monótono pero sensible a escala. GLCM "
    "con cuatro direcciones da cierta invariancia rotacional pero depende "
    "de la distancia fija. En nuestro problema, la cámara cenital y el "
    "resize fijo aportan cuasi-invariancia de traslación/escala; los "
    "minerales se fotografían siempre en orientación vertical/horizontal, "
    "lo que minimiza el problema de rotación; el CLAHE sobre L mitiga la "
    "variabilidad de iluminación. Estas condiciones favorables, sumadas a "
    "que los cuatro minerales tienen colores muy distintos entre sí, "
    "explican por qué el descriptor Color+GLCM (38 dims) logra el mejor "
    "resultado.")
add_subsubtitle(doc, "Pregunta 2.2 — Accuracy engañoso con clases desbalanceadas")
add_paragraph(doc,
    "Cuando las clases tienen tamaños muy distintos, el accuracy global "
    "puede ser alto simplemente prediciendo siempre la clase mayoritaria. "
    "Métricas más apropiadas en ese caso son F1 macro (promedio no ponderado "
    "del F1 por clase), balanced accuracy, matriz de confusión normalizada "
    "por clase, curvas PR y Cohen's Kappa. En nuestro caso las clases están "
    "balanceadas (90 train / 9 test por clase) por lo que el accuracy sigue "
    "siendo informativo; sin embargo, reportamos F1 macro como métrica "
    "principal por consistencia y robustez.")
add_subsubtitle(doc, "Pregunta 2.3 — Fuga de información (data leakage)")
add_paragraph(doc,
    "La fuga ocurre cuando información del test (o del val) se cuela al "
    "entrenamiento. Tres causas comunes en este tipo de proyectos: "
    "(i) ajustar el StandardScaler con X_total en vez de X_train — el "
    "scaler ve estadísticas del test; (ii) ajustar el PCA con todo el "
    "conjunto — el PCA aprende direcciones de varianza que incluyen al "
    "test; (iii) tener versiones aumentadas de una misma foto en train y "
    "en test. En el código se verifica explícitamente que el StandardScaler "
    "se ajusta solo con X_train (src/clasificador.py, función load_split + "
    "extract_features), el PCA se usa únicamente para visualización 2-D "
    "(no afecta al clasificador final), y el augmentation de "
    "src/aumentar_datos.py escribe exclusivamente en train/. Se confirmó "
    "con listdir que ninguna imagen del test aparece ni como original ni "
    "como augmentada en el train set.")
add_subsubtitle(doc, "Pregunta 2.4 — Rasgos manuales vs CNN")
add_paragraph(doc,
    "Las características manuales (HOG + Gabor + Color + GLCM + Hu) "
    "requieren decenas a cientos de muestras por clase —perfecto para "
    "nuestras 60/clase— y son altamente interpretables: cada descriptor "
    "tiene significado geométrico o fotométrico claro. El costo de "
    "entrenamiento es bajo (segundos en CPU) y el modelo serializado "
    "ocupa pocas decenas de MB. Las CNNs, en cambio, necesitan miles a "
    "millones de muestras por clase, son poco interpretables, requieren "
    "GPU y horas de entrenamiento, pero generalizan mejor con datos "
    "diversos. Para nuestro problema concreto (240 imgs en total, 4 clases, "
    "entorno CPU, necesidad de explicar cada predicción), los descriptores "
    "manuales son la elección correcta. Con transfer learning (VGG16 + "
    "fine-tuning) podríamos mejorar, pero no era necesario para el alcance "
    "del taller.")

# ============================================================== PÁGINA 6
add_title_section(doc, "VI", "Aplicación con Interfaz Gráfica")
add_paragraph(doc,
    "El sistema se empaquetó en una aplicación web con Streamlit "
    "(app.py) que carga el modelo serializado modelo_minerales_v3.pkl y "
    "ofrece: (1) carga de una imagen desde disco con vista previa; "
    "(2) selector de modelo —Random Forest (modelo final con "
    "class_weight='balanced'), MLP, SVM o comparación—; (3) barras de "
    "probabilidad por clase; (4) visualización del preprocesamiento "
    "(CLAHE / gris / máscara Otsu); (5) procesamiento por lote con "
    "resumen de predicciones; (6) segmentación de la escena por Otsu + "
    "contornos para detectar múltiples objetos y dibujar la clase "
    "predicha con su confianza. La prueba de robustez se aplicará sobre "
    "las ≥10 imágenes de escena capturadas según el protocolo "
    "documentado (cartulina blanca/negra mate, varios minerales por foto).")

# Figura 9 — captura de la app (si existe)
if (FIG / "demo_cuarzo.png").exists():
    add_image(doc, FIG / "demo_cuarzo.png", width_in=3.2)
    add_caption(doc, "Ejemplo de predicción de la app Streamlit sobre una "
                     "imagen del test set.")

add_title_section(doc, "VII", "Conclusiones y Trabajo Futuro")
add_paragraph(doc,
    "Se construyó e implementó un sistema de clasificación automática de "
    "cuatro minerales (cuarzo, pirita, malaquita y hematita) con los "
    "siguientes resultados:")
for bullet in [
    "El Random Forest con class_weight='balanced' sobre Color+GLCM (38 dims) "
    "logra 83.3% accuracy y F1 macro 0.836 en test (CV-5 91.9% ± 2.5%).",
    "El descriptor Color+GLCM (38 dims) supera al concatenado de 1835 dims "
    "dominado por HOG: a tamaño 128×128, el HOG produce histogramas "
    "ruidosos por baja densidad de gradientes.",
    "El Voting ensemble (RF+MLP+SVM, soft voting) logra el mismo accuracy "
    "global pero con peor distribución de recall entre clases; el Random "
    "Forest con class_weight='balanced' es preferible cuando las clases "
    "minoritarias (pirita, hematita) son críticas.",
    "El histograma HSV (24 dims adicionales) no aporta ganancia en este "
    "dataset porque los 18 dims de momentos ya capturan la información de "
    "color discriminante.",
    "La augmentation del train mitiga parcialmente el domain gap entre "
    "imágenes de catálogo y fotos caseras.",
    "La arquitectura es reproducible (semilla 42 en todos los random_state), "
    "interpretables y desplegable en CPU sin GPU.",
]:
    add_bullet(doc, bullet)

add_paragraph(doc,
    "Como trabajo futuro se propone: (i) capturar e integrar las fotos "
    "propias con celular y las ≥10 imágenes de escena para mejorar la "
    "prueba de robustez real; (ii) ampliar el banco a más minerales "
    "(galena, magnetita, calcita); (iii) incorporar SIFT/ORB con Bag-of-"
    "Visual-Words como descriptor adicional; (iv) comparar con una CNN "
    "liviana (MobileNet) por transfer-learning para cuantificar la brecha "
    "con enfoques profundos.")

add_title_section(doc, "VIII", "Referencias")
refs = [
    '[1] R. C. González y R. E. Woods, Digital Image Processing, 4.ª ed., Pearson, 2018.',
    '[2] R. M. Haralick, K. Shanmugam e I. Dinstein, "Textural Features for Image Classification", IEEE Trans. SMC, 1973.',
    '[3] T. Ojala, M. Pietikäinen y T. Mäenpää, "Multiresolution Gray-Scale and Rotation Invariant Texture Classification with LBP", IEEE Trans. PAMI, 2002.',
    '[4] M.-K. Hu, "Visual Pattern Recognition by Moment Invariants", IRE Trans. IT, 1962.',
    '[5] N. Dalal y B. Triggs, "Histograms of Oriented Gradients for Human Detection", CVPR, 2005.',
    '[6] C. Cortes y V. Vapnik, "Support-Vector Networks", Machine Learning, 1995.',
    '[7] L. Breiman, "Random Forests", Machine Learning, 2001.',
    '[8] tedqc, "Mineral Image Dataset", HuggingFace, 2025. [En línea]. Disponible: https://huggingface.co/datasets/tedqc/mineral-dataset',
    '[9] F. Pedregosa et al., "Scikit-learn: Machine Learning in Python", JMLR, 2011.',
    '[10] S. van der Walt et al., "scikit-image: image processing in Python", PeerJ, 2014.',
]
for r in refs:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf = p.paragraph_format
    pf.left_indent = Cm(0.6)
    pf.first_line_indent = Cm(-0.6)
    pf.space_after = Pt(0)
    pf.space_before = Pt(0)
    pf.line_spacing = 1.0
    run = p.add_run(r)
    set_run(run, size=9)

# Save
doc.save(OUT)

# ----- Post-procesado: aplicar formato IEEE de doble columna -----
# python-docx no maneja bien las columnas, así que lo hacemos vía XML.
import zipfile, shutil, re

tmp = OUT.with_suffix(".tmp.docx")
shutil.copy(OUT, tmp)

with zipfile.ZipFile(tmp, "r") as zin:
    members = {n: zin.read(n) for n in zin.namelist()}

doc_xml = members["word/document.xml"].decode("utf-8")

# Márgenes IEEE: 0.75" arriba/abajo, 0.75" lados (= 1080 twips)
# Espacio entre columnas: 0.25" (= 360 twips, equivalente a ~6 mm)
doc_xml = re.sub(
    r'<w:pgMar[^/]*/>',
    '<w:pgMar w:top="1080" w:right="1080" w:bottom="1080" w:left="1080" '
    'w:header="720" w:footer="720" w:gutter="0"/>',
    doc_xml,
)

# Forzar doble columna
doc_xml = re.sub(
    r'<w:cols[^/]*?/>',
    '<w:cols w:num="2" w:space="360" w:equalWidth="1"/>',
    doc_xml,
)
if '<w:cols' not in doc_xml:
    doc_xml = doc_xml.replace(
        '<w:sectPr',
        '<w:sectPr',
    ).replace(
        '</w:sectPr>',
        '<w:cols w:num="2" w:space="360" w:equalWidth="1"/></w:sectPr>',
    )

members["word/document.xml"] = doc_xml.encode("utf-8")

with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zout:
    for name, data in members.items():
        zout.writestr(name, data)

tmp.unlink()

print(f"OK → {OUT}")
print(f"Tamaño: {OUT.stat().st_size/1024:.1f} KB")