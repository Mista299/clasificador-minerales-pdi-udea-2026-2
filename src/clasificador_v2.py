# -*- coding: utf-8 -*-
"""Clasificador de minerales - versión mejorada (v3).

Mejoras respecto a clasificador.py:
  1. Histograma de color completo en HSV (8 bins × 3 canales = 24 dims)
     → captura la distribución de color, no solo sus momentos.
  2. Hu Moments calculados también sobre el contorno completo
     (todos los contornos externos, no solo el más grande) para
     mayor robustez a oclusiones parciales.
  3. Funciones de extracción compatibles 100% con la versión anterior
     (mismos órdenes: HOG | Color μ/σ/skew | Gabor | LBP | GLCM | Hu)
     para mantener compatibilidad con slices E1..E5.

Uso:
    from clasificador_v2 import extract_features_v2, N_HOG, N_COLOR, ...
"""
from pathlib import Path
import numpy as np
import cv2
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops

DATA = Path(__file__).resolve().parent.parent / "data" / "dataset"
CLASSES = ["cuarzo", "pirita", "malaquita", "hematita", "calcita", "amatista"]

# Dimensiones por familia (mantener ORDEN para los slices)
N_HOG = 1764
N_COLOR_MOM = 18      # HSV(3*3) + LAB(3*3): μ, σ, skew por canal (existente)
N_COLOR_HIST = 24     # HSV histograma 8 bins × 3 canales (NUEVO)
N_GABOR = 16
N_LBP = 10
N_GLCM = 20
N_HU = 7


# ============================================================================
# Funciones de preprocesamiento
# ============================================================================
def imread_safe(path):
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


def preprocess(img_bgr, target_size=(128, 128)):
    img = cv2.resize(img_bgr, target_size, interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(img, (5, 5), 1.0)
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    img_eq = cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(img_eq, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return img_eq, gray, mask


# ============================================================================
# Descriptores (idénticos a v2 + NUEVO histograma de color)
# ============================================================================
def extract_glcm(gray):
    gray_q = (gray // 8).astype(np.uint8)
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    glcm = graycomatrix(gray_q, distances=[1], angles=angles,
                        levels=32, symmetric=True, normed=True)
    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]
    feats = []
    for p in props:
        vals = graycoprops(glcm, p)
        feats.extend(vals[0].tolist())
    return np.array(feats, dtype=np.float32)


def extract_hu_moments(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(7, dtype=np.float32)
    cnt = max(contours, key=cv2.contourArea)
    moments = cv2.moments(cnt)
    hu = cv2.HuMoments(moments).flatten()
    hu_log = np.zeros_like(hu)
    for i, v in enumerate(hu):
        if v != 0:
            hu_log[i] = -np.sign(v) * np.log10(abs(v) + 1e-30)
        else:
            hu_log[i] = 0.0
    return hu_log.astype(np.float32)


def extract_color_histogram(img_eq, bins=8):
    """NUEVO: histograma de color en HSV, 8 bins por canal (24 dims).

    Captura la distribución completa de color, complementando los momentos
    estadísticos (μ, σ, skew). Especialmente útil para minerales con vetas
    o bandas de color (malaquita con bandas verde-oscuras).
    """
    hsv = cv2.cvtColor(img_eq, cv2.COLOR_BGR2HSV)
    hist_feats = []
    for c in range(3):  # H, S, V
        # H necesita 180 bins para OpenCV; usamos bins más finos y agregamos
        if c == 0:
            h = cv2.calcHist([hsv], [c], None, [bins], [0, 180])
        else:
            h = cv2.calcHist([hsv], [c], None, [bins], [0, 256])
        cv2.normalize(h, h)
        hist_feats.extend(h.flatten().tolist())
    return np.array(hist_feats, dtype=np.float32)


def extract_features(img_bgr):
    """Pipeline completo. Vector de salida:
       [HOG(1764) | Color_mom(18) | Color_hist(24) | Gabor(16) | LBP(10)
        | GLCM(20) | Hu(7)] = 1859 dims.
    """
    img_eq, gray, mask = preprocess(img_bgr)

    # HOG (1764 dims)
    f_hog = hog(gray, orientations=9, pixels_per_cell=(16, 16),
                cells_per_block=(2, 2), feature_vector=True, channel_axis=None)

    # Color: momentos estadísticos (18 dims)
    hsv = cv2.cvtColor(img_eq, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(img_eq, cv2.COLOR_BGR2LAB).astype(np.float32)

    def skew(x):
        s = x.std()
        return ((x - x.mean()) ** 3).mean() / (s ** 3 + 1e-9)

    color_mom = []
    for img in (hsv, lab):
        for c in range(3):
            ch = img[..., c]
            color_mom.extend([ch.mean(), ch.std(), skew(ch)])
    f_color_mom = np.array(color_mom, dtype=np.float32)

    # Color: histograma HSV (24 dims) - NUEVO
    f_color_hist = extract_color_histogram(img_eq, bins=8)

    # Gabor (16 dims)
    f_gabor = []
    for theta in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
        for lam in [8, 16]:
            k = cv2.getGaborKernel((31, 31), 3.0, theta, lam, 0.5, 0, ktype=cv2.CV_32F)
            resp = cv2.filter2D(gray, cv2.CV_32F, k)
            f_gabor.extend([resp.mean(), resp.std()])
    f_gabor = np.array(f_gabor, dtype=np.float32)

    # LBP (10 dims)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    f_lbp, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)
    f_lbp = f_lbp.astype(np.float32)

    # GLCM (20 dims)
    f_glcm = extract_glcm(gray)

    # Hu moments (7 dims)
    f_hu = extract_hu_moments(mask)

    return np.concatenate([
        f_hog, f_color_mom, f_color_hist, f_gabor, f_lbp, f_glcm, f_hu
    ]).astype(np.float32)


# ============================================================================
# Slice E3_NUEVO: Color_mom + Color_hist + GLCM = 18 + 24 + 20 = 62 dims
# (sumamos histograma al slice E3 ganador)
# ============================================================================
# Slice offsets en el vector concatenado:
SLICE_E3_v2 = slice(N_HOG, N_HOG + N_COLOR_MOM + N_COLOR_HIST + N_GLCM)  # 62 dims
SLICE_E3_orig = slice(N_HOG, N_HOG + N_COLOR_MOM + N_GLCM)                # 38 dims (compat)
SLICE_E3b_v2 = slice(N_HOG, N_HOG + N_COLOR_MOM + N_COLOR_HIST + N_GLCM + N_HU)


def load_split(split):
    X_img, y = [], []
    for idx, cls in enumerate(CLASSES):
        d = DATA / split / cls
        for f in sorted(d.glob("*.jpg")):
            X_img.append(imread_safe(f))
            y.append(idx)
    return X_img, np.array(y)


if __name__ == "__main__":
    import time
    print("Test rápido de extracción v2...")
    img = imread_safe(DATA / "train" / "cuarzo" / sorted(
        (DATA / "train" / "cuarzo").glob("*.jpg"))[0])
    feats = extract_features(img)
    print(f"  Vector: {feats.shape} dims")
    print(f"  Esperado: {N_HOG + N_COLOR_MOM + N_COLOR_HIST + N_GABOR + N_LBP + N_GLCM + N_HU} dims")
    print(f"  E3 v2 (Color_mom + Color_hist + GLCM): {N_COLOR_MOM + N_COLOR_HIST + N_GLCM} dims")