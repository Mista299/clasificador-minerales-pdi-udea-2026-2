"""Pipeline completo de clasificación de minerales (versión optimizada).

Mejoras vs clasificador.py:
  - Cachea el preprocesamiento (HOG usa el gray ya calculado).
  - Color usa uint8 directamente (más rápido).
  - Gabor reducido (4 orientaciones × 2 λ = 8 filtros).
"""
from pathlib import Path
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
)
from sklearn.decomposition import PCA
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data" / "dataset"
CLASSES = ["cuarzo", "pirita", "malaquita", "hematita"]


def imread_safe(path):
    """cv2.imread tolerante a rutas con tildes/Unicode (Windows)."""
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


def imwrite_safe(path, img, params=None):
    """cv2.imwrite tolerante a rutas con tildes/Unicode (Windows)."""
    ok, buf = cv2.imencode(Path(str(path)).suffix or ".jpg", img, params or [])
    if ok:
        buf.tofile(str(path))
    return ok


def preprocess(img_bgr, target_size=(128, 128)):
    """Resize + blur + CLAHE + Otsu → (img_eq, gray, mask)."""
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


def extract_glcm(gray):
    """GLCM / Haralick en 4 direcciones → 5 props × 4 dirs = 20 dims.

    Propiedades: contraste, disimilitud, homogeneidad, energía, correlación.
    """
    gray_q = (gray // 8).astype(np.uint8)   # 32 niveles
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
    """7 momentos de Hu: invariantes a traslación, escala y rotación.

    Calculados sobre los contornos externos de la máscara binaria.
    Devuelve el logaritmo del valor absoluto (Hu1..Hu7 en escala manejable).
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(7, dtype=np.float32)
    # Tomar el contorno más grande
    cnt = max(contours, key=cv2.contourArea)
    moments = cv2.moments(cnt)
    hu = cv2.HuMoments(moments).flatten()  # 7 valores
    # Transformar con -sign(x)*log(abs(x)) para comprimir el rango
    hu_log = np.zeros_like(hu)
    for i, v in enumerate(hu):
        if v != 0:
            hu_log[i] = -np.sign(v) * np.log10(abs(v) + 1e-30)
        else:
            hu_log[i] = 0.0
    return hu_log.astype(np.float32)


def extract_features(img_bgr):
    """Pipeline completo en una imagen."""
    img_eq, gray, mask = preprocess(img_bgr)

    # HOG (8100 dims → con 16x16 cells baja a ~1296, más manejable)
    f_hog = hog(gray, orientations=9, pixels_per_cell=(16, 16),
                cells_per_block=(2, 2), feature_vector=True, channel_axis=None)

    # Color: media, std y asimetría (skewness) por canal en HSV (3) y LAB (3) → 18 dims
    hsv = cv2.cvtColor(img_eq, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(img_eq, cv2.COLOR_BGR2LAB).astype(np.float32)

    def skew(x):
        """Asimetría de Fisher: E[(x-μ)³] / σ³. Robusta a outliers."""
        s = x.std()
        return ((x - x.mean()) ** 3).mean() / (s ** 3 + 1e-9)

    color_feats = []
    for img in (hsv, lab):
        for c in range(3):
            ch = img[..., c]
            color_feats.extend([ch.mean(), ch.std(), skew(ch)])
    f_color = np.array(color_feats, dtype=np.float32)

    # Gabor: 4 orientaciones × 2 λ = 8 filtros → media+std = 16 dims
    f_gabor = []
    for theta in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
        for lam in [8, 16]:
            k = cv2.getGaborKernel((31, 31), 3.0, theta, lam, 0.5, 0, ktype=cv2.CV_32F)
            resp = cv2.filter2D(gray, cv2.CV_32F, k)
            f_gabor.extend([resp.mean(), resp.std()])
    f_gabor = np.array(f_gabor, dtype=np.float32)

    # LBP: histograma uniforme → 10 dims
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    f_lbp, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)
    f_lbp = f_lbp.astype(np.float32)

    # GLCM / Haralick: matriz de co-ocurrencia en 4 direcciones → 5 props × 4 dirs = 20 dims
    f_glcm = extract_glcm(gray)

    # Momentos de Hu: 7 invariantes a traslación, escala y rotación → 7 dims
    # Aplicados sobre el contorno externo de la región segmentada
    f_hu = extract_hu_moments(mask)

    return np.concatenate([f_hog, f_color, f_gabor, f_lbp, f_glcm, f_hu]).astype(np.float32)


def load_split(split):
    X_img, y = [], []
    for idx, cls in enumerate(CLASSES):
        d = DATA / split / cls
        for f in sorted(d.glob("*.jpg")):
            X_img.append(imread_safe(f))
            y.append(idx)
    return X_img, np.array(y)


def main():
    print("=" * 60)
    print(" CLASIFICADOR DE MINERALES")
    print(" Cuarzo | Pirita | Malaquita | Hematita")
    print("=" * 60)

    print("\n[1/5] Cargando imágenes...")
    splits = {}
    for split in ["train", "val", "test"]:
        X_img, y = load_split(split)
        splits[split] = (X_img, y)
        print(f"  {split}: {len(X_img)} imágenes")

    print("\n[2/5] Extrayendo características...")
    feat = {}
    t_start = time.time()
    for split, (X_img, y) in splits.items():
        t0 = time.time()
        X_feat = np.array([extract_features(img) for img in X_img])
        feat[split] = (X_feat, y)
        print(f"  {split}: X={X_feat.shape}, y={y.shape}, tiempo={time.time()-t0:.1f}s")
    print(f"  Total extracción: {time.time()-t_start:.1f}s")

    X_train, y_train = feat["train"]
    X_val, y_val = feat["val"]
    X_test, y_test = feat["test"]

    print("\n[3/5] Normalizando (StandardScaler con train)...")
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    print("\n[4/5] PCA 2D para visualización...")
    pca = PCA(n_components=2).fit(X_train_s)
    X_train_pca = pca.transform(X_train_s)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for i, (cls, c) in enumerate(zip(CLASSES, colors)):
        m = y_train == i
        ax.scatter(X_train_pca[m, 0], X_train_pca[m, 1], label=cls, alpha=0.75, s=50, c=c, edgecolor="k", linewidth=0.5)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(f"PCA 2D — dim original={X_train.shape[1]}")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("resultados_pca.png", dpi=110); plt.show()

    print("\n[5/5] Entrenando clasificadores...")
    models = {
        "SVM (RBF)":      SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42),
        "k-NN (k=5)":     KNeighborsClassifier(n_neighbors=5, metric="euclidean"),
        "Random Forest":  RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "MLP (64,32)":    MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
    }
    results = []
    fig_cm, axes_cm = plt.subplots(1, len(models), figsize=(5 * len(models), 4.2))

    for idx, (name, model) in enumerate(models.items()):
        t0 = time.time()
        model.fit(X_train_s, y_train)
        t_train = time.time() - t0
        y_pred = model.predict(X_test_s)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        results.append({"Modelo": name, "Accuracy": acc, "F1 macro": f1, "Train (s)": t_train})
        print(f"\n  {name}: acc={acc:.3f}  F1={f1:.3f}  train={t_train:.2f}s")
        print(classification_report(y_test, y_pred, target_names=CLASSES, digits=3))

        cm = confusion_matrix(y_test, y_pred, normalize="true")
        ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(
            ax=axes_cm[idx], cmap="Blues", values_format=".2f", xticks_rotation=45, colorbar=False
        )
        axes_cm[idx].set_title(name)

        # ROC solo para SVM
        if name.startswith("SVM"):
            fig_roc, axes_lr = plt.subplots(1, 2, figsize=(12, 5))
            y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
            y_score = model.predict_proba(X_test_s)
            for c_idx, (cls, color) in enumerate(zip(CLASSES, colors)):
                fpr, tpr, _ = roc_curve(y_test_bin[:, c_idx], y_score[:, c_idx])
                roc_auc = auc(fpr, tpr)
                axes_lr[0].plot(fpr, tpr, label=f"{cls} (AUC={roc_auc:.2f})", color=color, lw=2)
            axes_lr[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
            axes_lr[0].set_xlabel("FPR"); axes_lr[0].set_ylabel("TPR")
            axes_lr[0].set_title("Curvas ROC — SVM RBF (test, one-vs-rest)")
            axes_lr[0].legend(loc="lower right"); axes_lr[0].grid(alpha=0.3)
            fpr_m, tpr_m, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
            roc_auc_m = auc(fpr_m, tpr_m)
            axes_lr[1].plot(fpr_m, tpr_m, label=f"micro-avg (AUC={roc_auc_m:.2f})", color="navy", lw=2.5)
            axes_lr[1].plot([0, 1], [0, 1], "k--", alpha=0.4)
            axes_lr[1].set_xlabel("FPR"); axes_lr[1].set_ylabel("TPR")
            axes_lr[1].set_title("Curva ROC micro-promedio (todas las clases)")
            axes_lr[1].legend(); axes_lr[1].grid(alpha=0.3)
            plt.tight_layout(); plt.savefig("resultados_roc.png", dpi=110); plt.show()

    fig_cm.suptitle("Matrices de confusión normalizadas (test)")
    plt.tight_layout(); plt.savefig("resultados_confusion.png", dpi=110); plt.show()

    print("\n" + "=" * 60)
    print(" RESUMEN FINAL — Test set (36 imágenes)")
    print("=" * 60)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    df.to_csv("resultados_modelos.csv", index=False)
    print("\n📊 Figuras: resultados_pca.png, resultados_confusion.png, resultados_roc.png")
    print("📄 Tabla:   resultados_modelos.csv")


if __name__ == "__main__":
    main()