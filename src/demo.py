"""Demo visual: predice sobre una imagen y muestra los pasos intermedios.

Genera una figura con 4 paneles:
  (1) Imagen original
  (2) Imagen preprocesada (después de CLAHE + Otsu)
  (3) Predicción del modelo
  (4) Barras de probabilidad por clase
"""
import sys, joblib, cv2, numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, '.')
import clasificador

# Cargar modelo
MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "modelo_minerales.pkl"
bundle = joblib.load(str(MODEL_PATH))
modelo = bundle["modelo"]
scaler = bundle["scaler"]
clases = bundle["clases"]

print("=" * 60)
print(" DEMO: Predicción visual con explicación")
print("=" * 60)
print(f"\nModelo: MLP(64,32) con Color+GLCM (32 dims)")
print(f"Accuracy en test: {bundle['accuracy_test']:.3f}\n")


def predecir_y_explicar(path_imagen):
    """Predice sobre una imagen y devuelve la figura explicativa."""
    img_bgr = clasificador.imread_safe(path_imagen)
    if img_bgr is None:
        print(f"  ⚠ No se pudo leer: {path_imagen}")
        return None

    # 1. Preprocesamiento
    img_eq, gray, mask = clasificador.preprocess(img_bgr)

    # 2. Extracción de features (Color + GLCM)
    feats_all = clasificador.extract_features(img_bgr)
    feats_slice = feats_all[bundle["slice_start"]:bundle["slice_end"]]
    feats_s = scaler.transform(feats_slice.reshape(1, -1))

    # 3. Predicción
    pred_idx = modelo.predict(feats_s)[0]
    proba = modelo.predict_proba(feats_s)[0]
    pred_cls = clases[pred_idx]
    conf = proba[pred_idx]

    # 4. Generar figura 2x2
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_eq_rgb = cv2.cvtColor(img_eq, cv2.COLOR_BGR2RGB)

    # Panel 1: Original
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title(f"1. Imagen original\n({img_bgr.shape[1]}×{img_bgr.shape[0]} px)", fontsize=11)
    axes[0, 0].axis("off")

    # Panel 2: Preprocesada (ecualizada + máscara)
    axes[0, 1].imshow(img_eq_rgb)
    axes[0, 1].imshow(mask, alpha=0.3, cmap="Reds")
    axes[0, 1].set_title("2. Preprocesamiento\nCLAHE en L + Otsu (rojo = máscara)", fontsize=11)
    axes[0, 1].axis("off")

    # Panel 3: Predicción destacada
    axes[1, 0].imshow(img_rgb)
    axes[1, 0].set_title(f"3. Predicción\n{pred_cls.upper()} ({conf*100:.1f}% confianza)",
                         fontsize=14, fontweight="bold", color="darkgreen" if conf > 0.5 else "red")
    axes[1, 0].axis("off")

    # Panel 4: Barras de probabilidad
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    bars = axes[1, 1].barh(clases, proba * 100, color=colors)
    axes[1, 1].set_xlabel("Probabilidad (%)")
    axes[1, 1].set_xlim(0, 100)
    axes[1, 1].set_title("4. Distribución de probabilidad", fontsize=11)
    axes[1, 1].grid(axis="x", alpha=0.3)
    for bar, p in zip(bars, proba):
        axes[1, 1].text(p * 100 + 1, bar.get_y() + bar.get_height() / 2,
                         f"{p*100:.1f}%", va="center", fontsize=10)

    plt.suptitle(f"Demo: Clasificador de Minerales — {Path(path_imagen).name}", fontsize=13)
    plt.tight_layout()
    return plt


# Procesar 4 imágenes (una por clase)
test_files = []
for cls in clases:
    files = list(Path(f"data/dataset/test/{cls}").glob("*.jpg"))
    if files:
        test_files.append((cls, str(files[0])))

print("Imágenes de demo:")
for cls, path in test_files:
    real = cls
    plt = predecir_y_explicar(path)
    if plt is not None:
        # Predecir también para imprimir
        bundle2 = bundle
        img = clasificador.imread_safe(path)
        feats = clasificador.extract_features(img)
        feats_s = scaler.transform(feats[bundle2["slice_start"]:bundle2["slice_end"]].reshape(1, -1))
        pred = bundle2["modelo"].predict(feats_s)[0]
        proba = bundle2["modelo"].predict_proba(feats_s)[0]
        ok = "✓" if cls == clases[pred] else "✗"
        print(f"  {ok} real={real:10s}  pred={clases[pred]:10s}  conf={proba[pred]*100:.1f}%")

        # Guardar figura
        out_name = f"demo_{real}.png"
        plt.savefig(out_name, dpi=110, bbox_inches="tight")
        plt.close()
        print(f"    → {out_name}")

print("\n✓ 4 demos generadas: demo_cuarzo.png, demo_pirita.png, demo_malaquita.png, demo_hematita.png")