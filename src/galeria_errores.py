"""Genera una galería visual de falsos positivos y falsos negativos.

Muestra las imágenes mal clasificadas del test set con:
  - Clase real
  - Clase predicha
  - Probabilidad asignada por el modelo
"""
import sys, joblib, cv2, numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

sys.path.insert(0, '.')
import clasificador

CLASSES = clasificador.CLASSES

# Cargar features y entrenar mejor modelo (MLP con Color+GLCM)
feat = joblib.load("data/dataset/features_v2.joblib")
X_train, y_train = feat["train"]
X_test, y_test = feat["test"]

N_HOG, N_COLOR, N_GABOR, N_LBP, N_GLCM, N_HU = 1764, 18, 16, 10, 20, 7
SLICE_E3 = slice(N_HOG, N_HOG + N_COLOR + N_GLCM)

X_tr = X_train[:, SLICE_E3]
X_te = X_test[:, SLICE_E3]
sc = StandardScaler().fit(X_tr)
X_tr_s = sc.transform(X_tr); X_te_s = sc.transform(X_te)

model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
model.fit(X_tr_s, y_train)
y_pred = model.predict(X_te_s)
y_proba = model.predict_proba(X_te_s)

acc = accuracy_score(y_test, y_pred)
print(f"Accuracy en test: {acc:.3f}")

# Cargar imágenes originales para mostrar
test_files = []
for cls_idx, cls in enumerate(CLASSES):
    d = Path("data/dataset/test") / cls
    for f in sorted(d.glob("*.jpg")):
        test_files.append((f, cls_idx))

print(f"Total test: {len(test_files)}")

# Identificar errores
errors = []
for i, (fpath, true_idx) in enumerate(test_files):
    pred_idx = y_pred[i]
    if true_idx != pred_idx:
        errors.append({
            "path": fpath,
            "true": CLASSES[true_idx],
            "pred": CLASSES[pred_idx],
            "conf": y_proba[i, pred_idx],
            "conf_true": y_proba[i, true_idx],
        })

print(f"Errores: {len(errors)}")

if not errors:
    print("¡No hay errores! El modelo clasifica perfectamente el test set.")
else:
    # Crear galería
    n = len(errors)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).ravel()

    for i, e in enumerate(errors):
        img = cv2.imread(str(e["path"]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[i].imshow(img)
        axes[i].set_title(
            f'Real: {e["true"]}\nPred: {e["pred"]} ({e["conf"]*100:.0f}%)\n'
            f'P(real)={e["conf_true"]*100:.1f}%',
            fontsize=10, color="red"
        )
        axes[i].axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"Galería de errores (FP + FN) — MLP + Color+GLCM — acc={acc:.3f}", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("resultados_galeria_errores.png", dpi=110, bbox_inches="tight")
    plt.close()
    print("✓ resultados_galeria_errores.png")

    # Análisis por par
    from collections import Counter
    pares = Counter([(e["true"], e["pred"]) for e in errors])
    print("\nPares de confusión (real → predicho):")
    for (r, p), n in pares.most_common():
        print(f"  {r:12s} → {p:12s}: {n}")