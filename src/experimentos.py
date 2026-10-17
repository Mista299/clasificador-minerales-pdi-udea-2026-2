"""Experimentos E1..E4 comparando familias de descriptores.

E1: solo HOG       → 1764 dims
E2: HOG + color + Gabor + LBP + GLCM (todo) → 1822 dims
E3: solo color + GLCM (sin forma) → 12 + 20 = 32 dims
E4: solo HOG + Gabor (forma + textura) → 1764 + 16 = 1780 dims

Clasificador: Random Forest (n_estimators=100), el ganador del experimento previo.
"""
import sys, joblib, numpy as np
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, '.')
import clasificador

print("=" * 60)
print(" EXPERIMENTOS E1..E4 — Familias de descriptores")
print("=" * 60)

# Recalcular features con GLCM
print("\nRecalculando features (con GLCM, 1822 dims)...")
import time
t0 = time.time()
feat = {}
for split in ["train", "val", "test"]:
    X_img, y = clasificador.load_split(split)
    X_feat = np.array([clasificador.extract_features(img) for img in X_img])
    feat[split] = (X_feat, y)
    print(f"  {split}: X={X_feat.shape} t={time.time()-t0:.1f}s")
joblib.dump(feat, "data/dataset/features_v2.joblib")

X_train, y_train = feat["train"]
X_test, y_test = feat["test"]

# Rangos de cada familia (calculados en clasificador.py)
N_HOG = 1764      # orientations=9, cells=16x16, blocks=2x2 → 8*8*4*9 = 2304 con cells 16,16
N_COLOR = 18       # HSV(3*3) + LAB(3*3): μ, σ, skew por canal
N_GABOR = 16       # 4θ × 2λ × 2 (mean+std)
N_LBP = 10         # hist uniforme
N_GLCM = 20        # 5 props × 4 dirs
N_HU = 7           # momentos de Hu (forma invariante)

# Definir slices
SLICES = {
    "E1: solo HOG":              slice(0, N_HOG),
    "E2: HOG+Color+Gabor+LBP+GLCM": slice(0, N_HOG + N_COLOR + N_GABOR + N_LBP + N_GLCM),
    "E2b: E2 + Hu":               slice(0, N_HOG + N_COLOR + N_GABOR + N_LBP + N_GLCM + N_HU),
    "E3: Color+GLCM":            slice(N_HOG, N_HOG + N_COLOR + N_GLCM),
    "E3b: Color+GLCM+Hu":         slice(N_HOG, N_HOG + N_COLOR + N_GLCM + N_HU),
    "E5: solo Hu":               slice(N_HOG + N_COLOR + N_GABOR + N_LBP + N_GLCM,
                                   N_HOG + N_COLOR + N_GABOR + N_LBP + N_GLCM + N_HU),
    "E4: HOG+Gabor":             slice(0, N_HOG + N_GABOR),
}

results = []
print("\nComparación de experimentos:")
print(f"{'Experimento':35s} {'Dims':>6s} {'Acc':>6s} {'F1':>6s}")
print("-" * 60)
for name, sl in SLICES.items():
    Xtr = X_train[:, sl]
    Xte = X_test[:, sl]
    sc = StandardScaler().fit(Xtr)
    Xtr_s = sc.transform(Xtr); Xte_s = sc.transform(Xte)
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(Xtr_s, y_train)
    yp = model.predict(Xte_s)
    acc = accuracy_score(y_test, yp)
    f1 = f1_score(y_test, yp, average="macro")
    results.append({"Experimento": name, "Dims": Xtr.shape[1], "Accuracy": acc, "F1 macro": f1})
    print(f"{name:35s} {Xtr.shape[1]:>6d} {acc:>6.3f} {f1:>6.3f}")

df = pd.DataFrame(results)
df.to_csv("resultados_experimentos.csv", index=False)
print("\n✓ resultados_experimentos.csv")
print(df.to_string(index=False))