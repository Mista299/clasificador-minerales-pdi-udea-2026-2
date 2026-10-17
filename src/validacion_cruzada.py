"""Validación cruzada 5-fold sobre el train set (sin usar val/test).

Compara el accuracy medio y desviación estándar de los 4 modelos con
los descriptores del ganador (Color+GLCM) y los del pipeline completo.
"""
import sys, joblib, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.insert(0, '.')

# Cargar features (versión con GLCM)
feat = joblib.load("data/dataset/features_v2.joblib")
X_train, y_train = feat["train"]
X_test, y_test = feat["test"]

# Slice ganador: Color + GLCM (índices 1764..1802)
N_HOG = 1764
N_COLOR = 18       # ahora incluye skewness
N_GABOR = 16
N_LBP = 10
N_GLCM = 20
N_HU = 7

SLICES = {
    "Color+GLCM (E3)":            slice(N_HOG, N_HOG + N_COLOR + N_GLCM),
    "Color+GLCM+Hu":             slice(N_HOG, N_HOG + N_COLOR + N_GLCM + N_HU),
    "HOG+Color+Gabor+LBP+GLCM (E2)": slice(0, N_HOG + N_COLOR + N_GABOR + N_LBP + N_GLCM),
    "HOG+Color+Gabor+LBP+GLCM+Hu": slice(0, N_HOG + N_COLOR + N_GABOR + N_LBP + N_GLCM + N_HU),
}

MODELS = {
    "SVM (RBF)":     SVC(kernel="rbf", C=10, gamma="scale", random_state=42),
    "k-NN (k=5)":    KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "MLP (64,32)":   MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
}

print("=" * 60)
print(" VALIDACIÓN CRUZADA 5-FOLD (train set)")
print("=" * 60)
print(f"train set: {X_train.shape[0]} muestras, {len(np.unique(y_train))} clases")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print(f"\n{'Experimento':35s} {'Modelo':15s} {'Acc (μ±σ)':15s} {'F1 (μ±σ)':15s}")
print("-" * 80)

import pandas as pd
rows = []
for exp_name, sl in SLICES.items():
    Xtr = X_train[:, sl]
    sc = StandardScaler().fit(Xtr)
    Xtr_s = sc.transform(Xtr)
    for model_name, model in MODELS.items():
        acc_scores = cross_val_score(model, Xtr_s, y_train, cv=skf, scoring="accuracy", n_jobs=-1)
        f1_scores = cross_val_score(model, Xtr_s, y_train, cv=skf, scoring="f1_macro", n_jobs=-1)
        acc_mean, acc_std = acc_scores.mean(), acc_scores.std()
        f1_mean, f1_std = f1_scores.mean(), f1_scores.std()
        rows.append({
            "Experimento": exp_name, "Modelo": model_name,
            "Acc μ": acc_mean, "Acc σ": acc_std,
            "F1 μ": f1_mean, "F1 σ": f1_std,
        })
        print(f"{exp_name:35s} {model_name:15s} {acc_mean:.3f}±{acc_std:.3f}  {f1_mean:.3f}±{f1_std:.3f}")

df = pd.DataFrame(rows)
df.to_csv("resultados_cv5.csv", index=False)
print(f"\n✓ resultados_cv5.csv")
print(df.to_string(index=False, float_format="%.3f"))