"""GridSearchCV para encontrar los mejores hiperparámetros del modelo ganador.

Modelos optimizados:
  - Random Forest (n_estimators, max_depth, min_samples_split)
  - MLP (hidden_layer_sizes, alpha, learning_rate)

Evalúa sobre el slice E3 (Color+GLCM, 38 dims) que es el ganador.
"""
import sys, joblib, numpy as np, pandas as pd, time
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, '.')

# Cargar features (versión con Hu + skewness)
print("Cargando features...")
feat = joblib.load("data/dataset/features_v2.joblib")
X_train, y_train = feat["train"]
X_test, y_test = feat["test"]
print(f"  train: {X_train.shape}, test: {X_test.shape}")

# Slice E3: Color + GLCM (con skew, sin HOG ni Hu)
N_HOG = 1764
N_COLOR = 18
N_GLCM = 20
SLICE = slice(N_HOG, N_HOG + N_COLOR + N_GLCM)
print(f"  slice E3 (Color+GLCM con skew): {SLICE}")

Xtr = X_train[:, SLICE]
Xte = X_test[:, SLICE]
sc = StandardScaler().fit(Xtr)
Xtr_s = sc.transform(Xtr)
Xte_s = sc.transform(Xte)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ===========================================================================
# 1. Random Forest
# ===========================================================================
print("\n" + "=" * 60)
print(" GridSearchCV — Random Forest")
print("=" * 60)
rf_params = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "max_features": ["sqrt", "log2"],
}
rf = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    rf_params, cv=skf, scoring="f1_macro", n_jobs=-1, verbose=1
)
t0 = time.time()
rf.fit(Xtr_s, y_train)
print(f"\nTiempo: {time.time()-t0:.1f}s")
print(f"Mejor F1 (CV): {rf.best_score_:.4f}")
print(f"Mejores params: {rf.best_params_}")

yp = rf.predict(Xte_s)
acc_test = accuracy_score(y_test, yp)
f1_test = f1_score(y_test, yp, average="macro")
print(f"Test accuracy: {acc_test:.4f}  F1: {f1_test:.4f}")

# ===========================================================================
# 2. MLP
# ===========================================================================
print("\n" + "=" * 60)
print(" GridSearchCV — MLP")
print("=" * 60)
mlp_params = {
    "hidden_layer_sizes": [(32,), (64, 32), (128, 64, 32), (64, 32, 16)],
    "alpha": [0.0001, 0.001, 0.01],
    "learning_rate_init": [0.001, 0.005],
}
mlp = GridSearchCV(
    MLPClassifier(max_iter=1000, random_state=42, early_stopping=True),
    mlp_params, cv=skf, scoring="f1_macro", n_jobs=-1, verbose=1
)
t0 = time.time()
mlp.fit(Xtr_s, y_train)
print(f"\nTiempo: {time.time()-t0:.1f}s")
print(f"Mejor F1 (CV): {mlp.best_score_:.4f}")
print(f"Mejores params: {mlp.best_params_}")

yp = mlp.predict(Xte_s)
acc_test_mlp = accuracy_score(y_test, yp)
f1_test_mlp = f1_score(y_test, yp, average="macro")
print(f"Test accuracy: {acc_test_mlp:.4f}  F1: {f1_test_mlp:.4f}")

# ===========================================================================
# Resumen
# ===========================================================================
print("\n" + "=" * 60)
print(" RESUMEN — Mejores hiperparámetros")
print("=" * 60)

results = pd.DataFrame([
    {
        "Modelo": "Random Forest",
        "Best CV F1": rf.best_score_,
        "Test Accuracy": acc_test,
        "Test F1": f1_test,
        "Mejores params": str(rf.best_params_),
    },
    {
        "Modelo": "MLP",
        "Best CV F1": mlp.best_score_,
        "Test Accuracy": acc_test_mlp,
        "Test F1": f1_test_mlp,
        "Mejores params": str(mlp.best_params_),
    },
])
print(results.to_string(index=False))
results.to_csv("resultados_gridsearch.csv", index=False)
print("\n✓ resultados_gridsearch.csv")

# Guardar el mejor modelo (MLP o RF)
if f1_test_mlp >= f1_test:
    best_model = mlp.choose if hasattr(mlp, 'choose') else mlp.best_estimator_
    best_name = "MLP"
    best_acc = acc_test_mlp
    best_f1 = f1_test_mlp
else:
    best_model = rf.best_estimator_
    best_name = "RF"
    best_acc = acc_test
    best_f1 = f1_test

print(f"\nMejor modelo global: {best_name}  acc={best_acc:.3f}  F1={best_f1:.3f}")
print(f"Hiperparámetros: {best_model.get_params()}")