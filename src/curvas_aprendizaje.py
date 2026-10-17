"""Curvas de aprendizaje: accuracy vs tamaño del train set.

Útil para diagnosticar sobreajuste (overfitting) o subajuste (underfitting):
  - Si train >> val y se mantiene: sobreajuste (más datos ayudarían)
  - Si ambos bajos: subajuste (más features/modelo más complejo)
"""
import sys, joblib, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import learning_curve

sys.path.insert(0, '.')

X_train, y_train = (joblib.load("data/dataset/features_v2.joblib"))["train"]
X_test, y_test = (joblib.load("data/dataset/features_v2.joblib"))["test"]

# Slice ganador: Color + GLCM (32 dims)
N_HOG, N_COLOR, N_GABOR, N_LBP, N_GLCM, N_HU = 1764, 18, 16, 10, 20, 7
SLICE_E3 = slice(N_HOG, N_HOG + N_COLOR + N_GLCM)

Xtr = X_train[:, SLICE_E3]; Xte = X_test[:, SLICE_E3]
sc = StandardScaler().fit(Xtr)
Xtr_s = sc.transform(Xtr); Xte_s = sc.transform(Xte)

models = {
    "MLP (64,32) [ganador CV]": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
    "SVM (RBF)":               SVC(kernel="rbf", C=10, gamma="scale", random_state=42),
    "Random Forest":            RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "k-NN (k=5)":               KNeighborsClassifier(n_neighbors=5),
}

# Curva de aprendizaje (train_size de 10% a 100%)
train_sizes = np.linspace(0.1, 1.0, 8)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for name, model in models.items():
    print(f"Calculando learning curve para {name}...")
    train_sizes_abs, train_scores, val_scores = learning_curve(
        model, Xtr_s, y_train, cv=5,
        train_sizes=train_sizes, scoring="accuracy", n_jobs=-1, random_state=42
    )
    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    axes[0].plot(train_sizes_abs, train_mean, "o-", label=f"{name} (train)")
    axes[0].fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.15)
    axes[1].plot(train_sizes_abs, val_mean, "o-", label=name)
    axes[1].fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.15)

axes[0].set_xlabel("Tamaño del train set"); axes[0].set_ylabel("Accuracy")
axes[0].set_title("Curva de aprendizaje — TRAIN score")
axes[0].legend(loc="lower right"); axes[0].grid(alpha=0.3); axes[0].set_ylim(0.5, 1.05)

axes[1].set_xlabel("Tamaño del train set"); axes[1].set_ylabel("Accuracy")
axes[1].set_title("Curva de aprendizaje — VALIDATION score (5-fold CV)")
axes[1].legend(loc="lower right"); axes[1].grid(alpha=0.3); axes[1].set_ylim(0.5, 1.05)

plt.tight_layout()
plt.savefig("resultados_curvas_aprendizaje.png", dpi=110)
print("✓ resultados_curvas_aprendizaje.png")