"""Voting ensemble de los 3 mejores modelos.

Compara soft voting (promedia probabilidades) vs hard voting (mayoría de votos).
Evalúa sobre el slice E3 (Color+GLCM, 38 dims).
"""
import sys, joblib, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report

sys.path.insert(0, '.')

# Cargar features
feat = joblib.load("data/dataset/features_v2.joblib")
X_train, y_train = feat["train"]
X_test, y_test = feat["test"]

N_HOG, N_COLOR, N_GLCM = 1764, 18, 20
SLICE = slice(N_HOG, N_HOG + N_COLOR + N_GLCM)
Xtr = X_train[:, SLICE]; Xte = X_test[:, SLICE]
sc = StandardScaler().fit(Xtr)
Xtr_s = sc.transform(Xtr); Xte_s = sc.transform(Xte)

CLS = ["cuarzo", "pirita", "malaquita", "hematita"]

# Modelos base (los mejores del gridsearch)
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
mlp = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=1000, alpha=0.0001,
                     learning_rate_init=0.005, random_state=42, early_stopping=True)
svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)

# Ensembles
ensembles = {
    "Soft Voting (RF+MLP+SVM)": VotingClassifier(
        estimators=[("rf", rf), ("mlp", mlp), ("svm", svm)],
        voting="soft", n_jobs=-1
    ),
    "Hard Voting (RF+MLP+SVM)": VotingClassifier(
        estimators=[("rf", rf), ("mlp", mlp), ("svm", svm)],
        voting="hard", n_jobs=-1
    ),
    "Soft Voting (RF+MLP)": VotingClassifier(
        estimators=[("rf", rf), ("mlp", mlp)],
        voting="soft", n_jobs=-1
    ),
    "Soft Voting (MLP+SVM)": VotingClassifier(
        estimators=[("mlp", mlp), ("svm", svm)],
        voting="soft", n_jobs=-1
    ),
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("=" * 60)
print(" VOTING ENSEMBLES — E3 (Color+GLCM, 38 dims)")
print("=" * 60)

rows = []
for name, ens in ensembles.items():
    # Validación cruzada
    cv_acc = cross_val_score(ens, Xtr_s, y_train, cv=skf, scoring="accuracy", n_jobs=-1)
    cv_f1 = cross_val_score(ens, Xtr_s, y_train, cv=skf, scoring="f1_macro", n_jobs=-1)
    # Test
    ens.fit(Xtr_s, y_train)
    yp = ens.predict(Xte_s)
    test_acc = accuracy_score(y_test, yp)
    test_f1 = f1_score(y_test, yp, average="macro")
    rows.append({
        "Ensemble": name,
        "CV Acc μ±σ": f"{cv_acc.mean():.3f}±{cv_acc.std():.3f}",
        "CV F1 μ±σ":  f"{cv_f1.mean():.3f}±{cv_f1.std():.3f}",
        "Test Acc":   f"{test_acc:.3f}",
        "Test F1":     f"{test_f1:.3f}",
    })
    print(f"\n{name}")
    print(f"  CV5: acc={cv_acc.mean():.3f}±{cv_acc.std():.3f}  F1={cv_f1.mean():.3f}±{cv_f1.std():.3f}")
    print(f"  Test: acc={test_acc:.3f}  F1={test_f1:.3f}")

print("\n" + "=" * 60)
print(" TABLA RESUMEN")
print("=" * 60)
df = pd.DataFrame(rows)
print(df.to_string(index=False))
df.to_csv("resultados_ensemble.csv", index=False)
print("\n✓ resultados_ensemble.csv")

# Ensemble ganador detallado
best_idx = df["Test F1"].astype(float).idxmax()
best_name = df.loc[best_idx, "Ensemble"]
print(f"\n🏆 Mejor ensemble: {best_name}")

# Re-entrenar el ganador y mostrar reporte detallado
print(f"\nReporte detallado del mejor ensemble ({best_name}):")
ensembles[best_name].fit(Xtr_s, y_train)
yp = ensembles[best_name].predict(Xte_s)
print(classification_report(y_test, yp, target_names=CLS, digits=3))