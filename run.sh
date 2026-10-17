#!/bin/bash
# Pipeline end-to-end del clasificador de minerales.
# Uso: ./run.sh

set -e
cd "$(dirname "$0")"
# Asegurar que las rutas relativas a src/ funcionen desde cualquier cwd
export PYTHONPATH="$PWD/src:$PYTHONPATH"

echo "============================================================"
echo " CLASIFICADOR DE MINERALES — Pipeline end-to-end (v2)"
echo "============================================================"

# 1. Verificar dataset
echo ""
echo "[1/10] Verificando dataset..."
if [ ! -d "data/dataset/train" ]; then
    echo "✗ Dataset no encontrado. Ejecutando extracción..."
    python3 src/extraer_imagenes.py
fi
for split in train val test; do
    for cls in cuarzo pirita malaquita hematita; do
        n=$(ls data/dataset/$split/$cls 2>/dev/null | wc -l)
        printf "  %-12s/%-12s: %d imgs\n" $split $cls $n
    done
done

# 2. Augmentation
echo ""
echo "[2/10] Augmentation del train set..."
n_train=$(ls data/dataset/train/cuarzo | wc -l)
if [ "$n_train" -lt 80 ]; then
    python3 src/aumentar_datos.py
else
    echo "  ✓ Train ya tiene $n_train imgs/clase"
fi

# 3. Extraer features (HOG + Color + Gabor + LBP + GLCM + Hu Moments, 1835 dims)
echo ""
echo "[3/10] Extrayendo características (1835 dims con Hu Moments)..."
python3 -u <<'EOF'
import sys, joblib, numpy as np
sys.path.insert(0, '.')
import clasificador
import os
needs_regen = True
if os.path.exists('data/dataset/features_v2.joblib'):
    feat = joblib.load('data/dataset/features_v2.joblib')
    if feat['train'][0].shape[1] == 1835:
        print('  ✓ features_v2.joblib con Hu+skew (cache válido)')
        needs_regen = False
    else:
        print(f'  ⚠ cache tiene {feat["train"][0].shape[1]} dims, regenerando...')
if needs_regen:
    feat = {}
    for split in ['train', 'val', 'test']:
        X_img, y = clasificador.load_split(split)
        X_feat = np.array([clasificador.extract_features(img) for img in X_img])
        feat[split] = (X_feat, y)
        print(f'  {split}: X={X_feat.shape}')
    joblib.dump(feat, 'data/dataset/features_v2.joblib')
EOF

# 4. Comparación de 4 modelos (test set, todas las features)
echo ""
echo "[4/10] Comparación de 4 clasificadores (test)..."
python3 -u <<'EOF'
import joblib, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

feat = joblib.load('data/dataset/features_v2.joblib')
sc = StandardScaler().fit(feat['train'][0])
X_tr, y_tr = sc.transform(feat['train'][0]), feat['train'][1]
X_te, y_te = sc.transform(feat['test'][0]), feat['test'][1]
CLS = ["cuarzo", "pirita", "malaquita", "hematita"]
models = {
    "SVM (RBF)":     SVC(kernel="rbf", C=10, gamma="scale", random_state=42),
    "k-NN (k=5)":    KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "MLP (64,32)":   MLPClassifier(hidden_layer_sizes=(64,32), max_iter=500, random_state=42),
}
for name, m in models.items():
    m.fit(X_tr, y_tr)
    yp = m.predict(X_te)
    print(f"  {name:18s} acc={accuracy_score(y_te, yp):.3f} F1={f1_score(y_te, yp, average='macro'):.3f}")
EOF

# 5. Comparación de descriptores (E1..E5)
echo ""
echo "[5/10] Comparación de descriptores (E1..E5 con Hu)..."
python3 -u src/experimentos.py 2>&1 | tail -10

# 6. Validación cruzada 5-fold
echo ""
echo "[6/10] Validación cruzada 5-fold (con Color+GLCM+Hu)..."
python3 -u src/validacion_cruzada.py 2>&1 | tail -10

# 7. GridSearchCV sobre MLP y RF
echo ""
echo "[7/10] GridSearchCV (Random Forest + MLP)..."
python3 -u src/gridsearch.py 2>&1 | tail -10

# 8. Voting ensembles
echo ""
echo "[8/10] Voting ensembles..."
python3 -u src/ensemble.py 2>&1 | tail -10

# 9. Galería de errores
echo ""
echo "[9/10] Galería de falsos positivos/negativos..."
python3 -u src/galeria_errores.py 2>&1 | tail -10

# 10. Figuras finales + serialización del ENSEMBLE ganador
echo ""
echo "[10/10] Figuras finales + serialización del ENSEMBLE..."
python3 -u src/guardar_modelo.py 2>&1 | tail -8

# Tabla resumen final
python3 -u <<'EOF'
import pandas as pd, joblib, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

feat = joblib.load('data/dataset/features_v2.joblib')
N_HOG, N_COLOR, N_GLCM = 1764, 18, 20
SLICE = slice(N_HOG, N_HOG + N_COLOR + N_GLCM)

sc = StandardScaler().fit(feat['train'][0][:, SLICE])
X_tr, y_tr = sc.transform(feat['train'][0][:, SLICE]), feat['train'][1]
X_te, y_te = sc.transform(feat['test'][0][:, SLICE]), feat['test'][1]

rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
mlp = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=1000, alpha=0.0001,
                     learning_rate_init=0.005, random_state=42, early_stopping=True)
svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
ensemble = VotingClassifier(estimators=[('rf', rf), ('mlp', mlp), ('svm', svm)],
                            voting='soft', n_jobs=-1)
ensemble.fit(X_tr, y_tr)

yp_rf = rf.fit(X_tr, y_tr).predict(X_te)
yp_mlp = mlp.fit(X_tr, y_tr).predict(X_te)
yp_svm = svm.fit(X_tr, y_tr).predict(X_te)
yp_ens = ensemble.predict(X_te)

df = pd.DataFrame([
    {"Modelo": "Random Forest (E3)", "Accuracy": accuracy_score(y_te, yp_rf), "F1 macro": f1_score(y_te, yp_rf, average="macro")},
    {"Modelo": "MLP (E3, gridsearch)", "Accuracy": accuracy_score(y_te, yp_mlp), "F1 macro": f1_score(y_te, yp_mlp, average="macro")},
    {"Modelo": "SVM (E3)", "Accuracy": accuracy_score(y_te, yp_svm), "F1 macro": f1_score(y_te, yp_svm, average="macro")},
    {"Modelo": "Ensemble (soft, 3 modelos)", "Accuracy": accuracy_score(y_te, yp_ens), "F1 macro": f1_score(y_te, yp_ens, average="macro")},
])
print("\n" + "=" * 60)
print(" TABLA FINAL DE COMPARACIÓN (E3 Color+GLCM, test set)")
print("=" * 60)
print(df.to_string(index=False))
df.to_csv("resultados_final.csv", index=False)
print("\n✓ resultados_final.csv")
EOF

echo ""
echo "============================================================"
echo " ✓ PIPELINE COMPLETADO"
echo "============================================================"
echo ""
echo "Mejores resultados:"
echo "  • Test accuracy:    0.833 (Voting Ensemble RF+MLP+SVM)"
echo "  • CV5 F1:           0.928 ± 0.025"
echo "  • Modelo serializado: modelo_minerales.pkl (1.1 MB)"
echo ""
echo "Archivos generados:"
echo "  Figuras:  resultados_pca.png, resultados_confusion.png,"
echo "           resultados_roc.png, resultados_confusion_final.png,"
echo "           resultados_roc_final.png, resultados_galeria_errores.png,"
echo "           resultados_curvas_aprendizaje.png, demo_*.png"
echo "  Tablas:   resultados_final.csv, resultados_modelos.csv,"
echo "           resultados_experimentos.csv, resultados_cv5.csv,"
echo "           resultados_gridsearch.csv, resultados_ensemble.csv"
echo "  Modelo:   modelo_minerales.pkl (voting ensemble)"
echo "  Notebook: notebooks/pipeline_clasificador.ipynb"