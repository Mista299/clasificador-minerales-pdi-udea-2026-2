"""Serializa el voting ensemble como modelo principal para la GUI.

El ensemble (RF + MLP + SVM con soft voting) logra 92.8% en CV5-fold
y 83.3% en test, siendo el mejor resultado del proyecto.

Uso desde la GUI (compañero):
    import joblib
    bundle = joblib.load('modelo_minerales.pkl')
    modelo = bundle['modelo']     # VotingClassifier
    scaler = bundle['scaler']
    sl = slice(bundle['slice_start'], bundle['slice_end'])
    clases = bundle['clases']

    feats_all = clasificador.extract_features(img_bgr)
    feats_slice = feats_all[sl]
    feats_s = scaler.transform(feats_slice.reshape(1, -1))
    pred = modelo.predict(feats_s)[0]
    proba = modelo.predict_proba(feats_s)[0]
"""
import sys, joblib, numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

sys.path.insert(0, '.')
import clasificador

print("=" * 60)
print(" Serialización del VOTING ENSEMBLE (modelo principal)")
print("=" * 60)

# Cargar features
feat = joblib.load("data/dataset/features_v2.joblib")
X_train, y_train = feat["train"]
X_test, y_test = feat["test"]

# Slice E3: Color + GLCM (con skew, sin HOG ni Hu)
N_HOG = 1764
N_COLOR = 18
N_GLCM = 20
SLICE = slice(N_HOG, N_HOG + N_COLOR + N_GLCM)

X_tr = X_train[:, SLICE]
X_te = X_test[:, SLICE]
sc = StandardScaler().fit(X_tr)
X_tr_s = sc.transform(X_tr); X_te_s = sc.transform(X_te)

# Construir ensemble ganador
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
mlp = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=1000, alpha=0.0001,
                     learning_rate_init=0.005, random_state=42, early_stopping=True)
svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)

ensemble = VotingClassifier(
    estimators=[("rf", rf), ("mlp", mlp), ("svm", svm)],
    voting="soft", n_jobs=-1
)
ensemble.fit(X_tr_s, y_train)

# Validar en test
yp = ensemble.predict(X_te_s)
acc = accuracy_score(y_test, yp)
print(f"\nAccuracy en test: {acc:.3f}")
print(classification_report(y_test, yp, target_names=clasificador.CLASSES, digits=3))

# Guardar bundle
bundle = {
    "modelo": ensemble,
    "scaler": sc,
    "slice": SLICE,
    "slice_start": N_HOG,
    "slice_end": N_HOG + N_COLOR + N_GLCM,
    "clases": clasificador.CLASSES,
    "n_features": N_COLOR + N_GLCM,  # 38
    "accuracy_test": acc,
    "accuracy_cv5": 0.928,  # del ensemble.py
    "modelo_tipo": "VotingClassifier (Soft, RF+MLP+SVM)",
    "version": "2.0",
    "descripcion": (
        "Clasificador de minerales (Cuarzo/Pirita/Malaquita/Hematita). "
        "Voting ensemble (soft) de RF + MLP + SVM, con descriptores Color+GLCM "
        "(38 dims). Accuracy CV5=92.8%, test=83.3%."
    ),
}

OUT = Path(__file__).resolve().parent.parent / "model" / "modelo_minerales.pkl"
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT = str(OUT)
joblib.dump(bundle, OUT)
size_kb = Path(OUT).stat().st_size / 1024
print(f"\n✓ Modelo (ensemble) guardado en: {OUT} ({size_kb:.1f} KB)")

# Verificar carga
bundle_test = joblib.load(OUT)
print(f"  Tipo:      {bundle_test['modelo_tipo']}")
print(f"  Clases:    {bundle_test['clases']}")
print(f"  Slice:     [{bundle_test['slice_start']}:{bundle_test['slice_end']}]  ({bundle_test['n_features']} dims)")
print(f"  Acc test:  {bundle_test['accuracy_test']:.3f}")
print(f"  Acc CV5:   {bundle_test['accuracy_cv5']:.3f}")

# Actualizar README de uso
readme = """# Modelo serializado (v2 — voting ensemble)

## Mejora respecto a v1
- **v1:** MLP con Color+GLCM → test 80.6%, CV5 89.2%
- **v2:** Voting ensemble (RF+MLP+SVM) con Color+GLCM → **test 83.3%, CV5 92.8%** ⭐

## Archivo
- `modelo_minerales.pkl` (~{size:.1f} KB)

## Contenido
```python
bundle = joblib.load('modelo_minerales.pkl')
modelo = bundle['modelo']        # VotingClassifier (RF+MLP+SVM, soft)
scaler = bundle['scaler']        # StandardScaler
clases = bundle['clases']        # ['cuarzo', 'pirita', 'malaquita', 'hematita']
slice_start = bundle['slice_start']  # 1764
slice_end = bundle['slice_end']      # 1802
```

## Uso desde la interfaz

```python
import sys; sys.path.insert(0, '.')
import joblib, cv2, numpy as np
import clasificador

bundle = joblib.load('modelo_minerales.pkl')
modelo = bundle['modelo']
scaler = bundle['scaler']

def predecir(img_bgr):
    feats_all = clasificador.extract_features(img_bgr)
    feats_slice = feats_all[bundle['slice_start']:bundle['slice_end']]
    feats_s = scaler.transform(feats_slice.reshape(1, -1))
    idx = modelo.predict(feats_s)[0]
    proba = modelo.predict_proba(feats_s)[0]
    return bundle['clases'][idx], proba

# Ejemplo
img = cv2.imread('foto_cuartzo.jpg')
pred, proba = predecir(img)
print(f'{{pred.upper()}} ({{proba.max()*100:.1f}}% confianza)')
```

## Métricas del ensemble
- Accuracy en test: {acc:.3f}
- Accuracy CV 5-fold: 0.928 ± 0.025
- F1 macro CV 5-fold: 0.928 ± 0.025

## Desglose por clase (test)
| Clase | P | R | F1 |
|---|---|---|---|
| Cuarzo | 0.80 | 0.89 | 0.84 |
| Pirita | 0.70 | 0.78 | 0.74 |
| Malaquita | 1.00 | 0.89 | 0.94 |
| Hematita | 0.88 | 0.78 | 0.82 |

## Dependencias para usar el modelo
- numpy
- scikit-learn (>=1.5)
- opencv-python
- scikit-image
""".format(size=size_kb, acc=acc)

Path("MODELO_README.md").write_text(readme, encoding="utf-8")
print(f"\n✓ Instrucciones actualizadas en: MODELO_README.md")