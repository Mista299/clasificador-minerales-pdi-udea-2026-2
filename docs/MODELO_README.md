# Modelo serializado (v2 — voting ensemble)

## Mejora respecto a v1
- **v1:** MLP con Color+GLCM → test 80.6%, CV5 89.2%
- **v2:** Voting ensemble (RF+MLP+SVM) con Color+GLCM → **test 83.3%, CV5 92.8%** ⭐

## Archivo
- `modelo_minerales.pkl` (~1149.5 KB)

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
print(f'{pred.upper()} ({proba.max()*100:.1f}% confianza)')
```

## Métricas del ensemble
- Accuracy en test: 0.833
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
