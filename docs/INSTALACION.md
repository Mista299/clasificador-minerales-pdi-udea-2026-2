# 🛠 Guía de instalación

## Opción A — Ya tienes Python con todo (caso actual)

Estás listo. No instales nada. Ejecuta directamente:

```bash
cd /home/mista/Escritorio/udea/2026-II/procesamiento-digital/trabajo2
./run.sh
```

## Opción B — Instalar en una máquina nueva

```bash
# 1. Clonar el proyecto
git clone <repo> trabajo2
cd trabajo2

# 2. Crear venv (recomendado)
python3 -m venv venv
source venv/bin/activate   # Linux/Mac
# .\venv\Scripts\activate  # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Verificar instalación
python3 -c "import cv2, sklearn, skimage; print('OK')"

# 5. (Solo si re-descargas el dataset desde HuggingFace)
python3 extraer_imagenes.py

# 6. Entrenar y ejecutar
./run.sh
python3 predecir.py --demo
```

## Verificar qué hay instalado

```bash
python3 -c "
import cv2, numpy, sklearn, skimage, matplotlib, joblib, pandas
print('opencv-python:', cv2.__version__)
print('numpy:', numpy.__version__)
print('scikit-learn:', sklearn.__version__)
print('scikit-image:', skimage.__version__)
print('matplotlib:', matplotlib.__version__)
print('joblib:', joblib.__version__)
print('pandas:', pandas.__version__)
"
```

Salida esperada:
```
opencv-python: 5.0.0
numpy: 2.4.4
scikit-learn: 1.8.0
scikit-image: 0.26.0
matplotlib: 3.10.8
joblib: 1.5.3
pandas: 3.0.2
```

## Versiones de Python soportadas

- ✅ **Python 3.10 – 3.14**
- ❌ Python 3.9 o inferior (puede fallar por cambios en numpy/joblib)

## Dependencias opcionales

- **PyTorch** — solo si quieres añadir una CNN con transfer learning
- **Gradio** — para la interfaz gráfica del compañero
- **pyarrow + huggingface_hub** — solo si re-descargas el dataset desde HuggingFace