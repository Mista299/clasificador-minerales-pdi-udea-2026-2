# 📋 Plan: Ampliar el dataset a 6 minerales (+2)

## 🎯 Objetivo

Pasar de **4 → 6 clases** agregando dos minerales nuevos al banco de imágenes:
- **Calcita** (CaCO₃) — blanco/incoloro, cristales romboédricos
- **Amatista** (SiO₂ + impurezas Fe) — violeta/lila, variante del cuarzo

## 📊 Justificación de la elección

| Mineral | Color | Textura | Disponibilidad HF | Razón |
|---|---|---|---|---|
| **Calcita** | Blanco/incoloro | Cristales romboédricos | ~1,090 imgs ✅ | Color único entre los actuales; primer mineral sin color saturado |
| **Amatista** | Violeta/lila | Vítrea cristalina | ~1,100 imgs ✅ | Color violeta único; el modelo aprende una subclase dentro del cuarzo |

**Verificación de disponibilidad (HuggingFace `tedqc/mineral-dataset`, 5 shards muestreados, extrapolado a 58):**

| Mineral | Conteo (5 shards) | Estimado total |
|---|---|---|
| quartz | 576 | ~6,700 |
| hematite | 186 | ~2,150 |
| topaz | 257 | ~3,000 |
| chalcedony | 212 | ~2,500 |
| calcite | 94 | ~1,090 ✅ |
| amethyst | 96 | ~1,100 ✅ |
| fluorite | 65 | ~750 |
| pyrite | 52 | ~600 |
| galena | 39 | ~450 ⚠️ |
| azurite | 12 | ~140 ❌ |

**Total: 6 clases con buena diversidad visual:**
- 2 minerales claros/translúcidos (cuarzo, calcita)
- 2 minerales con color distintivo (malaquita verde, amatista violeta)
- 2 minerales metálicos/oscuros (pirita dorada, hematita gris-rojo)

## 📝 Cambios necesarios

### Fase 1: Modificar `CLASSES` (3 archivos)

**`src/clasificador.py`** (línea 27):
```python
CLASSES = ["cuarzo", "pirita", "malaquita", "hematita", "calcita", "galena"]
```

**`src/clasificador_v2.py`** (línea 23):
```python
CLASSES = ["cuarzo", "pirita", "malaquita", "hematita", "calcita", "galena"]
```

**`src/aumentar_datos.py`** (línea 18):
```python
CLASSES = ["cuarzo", "pirita", "malaquita", "hematita", "calcita", "galena"]
```

### Fase 2: Actualizar `src/extraer_imagenes.py`

Agregar al mapeo de etiquetas inglés→español:
```python
TARGET = {
    "quartz": "cuarzo",
    "pyrite": "pirita",
    "malachite": "malaquita",
    "hematite": "hematita",
    "calcite": "calcita",     # NUEVO - blanco/incoloro
    "amethyst": "amatista",  # NUEVO - violeta/lila
}
```

**Verificar disponibilidad en HuggingFace:**
```bash
python3 -c "
from huggingface_hub import list_repo_files
import pandas as pd

files = list_repo_files('tedqc/mineral-dataset', repo_type='dataset')
print('Total parquet files:', len([f for f in files if f.endswith('.parquet')]))
print('Búsqueda en README...')
"
```

### Fase 3: Re-descargar y reentrenar

```bash
# Backup del dataset actual
cp -r data/dataset data/dataset_v4_backup

# Eliminar dataset actual y re-descargar (ahora con 6 clases)
rm -rf data/dataset/*
python3 src/extraer_imagenes.py   # descarga 60 imgs × 6 clases = 360

# Augmentar train (60 → 90 imgs/clase)
python3 src/aumentar_datos.py

# Re-extraer features v2 (incluye histograma HSV)
python3 src/pipeline_v3.py --solo-parte2
```

### Fase 4: Validar resultados

```bash
# Ver accuracy reportada
cat results/tables/resultados_v3.csv

# Comparar con accuracy anterior (83.3%):
#   - Si accuracy > 80%: ✅ modelo escaló bien
#   - Si accuracy 70-80%: ⚠️ necesita ajuste de hiperparámetros
#   - Si accuracy < 70%: ❌ reconsiderar los minerales elegidos
```

### Fase 5: Actualizar documentación

| Archivo | Cambio |
|---|---|
| `data/dataset/README.md` | Actualizar lista de minerales (cuarzo, pirita, malaquita, hematita → + calcita, galena) |
| `docs/informe_ieee.docx` | Regenerar tablas (6 clases en matriz de confusión) |
| `MODEL_PATH` (`app.py`) | Actualizar selector con 6 minerales |
| `model/modelo_minerales_v3.pkl` | Regenerar después de entrenar |

### Fase 6 (opcional): Fotos caseras

```bash
# Tomar 10+ fotos de cada mineral nuevo (calcita, galena)
python3 src/agregar_fotos_propias.py /ruta/a/fotos/calcita/
python3 src/agregar_fotos_propias.py /ruta/a/fotos/galena/

# Reentrenar para mejorar generalización
python3 ejecutar_todo.py --solo-parte2
```

## ⚠️ Riesgos y mitigaciones

| Riesgo | Impacto | Mitigación |
|---|---|---|
| Calcita muy similar al cuarzo blanco | Confusión cuarzo-calcita | Aumentar fotos con variación de iluminación |
| Amatista es variante del cuarzo | Posible confusión cuarzo-amatista | Revisar la matriz de confusión para detectar este patrón |
| Accuracy cae >15% | El modelo se vuelve poco confiable | Reducir a 5 clases (solo calcita) |
| Malaquita tiene solo ~220 imgs en HF | Margen limitado para augmentation | Mantener 60 imgs/clase (suficiente) |

## 📊 Métricas esperadas

| Escenario | Accuracy CV-5 | Accuracy Test |
|---|---|---|
| Calcita + Amatista sin fotos caseras | 80-88% | 75-82% |
| Con fotos caseras (10/clase) | 85-92% | 80-88% |
| Si accuracy < 70% | Reconsiderar minerales |

## 🗓️ Cronograma

| Día | Tarea | Tiempo |
|---|---|---|
| Día 1 mañana | Verificar dataset HF, modificar código | 2 h |
| Día 1 tarde | Re-descargar + entrenar + evaluar | 2 h |
| Día 2 mañana | Actualizar docs + regenerar informe | 2 h |
| Día 2 tarde | Fotos caseras (si hay tiempo) + reentrenar | 2 h |

**Total: 1-2 días**

## 📌 Lista de verificación

- [x] Verificar disponibilidad de calcite y amethyst en HuggingFace ✅ (~1100 imgs cada uno)
- [ ] Comprar/adquiririr muestras físicas (si se desea fotos caseras)
- [ ] Backup del dataset y modelo actuales
- [ ] Modificar `CLASSES` en 3 archivos: añadir "calcita", "amatista"
- [ ] Modificar mapeo en `extraer_imagenes.py`: añadir "calcite" → "calcita", "amethyst" → "amatista"
- [ ] Re-descargar dataset
- [ ] Reentrenar pipeline v3
- [ ] Validar accuracy (esperado >75%)
- [ ] Actualizar README del dataset
- [ ] Regenerar informe IEEE con 6 clases
- [ ] Actualizar `app.py`
- [ ] Tomar fotos caseras (opcional)
- [ ] Commit y push al repo