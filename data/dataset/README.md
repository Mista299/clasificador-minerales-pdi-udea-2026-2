# 📦 Banco de Imágenes — Clasificador de Minerales

## Descripción general

Banco de imágenes para clasificación de 4 minerales (cuarzo, pirita, malaquita, hematita) usado en el proyecto de **Procesamiento Digital de Imágenes — Tarea 2, Parte 2** de la Universidad de Antioquia.

- **Total de imágenes:** 240 (60 por clase, más augmentations)
- **División:** 70% train / 15% val / 15% test (estratificada)
- **Resolución:** variable (original) → 128×128 después de preprocesamiento

## Origen del dataset

El banco se construye combinando **dos fuentes** según lo exige el enunciado del taller:

### 1. Dataset público (`tedqc/mineral-dataset`)

**Fuente:** https://huggingface.co/datasets/tedqc/mineral-dataset

- **62,088 imágenes** en total, 781 etiquetas de minerales (split único `train`, 11.4 GB).
- **Autor / curador:** usuario `tedqc` de Hugging Face Hub (dataset subido el 11/2025).
- **Licencia:** Public Domain (CC0).
- **Selección:** se descargaron los shards parquet (`data/train-00000-of-00002.parquet`, `data/train-00001-of-00002.parquet`) y se filtraron las 4 clases objetivo: `quartz` → `cuarzo`, `pyrite` → `pirita`, `malachite` → `malaquita`, `hematite` → `hematita`.
- **Estilo:** fotos de **muestras de colección** (estilo catálogo mineralógico), fondo generalmente neutro o negro, alta calidad.
- **Cámara / equipo de captura:** **no documentado** por el curador. Los archivos JPEG distribuidos **no conservan metadatos EXIF** (verificado: las 168 imágenes originales de train no tienen tag EXIF). Se trata de imágenes escaneadas o re-capturadas de colecciones mineralógicas, no de un único dispositivo.
- **Resolución nativa:** anchura en el rango 689–1024 px (media ≈ 1019 px), altura 467–1024 px (media ≈ 745 px); todas en modo `RGB`, JPEG calidad ≈ 92 al re-guardar.
- **Imágenes usadas:** 168 originales (42 train + 9 val + 9 test por clase × 4 clases); más 168 augmentadas en train (ver § "Augmentation").
- **Limitación importante — _domain gap_:** al ser imágenes de catálogo profesional con iluminación controlada, existe una diferencia de dominio respecto a las fotos caseras con celular. Esta diferencia es la motivación principal del bloque de fotos propias descrito abajo y de la augmentation del train set.

**Comando de descarga:**
```bash
python3 extraer_imagenes.py   # descarga y filtra automáticamente
```

### 2. Fotos propias con celular (test set) — *pendiente de captura*

**Estado:** ⏳ aún no capturadas por el grupo. La estructura del directorio ya está preparada en `dataset/test/<clase>/` para recibirlas con nombre `celular_NNN.jpg` (ver § Pendientes).

**Protocolo de captura planeado:**
- **Cámara:** celular estándar de los integrantes del grupo (marca/modelo a llenar al capturar: _____________).
- **Resolución:** ~3000×4000 px (reducidas a 128×128 en preproc).
- **Iluminación:** luz natural indirecta de ventana, sin flash.
- **Fondo:** cartulina blanca mate (o negra para minerales metálicos claros), siguiendo el consejo del enunciado ("fondo homogéneo y cámara cenital").
- **Distancia:** 25–30 cm perpendicular al mineral.
- **Variabilidad:** 12 fotos por clase, distintas poses, posiciones y rotaciones.
- **Fecha de captura:** ____ / ____ / 2026.
- **Almacenamiento previsto:** `dataset/test/<clase>/celular_NNN.jpg`.

## Montaje fotográfico

```
        ventana
           │
           ▼ (luz natural indirecta)
   ┌───────────────────────┐
   │                       │
   │   cartulina blanca    │
   │                       │
   │   🪨 mineral          │
   │                       │
   └───────────────────────┘
              ▲
              │
        📱 celular
     (perpendicular, 25–30 cm,
      flash apagado, HDR off)
```

## Distribución de imágenes

```
dataset/
├── train/
│   ├── cuarzo/      (90 imgs: 42 originales + 48 augmentadas)
│   ├── pirita/      (90 imgs)
│   ├── malaquita/   (90 imgs)
│   └── hematita/    (90 imgs)
├── val/
│   ├── cuarzo/      (9 imgs)
│   ├── pirita/      (9 imgs)
│   ├── malaquita/   (9 imgs)
│   └── hematita/    (9 imgs)
└── test/
    ├── cuarzo/      (9 imgs dataset + N propias)
    ├── pirita/      (9 imgs dataset + N propias)
    ├── malaquita/   (9 imgs dataset + N propias)
    └── hematita/    (9 imgs dataset + N propias)
```

## Augmentation (solo train)

Para robustez frente a fotos caseras, se aplicaron al train set:

| Transformación | Rango | Probabilidad |
|---|---|---|
| Rotación | ±15° | 100% |
| Flip horizontal | sí/no | 50% |
| Brillo (β) | ±25 (sobre 0–255) | 100% |
| Contraste (α) | 0.8 – 1.2 | 100% |
| Ruido gaussiano | σ ∈ [3, 12] | 50% |
| Blur Gaussiano | k = 3 o 5 | 40% |
| Compresión JPEG | calidad 50–85 | 30% |

Script: `aumentar_datos.py` (no afecta val/test).

## Reproducibilidad

Para regenerar el dataset desde cero:

```bash
# 1. Descargar y filtrar el dataset público (~5 min, 2.4 GB de descarga)
python3 extraer_imagenes.py

# 2. Aplicar augmentation al train (~30 s)
python3 aumentar_datos.py

# 3. (Opcional) Agregar fotos propias
#    Primero crea dataset/test/<clase>/ y copia tus celular_*.jpg, luego:
python3 agregar_fotos_propias.py /ruta/a/tus/fotos/
```

## Citación del dataset original

```
@dataset{tedqc_mineral_dataset,
  title={Mineral Image Dataset (62k imágenes, 781 clases)},
  author={tedqc},
  year={2025},
  url={https://huggingface.co/datasets/tedqc/mineral-dataset}
}
```

## Limitaciones conocidas

- El dataset público es de **muestras de colección** (estilo catálogo de minerales), no de rocas en contexto natural. Esto puede introducir un *domain gap* entre train y test que mitiga el cellphone capture con fotos propias.
- Solo 4 clases: una ampliación a más minerales (galena, magnetita, calcita, etc.) requeriría descargar más archivos parquet.
- Las fotos del dataset están centradas y aisladas; las fotos caseras propias pueden incluir dedos, sombras parciales o fondo menos uniforme, lo que se espera que sea un reto realista del taller.