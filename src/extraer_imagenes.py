"""Extrae imágenes de las 4 clases y organiza en train/val/test.

Descarga los parquets del dataset tedqc/mineral-dataset (HuggingFace) a un
cache local (data/_parquet_cache) bajo demanda: procesa los shards en orden y
se detiene en cuanto cada clase tiene suficientes candidatos.

Distribución por clase (60 imgs total):
  train: 42  (70%)
  val:    9  (15%)
  test:   9  (15%)  ← estos se reemplazarán luego con fotos propias

Minerales incluidos:
  - cuarzo:    translúcido/blanco
  - pirita:    dorado metálico
  - malaquita: verde intenso
  - hematita:  gris metálico oscuro
"""
from pathlib import Path
import pyarrow.parquet as pq
from PIL import Image
import io
import random
import shutil
import sys

for _s in (sys.stdout, sys.stderr):          # consolas cp1252 (Windows)
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

random.seed(42)

REPO = "tedqc/mineral-dataset"
OUT = Path(__file__).resolve().parent.parent / "data" / "dataset"
CACHE = OUT.parent / "_parquet_cache"
SPLITS = {"train": 42, "val": 9, "test": 9}
TARGET = {
    "quartz": "cuarzo",
    "pyrite": "pirita",
    "malachite": "malaquita",
    "hematite": "hematita",
}
MAX_PER_CLASS = sum(SPLITS.values())  # 60

from huggingface_hub import list_repo_files, hf_hub_download

# Limpiar y recrear estructura
if OUT.exists():
    shutil.rmtree(OUT)
for split in SPLITS:
    for cls_es in TARGET.values():
        (OUT / split / cls_es).mkdir(parents=True, exist_ok=True)

# Recolectar índices por clase (descarga shards bajo demanda, solo la columna label)
print("Recolectando índices de imágenes por clase (descargando shards de HuggingFace)...")
indices_by_class = {k: [] for k in TARGET}
parquets = sorted(f for f in list_repo_files(REPO, repo_type="dataset") if f.endswith(".parquet"))

for f in parquets:
    if all(len(v) >= MAX_PER_CLASS * 2 for v in indices_by_class.values()):
        break
    local = hf_hub_download(REPO, f, repo_type="dataset", local_dir=CACHE)
    print(f"  shard {f}")
    pf = pq.ParquetFile(local)
    for rg in range(pf.num_row_groups):
        table = pf.read_row_group(rg, columns=["label"])
        for j, lbl in enumerate(table.column("label").to_pylist()):
            if lbl in TARGET and len(indices_by_class[lbl]) < MAX_PER_CLASS * 2:
                indices_by_class[lbl].append((local, rg, j))
        del table

print(f"\nÍndices recolectados por clase:")
for k, v in indices_by_class.items():
    print(f"  {k}: {len(v)}")

faltan = [k for k, v in indices_by_class.items() if len(v) < MAX_PER_CLASS]
if faltan:
    print(f"⚠ Clases con menos de {MAX_PER_CLASS} candidatos en los shards descargados: {faltan}")

# Barajar y tomar MAX_PER_CLASS por clase
for k in indices_by_class:
    random.shuffle(indices_by_class[k])
    indices_by_class[k] = indices_by_class[k][:MAX_PER_CLASS]

# Asignar a splits
print("\nExtrayendo y guardando imágenes...")
for cls_en, cls_es in TARGET.items():
    paths = indices_by_class[cls_en]
    split_assign = []
    for split, n in SPLITS.items():
        split_assign.extend([split] * n)
    random.shuffle(paths)
    random.shuffle(split_assign)

    for idx, (fname, rg_idx, row_idx) in enumerate(paths):
        split = "train" if idx < 42 else ("val" if idx < 51 else "test")
        pf = pq.ParquetFile(fname)
        table = pf.read_row_group(rg_idx, columns=["image"])
        img_struct = table.column("image")[row_idx].as_py()
        img_bytes = img_struct["bytes"]
        img = Image.open(io.BytesIO(img_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")
        out_path = OUT / split / cls_es / f"{cls_en}_{idx:03d}.jpg"
        img.save(out_path, "JPEG", quality=92)
        del table

print("\n✓ Extracción completada")
print("\nDistribución final:")
for split in SPLITS:
    print(f"\n  {split.upper()}:")
    for cls_es in TARGET.values():
        n = len(list((OUT / split / cls_es).glob("*.jpg")))
        print(f"    {cls_es:12s}: {n} imgs")

# Eliminar el cache de parquets para liberar espacio
print("\nLiberando cache de parquets...")
if CACHE.exists():
    shutil.rmtree(CACHE)
print("✓ Cache eliminado")
