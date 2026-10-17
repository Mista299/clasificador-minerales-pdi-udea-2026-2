"""Extrae imágenes de las 4 clases y organiza en train/val/test.

Distribución por clase (60 imgs total):
  train: 42  (70%)
  val:    9  (15%)
  test:   9  (15%)  ← estos se reemplazarán luego con fotos propias
"""
from pathlib import Path
import pyarrow.parquet as pq
from PIL import Image
import io
import random

random.seed(42)

OUT = Path(__file__).resolve().parent.parent / "data" / "dataset"
SPLITS = {"train": 42, "val": 9, "test": 9}
TARGET = {"quartz": "cuarzo", "pyrite": "pirita", "malachite": "malaquita", "hematite": "hematita"}
MAX_PER_CLASS = sum(SPLITS.values())  # 60

# Limpiar y recrear estructura
import shutil
if OUT.exists():
    shutil.rmtree(OUT)
for split in SPLITS:
    for cls_es in TARGET.values():
        (OUT / split / cls_es).mkdir(parents=True, exist_ok=True)

# Recolectar índices por clase (sin cargar imágenes todavía)
print("Recolectando índices de imágenes por clase...")
indices_by_class = {k: [] for k in TARGET}

for i in range(11):
    fname = f"/tmp/opencode/parquet/train-000{i:02d}.parquet"
    pf = pq.ParquetFile(fname)
    for rg in range(pf.num_row_groups):
        table = pf.read_row_group(rg)
        labels = table.column("label").to_pylist()
        for j, lbl in enumerate(labels):
            if lbl in TARGET and len(indices_by_class[lbl]) < MAX_PER_CLASS * 2:
                indices_by_class[lbl].append((i, rg, j))
        # Liberar memoria
        del table

print(f"\nÍndices recolectados por clase:")
for k, v in indices_by_class.items():
    print(f"  {k}: {len(v)}")

# Barajar y tomar MAX_PER_CLASS por clase
for k in indices_by_class:
    random.shuffle(indices_by_class[k])
    indices_by_class[k] = indices_by_class[k][:MAX_PER_CLASS]

# Asignar a splits
print("\nExtrayendo y guardando imágenes...")
for cls_en, cls_es in TARGET.items():
    paths = indices_by_class[cls_en]
    # Asignar por split en orden
    split_assign = []
    for split, n in SPLITS.items():
        split_assign.extend([split] * n)
    random.shuffle(paths)
    random.shuffle(split_assign)

    for idx, (file_idx, rg_idx, row_idx) in enumerate(paths):
        split = ["train", "val", "test"][0] if idx < 42 else ("val" if idx < 51 else "test")
        fname = f"/tmp/opencode/parquet/train-000{file_idx:02d}.parquet"
        # Lazy load del row group
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

# Eliminar parquets para liberar espacio
print("\nLiberando parquets...")
for i in range(11):
    f = Path(f"/tmp/opencode/parquet/train-000{i:02d}.parquet")
    if f.exists():
        f.unlink()
print(f"Espacio liberado: {sum(f.stat().st_size for f in Path('/tmp/opencode/parquet').glob('*.parquet') if f.exists()) / 1e6:.1f} MB restantes")