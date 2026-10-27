"""Agrega fotos propias del celular al dataset.

Uso:
    python3 agregar_fotos_propias.py <carpeta_origen>

La carpeta origen debe tener subcarpetas (en español o inglés):
    <carpeta_origen>/
    ├── cuarzo/      <- fotos de cuarzo
    ├── pirita/      <- fotos de pirita
    ├── malaquita/   <- fotos de malaquita
    └── hematita/    <- fotos de hematita

O nombres en inglés:
    quartz/  pyrite/  malachite/  hematite/

Las fotos se copian a dataset/test/<clase_es>/celular_NNN.jpg
"""
import sys
from pathlib import Path
import shutil
import cv2
import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data" / "dataset"
SPLIT = "test"   # las fotos propias van al test set
CLASS_MAP = {
    "cuarzo": "cuarzo", "quartz": "cuarzo",
    "pirita": "pirita", "pyrite": "pirita",
    "malaquita": "malaquita", "malachite": "malaquita",
    "hematita": "hematita", "hematite": "hematita",
}

ALL_CLASSES = ["cuarzo", "pirita", "malaquita", "hematita"]


def main():
    if len(sys.argv) != 2:
        print("Uso: python3 agregar_fotos_propias.py <carpeta_origen>")
        sys.exit(1)

    src = Path(sys.argv[1])
    if not src.exists():
        print(f"✗ Carpeta no encontrada: {src}")
        sys.exit(1)

    total = 0
    print(f"Procesando fotos propias desde: {src.resolve()}")
    print(f"Destino: dataset/{SPLIT}/<clase>/celular_NNN.jpg\n")

    for src_dir in sorted(src.iterdir()):
        if not src_dir.is_dir():
            continue
        cls_en = src_dir.name.lower()
        cls_es = CLASS_MAP.get(cls_en)
        if cls_es is None:
            print(f"⚠ Carpeta '{src_dir.name}' no reconocida, saltando")
            continue

        dst_dir = DATA / SPLIT / cls_es
        dst_dir.mkdir(parents=True, exist_ok=True)

        # Buscar siguiente índice
        existing = sorted(dst_dir.glob("celular_*.jpg"))
        next_idx = 1
        if existing:
            last = existing[-1].stem
            try:
                next_idx = int(last.split("_")[-1]) + 1
            except ValueError:
                pass

        added = 0
        for f in sorted(src_dir.iterdir()):
            if f.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            # Validar que se puede leer
            img = cv2.imdecode(np.fromfile(str(f), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                print(f"  ⚠ No se pudo leer: {f.name}")
                continue
            if img.shape[0] < 50 or img.shape[1] < 50:
                print(f"  ⚠ Muy pequeña: {f.name} ({img.shape})")
                continue
            dst = dst_dir / f"celular_{next_idx:03d}.jpg"
            shutil.copy2(f, dst)
            added += 1
            next_idx += 1
            total += 1

        print(f"  ✓ {cls_es:12s}: {added} fotos agregadas")

    print(f"\n{'='*60}")
    print(f" ✓ {total} fotos propias agregadas al test set")
    print(f"{'='*60}")
    print("\nDistribución final del test set:")
    for cls in ALL_CLASSES:
        n = len(list((DATA / SPLIT / cls).glob("*.jpg")))
        print(f"  {cls:12s}: {n} imgs")
    print("\nEjecuta './run.sh' para re-entrenar y ver la nueva accuracy.")


if __name__ == "__main__":
    main()