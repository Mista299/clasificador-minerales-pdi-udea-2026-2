"""Augmentation para hacer el dataset más robusto a fotos caseras.

Aplica transformaciones SOLO al train set:
  - Rotaciones aleatorias (±15°)
  - Flip horizontal
  - Cambios de brillo/contraste
  - Ruido gaussiano (simula cámara de celular)
  - Blur ligero (simula movimiento de celular)

NO toca val/test (deben quedar originales para evaluación justa).
"""
from pathlib import Path
import random
import numpy as np
import cv2

DATA = Path(__file__).resolve().parent.parent / "data" / "dataset"
CLASSES = ["cuarzo", "pirita", "malaquita", "hematita"]
TARGET_PER_CLASS = 90   # aumentar de 42 a 90 imgs en train
SEED = 42

random.seed(SEED)
np.random.seed(SEED)


def augment_image(img):
    """Aplica augmentations aleatorias para simular fotos caseras."""
    h, w = img.shape[:2]
    out = img.copy()

    # 1. Rotación ±15°
    angle = random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    out = cv2.warpAffine(out, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # 2. Flip horizontal (50% prob)
    if random.random() < 0.5:
        out = cv2.flip(out, 1)

    # 3. Brillo/contraste (simula distintas iluminaciones caseras)
    alpha = random.uniform(0.8, 1.2)   # contraste
    beta = random.uniform(-25, 25)     # brillo
    out = np.clip(alpha * out + beta, 0, 255).astype(np.uint8)

    # 4. Ruido gaussiano (simula sensor de celular barato)
    if random.random() < 0.5:
        sigma = random.uniform(3, 12)
        noise = np.random.normal(0, sigma, out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 5. Blur ligero (simula movimiento de manos)
    if random.random() < 0.4:
        k = random.choice([3, 5])
        out = cv2.GaussianBlur(out, (k, k), 0)

    # 6. JPEG compression artifacts (celular guarda en JPEG)
    if random.random() < 0.3:
        q = random.randint(50, 85)
        _, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, q])
        out = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    return out


def main():
    print("=" * 60)
    print(" Augmentation del train set (fotos caseras simuladas)")
    print("=" * 60)
    print(f"Target: {TARGET_PER_CLASS} imgs/clase en train (actualmente 42)")
    print(f"Transformaciones: rotación ±15°, flip H, brillo, ruido, blur, JPEG\n")

    for cls in CLASSES:
        train_dir = DATA / "train" / cls
        existing = sorted(train_dir.glob("*.jpg"))
        n_to_generate = max(0, TARGET_PER_CLASS - len(existing))
        print(f"  {cls}: {len(existing)} originales, generando {n_to_generate} augmentadas...")

        # Generar augmentaciones a partir de las imágenes existentes
        for i in range(n_to_generate):
            src = existing[i % len(existing)]
            img = cv2.imdecode(np.fromfile(str(src), np.uint8), cv2.IMREAD_COLOR)  # rutas Unicode
            aug = augment_image(img)
            out_name = f"aug_{i:03d}.jpg"
            ok, buf = cv2.imencode(".jpg", aug, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if ok:
                buf.tofile(str(train_dir / out_name))

    print("\n✓ Augmentation completada")
    print("\nDistribución final:")
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            n = len(list((DATA / split / cls).glob("*.jpg")))
            print(f"  {split:6s}/{cls:12s}: {n}")


if __name__ == "__main__":
    main()