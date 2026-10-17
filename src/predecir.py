"""Predicción CLI (sin GUI) — uso:
    python3 predecir.py <imagen>
    python3 predecir.py <carpeta>            # procesa todas las imágenes
    python3 predecir.py --demo                # prueba con 4 imágenes del test
"""
import sys, joblib, cv2, numpy as np
from pathlib import Path

sys.path.insert(0, '.')
import clasificador

CLASS_COLORS = {
    "cuarzo":   "\033[94m",   # azul
    "pirita":   "\033[93m",   # amarillo
    "malaquita":"\033[92m",   # verde
    "hematita": "\033[91m",   # rojo
}
RESET = "\033[0m"


def predecir(path_imagen: str, bundle: dict) -> tuple[str, np.ndarray]:
    """Devuelve (clase_predicha, vector_de_probabilidades)."""
    img_bgr = cv2.imread(path_imagen)
    if img_bgr is None:
        raise FileNotFoundError(f"No se pudo leer: {path_imagen}")
    feats_all = clasificador.extract_features(img_bgr)
    feats_slc = feats_all[bundle["slice_start"]:bundle["slice_end"]]
    feats_s   = bundle["scaler"].transform(feats_slc.reshape(1, -1))
    idx       = bundle["modelo"].predict(feats_s)[0]
    proba     = bundle["modelo"].predict_proba(feats_s)[0]
    return bundle["clases"][idx], proba


def barra(prob: float, ancho: int = 30) -> str:
    n = int(prob * ancho)
    return "█" * n + "░" * (ancho - n)


def mostrar_resultado(path: str, clase_pred: str, proba: np.ndarray, clases: list):
    name = Path(path).name
    color = CLASS_COLORS.get(clase_pred, "")
    print(f"\n{'─' * 60}")
    print(f"📷 {path}")
    print(f"{'─' * 60}")
    print(f"  {color}▶ Predicción: {clase_pred.upper()}{RESET}  ({proba[clases.index(clase_pred)]*100:.1f}% confianza)")
    print(f"\n  Distribución de probabilidad:")
    for c, p in zip(clases, proba):
        c_color = CLASS_COLORS.get(c, "")
        bar = barra(p)
        marker = " ◀" if c == clase_pred else "  "
        print(f"    {c_color}{c:10s}{RESET} {bar} {p*100:5.1f}%{marker}")
    print(f"{'─' * 60}")


def main():
    if len(sys.argv) < 2:
        print("Uso:")
        print("  python3 predecir.py <imagen.jpg>")
        print("  python3 predecir.py <carpeta/>")
        print("  python3 predecir.py --demo")
        sys.exit(1)

    # Cargar modelo (solo una vez)
    print("Cargando modelo modelo_minerales.pkl...")
    MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "modelo_minerales.pkl"
    bundle = joblib.load(str(MODEL_PATH))
    clases = bundle["clases"]
    print(f"  Clases: {clases}")
    print(f"  Accuracy reportado: {bundle['accuracy_test']:.3f}")

    target = sys.argv[1]

    if target == "--demo":
        # Demo: probar con 1 imagen de cada clase del test
        print("\n=== MODO DEMO ===")
        for cls in clases:
            files = list(Path(f"data/dataset/test/{cls}").glob("*.jpg"))
            if files:
                f = str(files[0])
                clase_pred, proba = predecir(f, bundle)
                ok = "✓" if clase_pred == cls else "✗"
                print(f"\n{ok} {cls} → {clase_pred} ({proba[clases.index(clase_pred)]*100:.1f}%)")
        return

    p = Path(target)
    if p.is_dir():
        # Procesar todas las imágenes de la carpeta
        extensiones = {".jpg", ".jpeg", ".png", ".bmp"}
        imagenes = sorted([f for f in p.iterdir() if f.suffix.lower() in extensiones])
        if not imagenes:
            print(f"✗ No se encontraron imágenes en {target}")
            sys.exit(1)

        print(f"\nProcesando {len(imagenes)} imágenes en {p.resolve()}...")
        resultados = []
        for f in imagenes:
            clase_pred, proba = predecir(str(f), bundle)
            resultados.append((f.name, clase_pred, proba.max()))
            mostrar_resultado(str(f), clase_pred, proba, clases)

        # Resumen
        print(f"\n{'═' * 60}")
        print(f" RESUMEN: {len(resultados)} imágenes procesadas")
        from collections import Counter
        conteo = Counter(r[1] for r in resultados)
        for c in clases:
            n = conteo.get(c, 0)
            print(f"  {c:12s}: {n} imgs")
        print(f"{'═' * 60}")

    elif p.is_file():
        clase_pred, proba = predecir(str(p), bundle)
        mostrar_resultado(str(p), clase_pred, proba, clases)
    else:
        print(f"✗ No existe: {target}")
        sys.exit(1)


if __name__ == "__main__":
    main()