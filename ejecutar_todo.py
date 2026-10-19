# -*- coding: utf-8 -*-
"""Ejecutor centralizado — Tarea 2 PDI (Parte 1: taller de filtros + Parte 2: clasificador).

Uso:
    python ejecutar_todo.py                # Parte 1 + Parte 2 completas
    python ejecutar_todo.py --solo-parte1  # solo el taller de filtros (notebooks .py)
    python ejecutar_todo.py --solo-parte2  # solo el pipeline del clasificador
    python ejecutar_todo.py --app          # todo lo anterior + abre la interfaz web
    python ejecutar_todo.py --solo-app     # solo la interfaz web (streamlit)

Notas:
  * Parte 1 ejecuta los notebooks .py como scripts y guarda TODAS las figuras en
    results/figures_taller/<notebook>/ (no abre ventanas).
  * Parte 2 equivale a run.sh pero en Python puro (funciona igual en Windows/Linux).
    Si falta data/dataset intenta descargarlo; si no puede, omite la Parte 2 con aviso.
  * Las imágenes propias del taller se configuran en el dict RUTAS de cada notebook.
"""
import argparse
import os
import runpy
import subprocess
import sys
import time
from pathlib import Path

for _s in (sys.stdout, sys.stderr):          # consolas cp1252 (Windows): UTF-8 sin romper
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
NOTEBOOKS = ROOT / "notebook"
RESULTS = ROOT / "results"
FIGS_TALLER = RESULTS / "figures_taller"

TALLERES = [
    NOTEBOOKS / "tarea2_parte1_modulos_a_b_(2).py",
    NOTEBOOKS / "tarea2_parte1_modulos_c_d_e.py",
]
CLASES = ["cuarzo", "pirita", "malaquita", "hematita"]


def titulo(txt):
    print("\n" + "=" * 64 + f"\n {txt}\n" + "=" * 64)


def subpaso(txt):
    print(f"\n── {txt} " + "─" * max(0, 56 - len(txt)))


# ---------------------------------------------------------------- Parte 1 ----
def correr_taller(ruta_py: Path) -> int:
    """Ejecuta un notebook exportado a .py con backend Agg, guardando cada figura."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = FIGS_TALLER / ruta_py.stem
    outdir.mkdir(parents=True, exist_ok=True)
    contador = [0]

    def guardar_y_cerrar(*_a, **_k):
        for num in plt.get_fignums():
            contador[0] += 1
            plt.figure(num).savefig(outdir / f"fig_{contador[0]:02d}.png",
                                    dpi=110, bbox_inches="tight")
        plt.close("all")

    plt.show = guardar_y_cerrar          # cada plt.show() del notebook -> PNG
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.chdir(NOTEBOOKS)                   # por si el notebook usa rutas relativas
    try:
        runpy.run_path(str(ruta_py), run_name="__main__")
    finally:
        os.chdir(ROOT)
        plt.show = lambda *a, **k: None
    return contador[0]


def parte1():
    titulo("PARTE 1 — Taller de filtros (secciones 2.1–2.6)")
    for nb in TALLERES:
        if not nb.exists():
            print(f"  ⚠ No se encontró {nb.name}, se omite.")
            continue
        subpaso(f"Ejecutando {nb.name}")
        t0 = time.perf_counter()
        n = correr_taller(nb)
        print(f"  ✓ {nb.name}: {n} figuras guardadas en "
              f"results/figures_taller/{nb.stem}/  ({time.perf_counter()-t0:.0f} s)")


# ---------------------------------------------------------------- Parte 2 ----
def py(cmd, descripcion, permitir_fallo=True):
    """Ejecuta `python <cmd>` desde la raíz del proyecto con src/ en PYTHONPATH."""
    subpaso(descripcion)
    env = dict(os.environ,
               PYTHONPATH=str(SRC) + os.pathsep + os.environ.get("PYTHONPATH", ""),
               PYTHONUTF8="1", PYTHONIOENCODING="utf-8")  # consola cp1252 (Windows)
    r = subprocess.run([sys.executable, "-u"] + cmd, cwd=ROOT, env=env,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    lineas = (r.stdout + r.stderr).strip().splitlines()
    for l in lineas[-12:]:
        print("   ", l)
    if r.returncode != 0:
        print(f"   ✗ FALLO (código {r.returncode}) — se continúa con el siguiente paso")
        return False
    return True


def dataset_ok():
    # El dataset vale sólo si hay imágenes reales en cada clase del train
    return all(contar("train", c) > 0 for c in CLASES)


def contar(split, clase):
    d = ROOT / "data/dataset" / split / clase
    return len(list(d.iterdir())) if d.is_dir() else 0


def extraer_features():
    """Equivale al paso [3/10] de run.sh: cachea features_v2.joblib (1835 dims)."""
    import joblib
    import numpy as np
    sys.path.insert(0, str(SRC))
    import clasificador

    cache = ROOT / "data/dataset/features_v2.joblib"
    if cache.exists():
        feat = joblib.load(cache)
        X = feat["train"][0]
        if X.ndim == 2 and X.shape[0] > 0 and X.shape[1] == 1835:
            print("   ✓ features_v2.joblib con Hu+skew (cache válido)")
            return
        print("   ⚠ cache inválido o vacío, regenerando...")
    feat = {}
    for split in ("train", "val", "test"):
        X_img, y = clasificador.load_split(split)
        X_feat = np.array([clasificador.extract_features(i) for i in X_img])
        feat[split] = (X_feat, y)
        print(f"   {split}: X={X_feat.shape}")
    joblib.dump(feat, cache)


def comparacion_4_modelos():
    """Equivale al paso [4/10] de run.sh."""
    import joblib
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score, f1_score

    feat = joblib.load(ROOT / "data/dataset/features_v2.joblib")
    sc = StandardScaler().fit(feat["train"][0])
    X_tr, y_tr = sc.transform(feat["train"][0]), feat["train"][1]
    X_te, y_te = sc.transform(feat["test"][0]), feat["test"][1]
    for name, m in {
        "SVM (RBF)": SVC(kernel="rbf", C=10, gamma="scale", random_state=42),
        "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "MLP (64,32)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
    }.items():
        m.fit(X_tr, y_tr)
        yp = m.predict(X_te)
        print(f"   {name:18s} acc={accuracy_score(y_te, yp):.3f} F1={f1_score(y_te, yp, average='macro'):.3f}")


def resumen_final():
    """Equivale a la tabla final de run.sh (ensemble E3 sobre test)."""
    import joblib
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, f1_score

    feat = joblib.load(ROOT / "data/dataset/features_v2.joblib")
    SLICE = slice(1764, 1764 + 18 + 20)          # E3 = Color + GLCM
    sc = StandardScaler().fit(feat["train"][0][:, SLICE])
    X_tr, y_tr = sc.transform(feat["train"][0][:, SLICE]), feat["train"][1]
    X_te, y_te = sc.transform(feat["test"][0][:, SLICE]), feat["test"][1]

    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=1000, alpha=0.0001,
                        learning_rate_init=0.005, random_state=42, early_stopping=True)
    svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)
    ens = VotingClassifier([("rf", rf), ("mlp", mlp), ("svm", svm)], voting="soft", n_jobs=-1)

    filas = []
    for nombre, m in [("Random Forest (E3)", rf), ("MLP (E3, gridsearch)", mlp),
                      ("SVM (E3)", svm), ("Ensemble (soft, 3 modelos)", ens)]:
        m.fit(X_tr, y_tr)
        yp = m.predict(X_te)
        filas.append({"Modelo": nombre, "Accuracy": round(accuracy_score(y_te, yp), 3),
                      "F1 macro": round(f1_score(y_te, yp, average="macro"), 3)})
    df = pd.DataFrame(filas)
    print("\n" + df.to_string(index=False))
    out = RESULTS / "tables/resultados_final.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n   ✓ {out.relative_to(ROOT)}")


def intentar(fn, descripcion):
    """Ejecuta un paso inline; si falla, avisa y continúa con el siguiente."""
    try:
        fn()
        return True
    except Exception as e:
        print(f"   ✗ FALLO en '{descripcion}': {type(e).__name__}: {e}")
        return False


def parte2():
    titulo("PARTE 2 — Pipeline del clasificador de minerales")

    subpaso("[1/10] Verificando dataset")
    if not dataset_ok():
        print("   Dataset no encontrado. Descargando desde HuggingFace...")
        py(["src/extraer_imagenes.py"], "Extrayendo dataset (tedqc/mineral-dataset)")
    if not dataset_ok():
        print("   ✗ Sin dataset no se puede correr el pipeline de Parte 2.")
        print("     Ejecuta manualmente:  python src/extraer_imagenes.py")
        print("     (La app web y predecir.py sí funcionan: usan model/modelo_minerales.pkl)")
        return
    for s in ("train", "val", "test"):
        for c in CLASES:
            print(f"   {s:>5}/{c:<10}: {contar(s, c)} imgs")

    subpaso("[2/10] Augmentation del train set")
    if contar("train", "cuarzo") < 80:
        py(["src/aumentar_datos.py"], "Aumentando datos")
    else:
        print(f"   ✓ Train ya tiene {contar('train','cuarzo')} imgs/clase")

    subpaso("[3/10] Extrayendo características (1835 dims)")
    if not intentar(extraer_features, "extracción de características"):
        return

    subpaso("[4/10] Comparación de 4 clasificadores (test)")
    intentar(comparacion_4_modelos, "comparación de clasificadores")

    for i, (script, desc) in enumerate([
        ("src/experimentos.py",        "Comparación de descriptores E1..E5"),
        ("src/validacion_cruzada.py",  "Validación cruzada 5-fold"),
        ("src/gridsearch.py",          "GridSearchCV (RF + MLP)"),
        ("src/ensemble.py",            "Voting ensembles"),
        ("src/galeria_errores.py",     "Galería de falsos positivos/negativos"),
        ("src/guardar_modelo.py",      "Figuras finales + serialización del ensemble"),
    ], start=5):
        py([script], f"[{i}/10] {desc}")

    subpaso("Tabla resumen final (E3 Color+GLCM, test set)")
    intentar(resumen_final, "tabla resumen final")
    titulo("✓ PIPELINE COMPLETADO — ver results/ y model/modelo_minerales.pkl")


# -------------------------------------------------------------------- App ----
def app():
    titulo("Interfaz web — Streamlit (Ctrl+C para detener)")
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], cwd=ROOT, env=env)


def main():
    ap = argparse.ArgumentParser(description="Ejecuta toda la Tarea 2 de una vez.")
    ap.add_argument("--solo-parte1", action="store_true", help="solo taller de filtros")
    ap.add_argument("--solo-parte2", action="store_true", help="solo pipeline del clasificador")
    ap.add_argument("--solo-app", action="store_true", help="solo la interfaz web")
    ap.add_argument("--app", action="store_true", help="al final, abrir la interfaz web")
    args = ap.parse_args()

    if args.solo_app:
        app(); return
    if args.solo_parte1:
        parte1()
    elif args.solo_parte2:
        parte2()
    else:
        parte1()
        parte2()

    if args.app:
        app()
    else:
        print("\nPara la interfaz web:  python -m streamlit run app.py")


if __name__ == "__main__":
    main()
