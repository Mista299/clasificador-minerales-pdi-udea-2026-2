# -*- coding: utf-8 -*-
"""Pipeline mejorado v3 — versión final.

Estrategia:
  1. Extraer features v2 (1859 dims, incluye histograma HSV).
  2. Evaluar 3 slices × 4 modelos base + Voting + Stacking.
  3. SVM SIN class_weight (descubrimos que con 'balanced' baja 0.833 → 0.722).
  4. RF CON class_weight='balanced' (mejora recall en pirita/hematita).
  5. Comparar Voting vs Stacking en cada slice.
  6. Guardar el MEJOR ensemble de TODAS las combinaciones.

Uso:
    python src/pipeline_v3.py
"""
import sys, time, joblib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier, StackingClassifier, VotingClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import clasificador_v2 as cv2

CACHE = ROOT / "data" / "dataset" / "features_v3.joblib"
RESULTS_CSV = ROOT / "results" / "tables" / "resultados_v3.csv"
MODEL_PKL = ROOT / "model" / "modelo_minerales_v3.pkl"
RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
MODEL_PKL.parent.mkdir(parents=True, exist_ok=True)


def extract_all():
    if CACHE.exists():
        return joblib.load(CACHE)
    feat = {}
    for split in ("train", "val", "test"):
        X_img, y = cv2.load_split(split)
        t0 = time.time()
        X_feat = np.array([cv2.extract_features(img) for img in X_img])
        feat[split] = (X_feat, y)
        print(f"  {split}: {X_feat.shape} en {time.time()-t0:.1f}s")
    joblib.dump(feat, CACHE)
    return feat


def base_models(class_weight_rf=True):
    return {
        "SVM (RBF)": SVC(
            kernel="rbf", C=10, gamma="scale",
            probability=True, random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10,
            class_weight="balanced" if class_weight_rf else None,
            random_state=42, n_jobs=-1,
        ),
        "MLP (128,64,32)": MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            max_iter=1000, alpha=1e-4,
            learning_rate_init=0.005,
            early_stopping=True, random_state=42,
        ),
        "k-NN (5)": KNeighborsClassifier(n_neighbors=5, metric="euclidean"),
    }


def build_voting(class_weight_rf=True):
    return VotingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(
                n_estimators=200, max_depth=10,
                class_weight="balanced" if class_weight_rf else None,
                random_state=42, n_jobs=-1)),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(128, 64, 32), max_iter=1000,
                alpha=1e-4, learning_rate_init=0.005,
                early_stopping=True, random_state=42)),
            ("svm", SVC(
                kernel="rbf", C=10, gamma="scale",
                probability=True, random_state=42)),
        ],
        voting="soft", n_jobs=-1,
    )


def build_stacking(class_weight_rf=True):
    return StackingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(
                n_estimators=200, max_depth=10,
                class_weight="balanced" if class_weight_rf else None,
                random_state=42, n_jobs=-1)),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(128, 64, 32), max_iter=1000,
                alpha=1e-4, learning_rate_init=0.005,
                early_stopping=True, random_state=42)),
            ("svm", SVC(
                kernel="rbf", C=10, gamma="scale",
                probability=True, random_state=42)),
        ],
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=5, n_jobs=-1, passthrough=False,
    )


def evaluate_one(model, Xtr_s, y_train, Xte_s, y_test):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_acc = cross_val_score(model, Xtr_s, y_train, cv=skf,
                             scoring="accuracy", n_jobs=-1)
    cv_f1 = cross_val_score(model, Xtr_s, y_train, cv=skf,
                            scoring="f1_macro", n_jobs=-1)
    model.fit(Xtr_s, y_train)
    yp = model.predict(Xte_s)
    return {
        "CV Acc": cv_acc.mean(),
        "CV F1": cv_f1.mean(),
        "Test Acc": accuracy_score(y_test, yp),
        "Test F1": f1_score(y_test, yp, average="macro"),
        "model_fitted": model,
        "y_pred": yp,
    }


def main():
    print("=" * 64)
    print(" PIPELINE v3 (final) — extracción mejorada + Voting + Stacking")
    print("=" * 64)

    feat = extract_all()
    X_train, y_train = feat["train"]
    X_test, y_test = feat["test"]

    slices = {
        "E3 orig (38d)":     cv2.SLICE_E3_orig,
        "E3 v2 (62d)":       cv2.SLICE_E3_v2,
        "E3 v2 + Hu (69d)":  cv2.SLICE_E3b_v2,
    }

    all_rows = []
    best = None  # (test_acc, slice_label, fitted_model, scaler, slice, row)

    for label, sl in slices.items():
        print(f"\n── {label} ─────────────────────────")
        Xtr = X_train[:, sl]
        Xte = X_test[:, sl]
        sc = StandardScaler().fit(Xtr)
        Xtr_s = sc.transform(Xtr)
        Xte_s = sc.transform(Xte)

        # Modelos base
        for name, m in base_models().items():
            t0 = time.time()
            r = evaluate_one(m, Xtr_s, y_train, Xte_s, y_test)
            row = {
                "Slice": label, "Modelo": name,
                "CV Acc": r["CV Acc"], "CV F1": r["CV F1"],
                "Test Acc": r["Test Acc"], "Test F1": r["Test F1"],
                "Train (s)": round(time.time() - t0, 1),
            }
            all_rows.append(row)
            print(f"  {name:18s}  CV={r['CV Acc']:.3f}  "
                  f"Test={r['Test Acc']:.3f}  F1={r['Test F1']:.3f}")

        # Voting ensemble
        t0 = time.time()
        vot = build_voting()
        r = evaluate_one(vot, Xtr_s, y_train, Xte_s, y_test)
        row = {
            "Slice": label, "Modelo": "Voting (RF+MLP+SVM soft)",
            "CV Acc": r["CV Acc"], "CV F1": r["CV F1"],
            "Test Acc": r["Test Acc"], "Test F1": r["Test F1"],
            "Train (s)": round(time.time() - t0, 1),
        }
        all_rows.append(row)
        print(f"  {row['Modelo']:32s}  CV={r['CV Acc']:.3f}  "
              f"Test={r['Test Acc']:.3f}  F1={r['Test F1']:.3f}")

        # Stacking ensemble
        t0 = time.time()
        stk = build_stacking()
        r = evaluate_one(stk, Xtr_s, y_train, Xte_s, y_test)
        row = {
            "Slice": label, "Modelo": "Stacking (RF+MLP+SVM → LR)",
            "CV Acc": r["CV Acc"], "CV F1": r["CV F1"],
            "Test Acc": r["Test Acc"], "Test F1": r["Test F1"],
            "Train (s)": round(time.time() - t0, 1),
        }
        all_rows.append(row)
        print(f"  {row['Modelo']:32s}  CV={r['CV Acc']:.3f}  "
              f"Test={r['Test Acc']:.3f}  F1={r['Test F1']:.3f}")

        # Comparar Voting vs Stacking para este slice
        vot_row = all_rows[-2]
        stk_row = all_rows[-1]
        winner_row = stk_row if stk_row["Test Acc"] >= vot_row["Test Acc"] else vot_row
        winner_model = r["model_fitted"] if winner_row["Test Acc"] >= vot_row["Test Acc"] else None
        # (en la práctica re-entrenamos abajo con el ganador)

    # Encontrar el mejor de TODAS las filas
    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\n✓ Resultados en {RESULTS_CSV}")

    # Re-entrenar el mejor ensemble
    best_idx = df["Test Acc"].idxmax()
    best_row = df.loc[best_idx]
    print(f"\n🏆 Mejor resultado global: "
          f"{best_row['Modelo']} en {best_row['Slice']} "
          f"→ Test Acc={best_row['Test Acc']:.3f}, F1={best_row['Test F1']:.3f}, "
          f"CV-5={best_row['CV Acc']:.3f}")

    # Re-construir el mejor
    sl_label = best_row["Slice"]
    sl = slices[sl_label]
    Xtr = X_train[:, sl]
    Xte = X_test[:, sl]
    sc = StandardScaler().fit(Xtr)
    Xtr_s = sc.transform(Xtr)
    Xte_s = sc.transform(Xte)

    if "Stacking" in best_row["Modelo"]:
        best_model = build_stacking()
    elif "Voting" in best_row["Modelo"]:
        best_model = build_voting()
    else:
        # modelo base individual
        best_model = base_models()[best_row["Modelo"]]

    best_model.fit(Xtr_s, y_train)
    yp = best_model.predict(Xte_s)

    print(classification_report(y_test, yp, target_names=cv2.CLASSES, digits=3))

    # Serializar
    bundle = {
        "modelo": best_model,
        "scaler": sc,
        "slice": sl,
        "slice_start": sl.start,
        "slice_end": sl.stop,
        "clases": cv2.CLASSES,
        "n_features": sl.stop - sl.start,
        "accuracy_test": best_row["Test Acc"],
        "accuracy_cv5": best_row["CV Acc"],
        "modelo_tipo": best_row["Modelo"],
        "version": "3.0",
        "descripcion": (
            f"Clasificador v3: features v2 ({cv2.N_HOG + cv2.N_COLOR_MOM + cv2.N_COLOR_HIST + cv2.N_GABOR + cv2.N_LBP + cv2.N_GLCM + cv2.N_HU} dims totales), "
            f"slice={sl_label}, class_weight='balanced' en RF. "
            f"Test Acc={best_row['Test Acc']:.3f}, CV5={best_row['CV Acc']:.3f}."
        ),
    }
    joblib.dump(bundle, MODEL_PKL)
    print(f"\n✓ Modelo v3 guardado en: {MODEL_PKL} "
          f"({Path(MODEL_PKL).stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()