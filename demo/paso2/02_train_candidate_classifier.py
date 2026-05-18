from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUTS_DIR / ".matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


FEATURES_PATH = OUTPUTS_DIR / "candidates_features.csv"
METRICS_PATH = OUTPUTS_DIR / "metrics.txt"
MODEL_COMPARISON_TXT = OUTPUTS_DIR / "model_comparison_metrics.txt"
MODEL_COMPARISON_CLEAN_TXT = OUTPUTS_DIR / "model_comparison_clean_labels.txt"
THRESHOLD_SWEEP_ALL_CSV = OUTPUTS_DIR / "threshold_sweep_all.csv"
THRESHOLDS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]

ALWAYS_EXCLUDE = {
    "label",
    "clean_label",
    "is_ambiguous",
    "matched_truth_id",
    "truth_row",
    "truth_x",
    "truth_y",
    "truth_z",
    "min_abs_dx",
    "min_abs_dy",
    "min_abs_dz",
    "min_dist_3d",
    "matching_mode",
    "name",
    "id",
}
POSITION_COLUMNS = {
    "x",
    "y",
    "z",
    "ra",
    "dec",
    "freq",
    "x_peak",
    "y_peak",
    "z_peak",
    "ra_peak",
    "dec_peak",
    "freq_peak",
}
SPATIO_SPECTRAL_HINTS = ("flux_", "peak_", "active_", "area_", "centroid_", "valid_centroid")


def prepare_X(df: pd.DataFrame, without_position: bool) -> pd.DataFrame:
    exclude = set(ALWAYS_EXCLUDE)
    if without_position:
        exclude.update(POSITION_COLUMNS)
    feature_df = df.drop(columns=[col for col in exclude if col in df.columns])
    X = feature_df.select_dtypes(include=[np.number]).drop(columns=["label", "clean_label"], errors="ignore")
    return X.fillna(X.median(numeric_only=True)).fillna(0.0)


def threshold_sweep(y_test: pd.Series, y_score: np.ndarray, experiment: str, model_name: str) -> pd.DataFrame:
    rows = []
    for threshold in THRESHOLDS:
        y_pred = y_score >= threshold
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        rows.append({
            "experiment": experiment,
            "model": model_name,
            "threshold": threshold,
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "tp_accepted": tp,
            "fp_accepted": fp,
        })
    return pd.DataFrame(rows)


def save_importance_png(features: pd.Index, importances: np.ndarray, path: Path, title: str) -> pd.DataFrame:
    importance_df = pd.DataFrame({"feature": features, "importance": importances}).sort_values("importance", ascending=False)
    top = importance_df.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["feature"], top["importance"], color="0.25")
    ax.set_xlabel("importance")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return importance_df


def evaluate_model(model, model_name: str, experiment: str, X: pd.DataFrame, y: pd.Series, importance_path: Path) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        stratify=y,
        random_state=42,
    )
    model.fit(X_train, y_train)
    y_score = model.predict_proba(X_test)[:, 1]
    y_pred = y_score >= 0.50
    sweep_df = threshold_sweep(y_test, y_score, experiment, model_name)

    roc_auc = roc_auc_score(y_test, y_score) if len(np.unique(y_test)) == 2 else np.nan
    pr_auc = average_precision_score(y_test, y_score) if len(np.unique(y_test)) == 2 else np.nan
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    report = classification_report(y_test, y_pred, zero_division=0)
    best_f1 = sweep_df.sort_values(["f1", "precision", "recall"], ascending=False).iloc[0]
    high_precision = sweep_df[sweep_df["precision"] >= 0.8]
    best_precision = None
    if not high_precision.empty:
        best_precision = high_precision.sort_values(["f1", "recall"], ascending=False).iloc[0]

    importances = getattr(model, "feature_importances_", np.zeros(X.shape[1]))
    importance_df = save_importance_png(X.columns, importances, importance_path, f"{model_name} - {experiment}")
    spatio_signal = importance_df[importance_df["feature"].str.startswith(SPATIO_SPECTRAL_HINTS)]["importance"].sum()

    text = (
        f"\n## {experiment} / {model_name}\n"
        f"Features usadas: {X.shape[1]}\n"
        f"ROC AUC: {roc_auc:.4f}\n"
        f"PR AUC: {pr_auc:.4f}\n"
        f"Precision@0.50: {precision_score(y_test, y_pred, zero_division=0):.4f}\n"
        f"Recall@0.50: {recall_score(y_test, y_pred, zero_division=0):.4f}\n"
        f"F1@0.50: {f1_score(y_test, y_pred, zero_division=0):.4f}\n"
        f"Mejor threshold por F1: {best_f1['threshold']:.2f} "
        f"(precision={best_f1['precision']:.4f}, recall={best_f1['recall']:.4f}, f1={best_f1['f1']:.4f})\n"
        + (
            f"Mejor threshold con precision >= 0.8: {best_precision['threshold']:.2f} "
            f"(precision={best_precision['precision']:.4f}, recall={best_precision['recall']:.4f}, f1={best_precision['f1']:.4f})\n"
            if best_precision is not None
            else "Mejor threshold con precision >= 0.8: no existe en el sweep\n"
        )
        + f"Importancia total features espacio-espectrales: {spatio_signal:.4f}\n"
        f"Confusion matrix @0.50:\n{cm}\n"
        f"Classification report @0.50:\n{report}\n"
        f"Top features:\n{importance_df.head(10).to_string(index=False)}\n"
    )
    return text, sweep_df, importance_df


def matching_counts(df: pd.DataFrame) -> str:
    lines = []
    for name, limits in {
        "strict": (10, 10, 25),
        "medium": (20, 20, 50),
        "loose": (30, 30, 75),
    }.items():
        if {"min_abs_dx", "min_abs_dy", "min_abs_dz"}.issubset(df.columns):
            dx, dy, dz = limits
            matched = (df["min_abs_dx"] <= dx) & (df["min_abs_dy"] <= dy) & (df["min_abs_dz"] <= dz)
            lines.append(f"{name}: TP={int(matched.sum())}, FP={int((~matched).sum())}")
    return "\n".join(lines)


def automatic_conclusions(df: pd.DataFrame, metrics: list[dict[str, object]], sweep_all: pd.DataFrame) -> str:
    with_pos = next((m for m in metrics if m["experiment"] == "with_position" and m["model"] == "RandomForest"), None)
    no_pos = next((m for m in metrics if m["experiment"] == "no_position" and m["model"] == "RandomForest"), None)
    conclusions = ["\n## Conclusiones automaticas"]
    conclusions.append(f"TP/FP limpios usados: TP={int((df['clean_label'] == 1).sum())}, FP={int((df['clean_label'] == 0).sum())}.")
    if {"min_abs_dx", "min_abs_dy", "min_abs_dz"}.issubset(df.columns):
        strict = ((df["min_abs_dx"] <= 10) & (df["min_abs_dy"] <= 10) & (df["min_abs_dz"] <= 25)).sum()
        medium = ((df["min_abs_dx"] <= 20) & (df["min_abs_dy"] <= 20) & (df["min_abs_dz"] <= 50)).sum()
        if strict < 0.6 * medium:
            conclusions.append("El matching estricto parece bastante duro: pierde muchos candidatos frente al criterio medio.")
        else:
            conclusions.append("El matching estricto no cambia drasticamente frente al criterio medio.")
    if with_pos and no_pos:
        delta = float(with_pos["pr_auc"]) - float(no_pos["pr_auc"])
        if abs(delta) > 0.05:
            conclusions.append(f"El modelo depende de la posicion: diferencia PR AUC con-posicion menos sin-posicion = {delta:.4f}.")
        else:
            conclusions.append("La dependencia de posicion parece baja segun PR AUC.")
    best = sweep_all.sort_values(["f1", "precision"], ascending=False).iloc[0]
    conclusions.append(f"Threshold razonable para filtrar FP segun F1: {best['threshold']:.2f} en {best['experiment']} / {best['model']}.")
    conclusions.append("Si ROC/PR AUC son bajos sin posicion, las features espacio-espectrales actuales aun aportan poca senal separadora.")
    return "\n".join(conclusions) + "\n"


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(FEATURES_PATH).replace([np.inf, -np.inf], np.nan)
    if "clean_label" not in df.columns:
        raise ValueError("Falta clean_label. Ejecuta primero 01_extract_spatio_spectral_features.py")

    total_rows = len(df)
    ambiguous_count = int((df["clean_label"] == -1).sum())
    train_df = df[df["clean_label"] != -1].copy()
    y = train_df["clean_label"].astype(int)
    tp_count = int((y == 1).sum())
    fp_count = int((y == 0).sum())

    if tp_count < 5 or fp_count < 5:
        message = (
            "No hay suficientes ejemplos TP/FP para entrenar; ampliar region SoFiA o bajar threshold.\n"
            f"Total filas: {total_rows}\n"
            f"Filas usadas para entrenamiento: {len(train_df)}\n"
            f"TP limpios: {tp_count}\n"
            f"FP limpios: {fp_count}\n"
            f"Ambiguos descartados: {ambiguous_count}\n"
        )
        MODEL_COMPARISON_CLEAN_TXT.write_text(message, encoding="utf-8")
        METRICS_PATH.write_text(message, encoding="utf-8")
        print(message)
        return

    all_text = [
        "Comparacion de modelos con clean_label sin data leakage",
        f"Total filas: {total_rows}",
        f"Filas usadas para entrenamiento: {len(train_df)}",
        f"TP limpios: {tp_count}",
        f"FP limpios: {fp_count}",
        f"Ambiguos descartados: {ambiguous_count}",
        f"Matching mode del CSV: {train_df['matching_mode'].iloc[0] if 'matching_mode' in train_df.columns else 'desconocido'}",
        "",
        "Conteos por matching diagnostico:",
        matching_counts(df),
    ]
    all_sweeps = []
    metric_rows = []

    experiments = {
        "with_position": prepare_X(train_df, without_position=False),
        "no_position": prepare_X(train_df, without_position=True),
    }
    for experiment, X in experiments.items():
        models = {
            "RandomForest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42),
        }
        if XGBClassifier is not None:
            scale_pos_weight = fp_count / tp_count if tp_count else 1.0
            models["XGBoost"] = XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight,
                random_state=42,
            )

        for model_name, model in models.items():
            stem = "rf" if model_name == "RandomForest" else "xgb"
            importance_path = OUTPUTS_DIR / f"feature_importance_{stem}_{experiment}.png"
            text, sweep_df, _ = evaluate_model(model, model_name, experiment, X, y, importance_path)
            all_text.append(text)
            all_sweeps.append(sweep_df)
            best_f1 = sweep_df.sort_values(["f1", "precision"], ascending=False).iloc[0]
            metric_rows.append({
                "experiment": experiment,
                "model": model_name,
                "pr_auc": float(text.split("PR AUC: ")[1].split("\n")[0]),
                "best_threshold": float(best_f1["threshold"]),
                "best_f1": float(best_f1["f1"]),
            })

    sweep_all = pd.concat(all_sweeps, ignore_index=True)
    sweep_all.to_csv(THRESHOLD_SWEEP_ALL_CSV, index=False)
    all_text.append(automatic_conclusions(train_df, metric_rows, sweep_all))
    final_text = "\n".join(all_text)
    MODEL_COMPARISON_CLEAN_TXT.write_text(final_text, encoding="utf-8")
    METRICS_PATH.write_text(final_text, encoding="utf-8")
    print(final_text)
    print(f"Metricas guardadas en: {MODEL_COMPARISON_CLEAN_TXT}")
    print(f"Sweep guardado en: {THRESHOLD_SWEEP_ALL_CSV}")


if __name__ == "__main__":
    main()
