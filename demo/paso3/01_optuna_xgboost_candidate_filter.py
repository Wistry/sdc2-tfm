from pathlib import Path
import json
import os

BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUTS_DIR / ".matplotlib"))

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    print("Optuna no esta instalado. Instala con: pip install optuna")
    raise SystemExit(0)

try:
    from xgboost import XGBClassifier
except ImportError:
    print("XGBoost no esta instalado. Instala con: pip install xgboost")
    raise SystemExit(0)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


FEATURES_PATH = BASE_DIR / "../paso2/outputs/candidates_features.csv"
EXPERIMENT_MODE = "no_position"
N_TRIALS = 100
RANDOM_STATE = 42

LEAKAGE_COLUMNS = {
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
    "x_peak",
    "y_peak",
    "z_peak",
    "ra",
    "dec",
    "freq",
    "ra_peak",
    "dec_peak",
    "freq_peak",
}


def prepare_dataset() -> tuple[pd.DataFrame, pd.Series, dict[str, int]]:
    df = pd.read_csv(FEATURES_PATH).replace([np.inf, -np.inf], np.nan)
    if "clean_label" not in df.columns:
        raise ValueError("Falta clean_label. Ejecuta primero demo/paso2/01_extract_spatio_spectral_features.py")

    total_rows = len(df)
    ambiguous_count = int((df["clean_label"] == -1).sum())
    df = df[df["clean_label"] != -1].copy()
    y = df["clean_label"].astype(int)

    exclude = set(LEAKAGE_COLUMNS)
    if EXPERIMENT_MODE == "no_position":
        exclude.update(POSITION_COLUMNS)
    elif EXPERIMENT_MODE != "with_position":
        raise ValueError(f"EXPERIMENT_MODE invalido: {EXPERIMENT_MODE}")

    feature_df = df.drop(columns=[col for col in exclude if col in df.columns])
    X = feature_df.select_dtypes(include=[np.number])
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)

    counts = {
        "total_rows": total_rows,
        "used_rows": len(df),
        "clean_tp": int((y == 1).sum()),
        "clean_fp": int((y == 0).sum()),
        "ambiguous_discarded": ambiguous_count,
    }
    return X, y, counts


def build_model(params: dict, scale_pos_weight: float) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=4,
        scale_pos_weight=scale_pos_weight,
        **params,
    )


def metrics_from_scores(y_true: pd.Series, y_score: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = y_score >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "false_positives_accepted": int(fp),
    }


def save_feature_importance(model: XGBClassifier, feature_names: pd.Index) -> None:
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    top = importance_df.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["feature"], top["importance"], color="0.25")
    ax.set_xlabel("importance")
    ax.set_title("Optuna XGBoost feature importance")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "optuna_feature_importance.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_optuna_plots(trials_df: pd.DataFrame) -> None:
    trials_df = trials_df.sort_values("number")
    best_so_far = trials_df["value"].cummax()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(trials_df["number"], best_so_far, marker="o", markersize=3)
    ax.set_xlabel("trial")
    ax.set_ylabel("best objective")
    ax.set_title("Optuna optimization history")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "optuna_optimization_history.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(trials_df["user_attrs_threshold"], trials_df["user_attrs_f1"], s=25, alpha=0.8)
    ax.set_xlabel("threshold")
    ax.set_ylabel("F1")
    ax.set_title("Threshold vs F1")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "optuna_threshold_vs_f1.png", dpi=160)
    plt.close(fig)


def main() -> None:
    X, y, counts = prepare_dataset()
    print(f"Filas totales: {counts['total_rows']}")
    print(f"Filas usadas: {counts['used_rows']}")
    print(f"TP limpios: {counts['clean_tp']}")
    print(f"FP limpios: {counts['clean_fp']}")
    print(f"Ambiguos descartados: {counts['ambiguous_discarded']}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = n_neg / n_pos if n_pos else 1.0

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.30, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        threshold = trial.suggest_float("threshold", 0.05, 0.80)
        model = build_model(params, scale_pos_weight)
        model.fit(X_train, y_train)
        y_score = model.predict_proba(X_test)[:, 1]
        metric_values = metrics_from_scores(y_test, y_score, threshold)
        for key, value in metric_values.items():
            trial.set_user_attr(key, float(value))
        trial.set_user_attr("threshold", float(threshold))

        if metric_values["precision"] < 0.75:
            return 0.0
        if metric_values["recall"] < 0.50:
            return 0.0
        return metric_values["f1"]

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=N_TRIALS)

    best_trial = study.best_trial
    best_params = dict(best_trial.params)
    best_threshold = float(best_params.pop("threshold"))
    best_model = build_model(best_params, scale_pos_weight)
    best_model.fit(X_train, y_train)
    y_score = best_model.predict_proba(X_test)[:, 1]
    y_pred = y_score >= best_threshold
    final_metrics = metrics_from_scores(y_test, y_score, best_threshold)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    report = classification_report(y_test, y_pred, zero_division=0)

    best_payload = {
        "experiment_mode": EXPERIMENT_MODE,
        "threshold": best_threshold,
        "params": best_params,
        "metrics": final_metrics,
    }
    (OUTPUTS_DIR / "optuna_best_params.json").write_text(json.dumps(best_payload, indent=2), encoding="utf-8")

    trials_df = study.trials_dataframe()
    trials_df.to_csv(OUTPUTS_DIR / "optuna_trials.csv", index=False)
    trials_df.to_csv(OUTPUTS_DIR / "optuna_trials_strict.csv", index=False)
    save_optuna_plots(trials_df)
    save_feature_importance(best_model, X.columns)

    metrics_text = (
        "Optuna XGBoost candidate filter\n"
        f"Modo usado: {EXPERIMENT_MODE}\n"
        f"Numero de features: {X.shape[1]}\n"
        f"Filas usadas: {counts['used_rows']}\n"
        f"TP limpios: {counts['clean_tp']}\n"
        f"FP limpios: {counts['clean_fp']}\n"
        f"Mejor F1: {final_metrics['f1']:.4f}\n"
        f"Precision: {final_metrics['precision']:.4f}\n"
        f"Recall: {final_metrics['recall']:.4f}\n"
        f"False positives aceptados: {final_metrics['false_positives_accepted']}\n"
        f"ROC AUC: {final_metrics['roc_auc']:.4f}\n"
        f"PR AUC: {final_metrics['pr_auc']:.4f}\n"
        f"Threshold optimo: {best_threshold:.4f}\n"
        f"Mejores hiperparametros:\n{json.dumps(best_params, indent=2)}\n"
        f"Confusion matrix:\n{cm}\n"
        f"Classification report:\n{report}\n"
    )
    (OUTPUTS_DIR / "optuna_metrics.txt").write_text(metrics_text, encoding="utf-8")
    (OUTPUTS_DIR / "optuna_metrics_strict.txt").write_text(metrics_text, encoding="utf-8")

    print("Mejor trial:")
    print(f"  number: {best_trial.number}")
    print(f"  objective: {best_trial.value:.4f}")
    print("Mejores hiperparametros:")
    print(json.dumps(best_params, indent=2))
    print(f"Precision: {final_metrics['precision']:.4f}")
    print(f"Recall: {final_metrics['recall']:.4f}")
    print(f"F1: {final_metrics['f1']:.4f}")
    print(f"False positives aceptados: {final_metrics['false_positives_accepted']}")
    print(f"ROC AUC: {final_metrics['roc_auc']:.4f}")
    print(f"PR AUC: {final_metrics['pr_auc']:.4f}")
    print(f"Threshold: {best_threshold:.4f}")
    print(f"Metricas guardadas en: {OUTPUTS_DIR / 'optuna_metrics_strict.txt'}")


if __name__ == "__main__":
    main()
