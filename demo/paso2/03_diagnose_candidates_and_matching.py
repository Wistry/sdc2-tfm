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
from astropy.io import fits
from astropy.wcs import WCS


FITS_PATH = BASE_DIR / "../../data/sky_dev_v2.fits"
TRUTH_PATH = BASE_DIR / "../../data/sky_dev_truthcat_v2.txt"
SOFIA_MANY_CANDIDATES_PATH = BASE_DIR / "../../results/master_test_many_candidates_cat.txt"
SOFIA_MEDIUM_PATH = BASE_DIR / "../../results/master_test_medium_cat.txt"
SOFIA_SMALL_PATH = BASE_DIR / "../../results/master_test_small_cat.txt"
DIAGNOSTICS_CSV = OUTPUTS_DIR / "matching_diagnostics.csv"
SUMMARY_TXT = OUTPUTS_DIR / "matching_summary.txt"

SOFIA_COLUMNS = [
    "name", "id", "x", "y", "z", "x_min", "x_max", "y_min", "y_max",
    "z_min", "z_max", "n_pix", "f_min", "f_max", "f_sum", "rel", "flag",
    "rms", "w20", "w50", "wm50", "z_w20", "z_w50", "z_wm50", "ell_maj",
    "ell_min", "ell_pa", "ell3s_maj", "ell3s_min", "ell3s_pa", "kin_pa",
    "err_x", "err_y", "err_z", "err_f_sum", "snr", "snr_max", "ra", "dec",
    "freq", "x_peak", "y_peak", "z_peak", "ra_peak", "dec_peak", "freq_peak",
]

CRITERIA = {
    "strict_box_dx10_dy10_dz25": ("box", 10, 10, 25),
    "medium_box_dx20_dy20_dz50": ("box", 20, 20, 50),
    "loose_box_dx30_dy30_dz75": ("box", 30, 30, 75),
    "dist3d_le30": ("dist", 30),
    "dist3d_le50": ("dist", 50),
    "dist3d_le75": ("dist", 75),
}


def choose_sofia_catalog() -> Path:
    if SOFIA_MANY_CANDIDATES_PATH.exists():
        return SOFIA_MANY_CANDIDATES_PATH
    if SOFIA_MEDIUM_PATH.exists():
        return SOFIA_MEDIUM_PATH
    return SOFIA_SMALL_PATH


def read_sofia_catalog(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", comment="#", names=SOFIA_COLUMNS, engine="python")
    for col in SOFIA_COLUMNS:
        if col != "name":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["x", "y", "z"]).reset_index(drop=True)


def read_truth_catalog(path: Path) -> pd.DataFrame:
    truth_df = pd.read_csv(path, sep=r"\s+", comment="#", engine="python")
    required = {"ra", "dec", "central_freq"}
    missing = required.difference(truth_df.columns)
    if missing:
        raise ValueError(f"Truth catalogue sin columnas esperadas: {sorted(missing)}")
    return truth_df


def add_truth_pixels(truth_df: pd.DataFrame, wcs: WCS) -> pd.DataFrame:
    truth_df = truth_df.copy()
    truth_x, truth_y, truth_z = wcs.world_to_pixel_values(
        truth_df["ra"].to_numpy(),
        truth_df["dec"].to_numpy(),
        truth_df["central_freq"].to_numpy(),
    )
    truth_df["truth_x"] = truth_x
    truth_df["truth_y"] = truth_y
    truth_df["truth_z"] = truth_z
    return truth_df


def nearest_truth_rows(sofia_df: pd.DataFrame, truth_df: pd.DataFrame) -> pd.DataFrame:
    truth_x = truth_df["truth_x"].to_numpy()
    truth_y = truth_df["truth_y"].to_numpy()
    truth_z = truth_df["truth_z"].to_numpy()
    truth_ids = truth_df["id"].to_numpy() if "id" in truth_df.columns else truth_df.index.to_numpy()
    rows = []

    keep_cols = ["name", "id", "x", "y", "z", "ra", "dec", "freq", "snr", "snr_max", "f_sum", "n_pix", "w20", "ell_maj", "ell_min"]
    for _, candidate in sofia_df.iterrows():
        dx_signed = float(candidate["x"]) - truth_x
        dy_signed = float(candidate["y"]) - truth_y
        dz_signed = float(candidate["z"]) - truth_z
        dist_3d = np.sqrt(dx_signed * dx_signed + dy_signed * dy_signed + dz_signed * dz_signed)
        nearest = int(np.nanargmin(dist_3d))
        row = {col: candidate.get(col, np.nan) for col in keep_cols}
        row.update({
            "nearest_truth_id": truth_ids[nearest],
            "nearest_truth_row": truth_df.index[nearest],
            "dx": dx_signed[nearest],
            "dy": dy_signed[nearest],
            "dz": dz_signed[nearest],
            "abs_dx": abs(dx_signed[nearest]),
            "abs_dy": abs(dy_signed[nearest]),
            "abs_dz": abs(dz_signed[nearest]),
            "min_dist_3d": dist_3d[nearest],
        })
        rows.append(row)
    return pd.DataFrame(rows)


def apply_criteria(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    summary_lines = []
    n_candidates = len(df)
    for name, criterion in CRITERIA.items():
        if criterion[0] == "box":
            _, max_dx, max_dy, max_dz = criterion
            matched = (df["abs_dx"] <= max_dx) & (df["abs_dy"] <= max_dy) & (df["abs_dz"] <= max_dz)
        else:
            _, max_dist = criterion
            matched = df["min_dist_3d"] <= max_dist
        df[name] = matched.astype(int)
        tp = int(matched.sum())
        fp = int(n_candidates - tp)
        pct = 100.0 * tp / n_candidates if n_candidates else 0.0
        summary_lines.append(f"{name}: TP={tp}, FP={fp}, TP%={pct:.2f}")
    return df, summary_lines


def save_plots(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["min_dist_3d"], bins=50, color="0.25")
    ax.set_xlabel("min_dist_3d")
    ax.set_ylabel("candidatos")
    ax.set_title("Distancia minima al truth")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "min_dist_3d_hist.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df["snr"], df["min_dist_3d"], s=10, alpha=0.5)
    ax.set_xlabel("snr")
    ax.set_ylabel("min_dist_3d")
    ax.set_title("SNR vs distancia al truth")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "snr_vs_dist.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df["f_sum"], df["min_dist_3d"], s=10, alpha=0.5)
    ax.set_xlabel("f_sum")
    ax.set_ylabel("min_dist_3d")
    ax.set_title("Flujo vs distancia al truth")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "flux_vs_dist.png", dpi=160)
    plt.close(fig)

    medium_tp = df["medium_box_dx20_dy20_dz50"] == 1
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(
        [df.loc[medium_tp, "snr"].dropna(), df.loc[~medium_tp, "snr"].dropna()],
        tick_labels=["TP medio", "FP medio"],
        showfliers=False,
    )
    ax.set_ylabel("snr")
    ax.set_title("SNR por TP/FP con matching medio")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "snr_tp_fp.png", dpi=160)
    plt.close(fig)


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    sofia_path = choose_sofia_catalog()
    sofia_df = read_sofia_catalog(sofia_path)
    truth_df = read_truth_catalog(TRUTH_PATH)

    with fits.open(FITS_PATH, memmap=True) as hdul:
        wcs = WCS(hdul[0].header)
        truth_df = add_truth_pixels(truth_df, wcs)

    diagnostics_df = nearest_truth_rows(sofia_df, truth_df)
    diagnostics_df, summary_lines = apply_criteria(diagnostics_df)
    diagnostics_df.to_csv(DIAGNOSTICS_CSV, index=False)
    save_plots(diagnostics_df)

    text = "\n".join([
        f"Catalogo SoFiA usado: {sofia_path}",
        f"Candidatos: {len(sofia_df)}",
        f"Truth rows: {len(truth_df)}",
        "",
        "Resumen de criterios:",
        *summary_lines,
        "",
        "10 candidatos mas cercanos:",
        diagnostics_df.sort_values("min_dist_3d").head(10).to_string(index=False),
        "",
    ])
    SUMMARY_TXT.write_text(text, encoding="utf-8")
    print(text)
    print(f"Tabla guardada en: {DIAGNOSTICS_CSV}")
    print(f"Resumen guardado en: {SUMMARY_TXT}")


if __name__ == "__main__":
    main()
