from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS

try:
    from scipy import ndimage
except ImportError:
    ndimage = None


BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
FITS_PATH = BASE_DIR / "../../data/sky_dev_v2.fits"
TRUTH_PATH = BASE_DIR / "../../data/sky_dev_truthcat_v2.txt"
SOFIA_MEDIUM_PATH = BASE_DIR / "../../results/master_test_medium_cat.txt"
SOFIA_MANY_CANDIDATES_PATH = BASE_DIR / "../../results/master_test_many_candidates_cat.txt"
SOFIA_SMALL_PATH = BASE_DIR / "../../results/master_test_small_cat.txt"

MAX_Z = 128
MAX_Y = 64
MAX_X = 64
MATCHING_MODE = "medium"
MATCHING_THRESHOLDS = {
    "strict": (10, 10, 25),
    "medium": (20, 20, 50),
    "loose": (30, 30, 75),
}

SOFIA_COLUMNS = [
    "name", "id", "x", "y", "z", "x_min", "x_max", "y_min", "y_max",
    "z_min", "z_max", "n_pix", "f_min", "f_max", "f_sum", "rel", "flag",
    "rms", "w20", "w50", "wm50", "z_w20", "z_w50", "z_wm50", "ell_maj",
    "ell_min", "ell_pa", "ell3s_maj", "ell3s_min", "ell3s_pa", "kin_pa",
    "err_x", "err_y", "err_z", "err_f_sum", "snr", "snr_max", "ra", "dec",
    "freq", "x_peak", "y_peak", "z_peak", "ra_peak", "dec_peak", "freq_peak",
]


def read_sofia_catalog(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", comment="#", names=SOFIA_COLUMNS, engine="python")
    for col in SOFIA_COLUMNS:
        if col != "name":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["x", "y", "z", "x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]).reset_index(drop=True)


def read_truth_catalog(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", comment="#", engine="python")
    required = {"ra", "dec", "central_freq"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Truth catalogue sin columnas esperadas: {sorted(missing)}")
    return df


def choose_sofia_catalog() -> Path:
    if SOFIA_MANY_CANDIDATES_PATH.exists():
        return SOFIA_MANY_CANDIDATES_PATH
    if SOFIA_MEDIUM_PATH.exists():
        return SOFIA_MEDIUM_PATH
    return SOFIA_SMALL_PATH


def clamp_interval(start: int, stop: int, max_size: int) -> tuple[int, int]:
    start = max(0, min(start, max_size))
    stop = max(0, min(stop, max_size))
    if stop <= start:
        stop = min(max_size, start + 1)
    return start, stop


def crop_around_center(start: int, stop: int, center: float, limit: int, max_size: int) -> tuple[int, int]:
    start, stop = clamp_interval(start, stop, max_size)
    if stop - start <= limit:
        return start, stop

    center_i = int(round(center))
    half = limit // 2
    new_start = center_i - half
    new_stop = new_start + limit
    if new_start < 0:
        new_start = 0
        new_stop = limit
    if new_stop > max_size:
        new_stop = max_size
        new_start = max(0, new_stop - limit)
    return new_start, new_stop


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0 or pd.isna(denominator):
        return np.nan
    return numerator / denominator


def centroid_path_length(x_values: np.ndarray, y_values: np.ndarray) -> float:
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    if valid.sum() < 2:
        return 0.0
    points = np.column_stack([x_values[valid], y_values[valid]])
    diffs = np.diff(points, axis=0)
    return float(np.sqrt((diffs * diffs).sum(axis=1)).sum())


def longest_true_run(values: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in values:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def count_local_peaks(values: np.ndarray) -> int:
    if len(values) < 3:
        return 0
    peaks = (values[1:-1] > values[:-2]) & (values[1:-1] > values[2:])
    return int(np.count_nonzero(peaks))


def connected_components_3d(active: np.ndarray) -> tuple[int, int]:
    if not active.any():
        return 0, 0
    if ndimage is not None:
        structure = np.ones((3, 3, 3), dtype=int)
        labeled, n_components = ndimage.label(active, structure=structure)
        if n_components == 0:
            return 0, 0
        sizes = np.bincount(labeled.ravel())[1:]
        return int(n_components), int(sizes.max())

    visited = np.zeros(active.shape, dtype=bool)
    active_points = np.argwhere(active)
    n_components = 0
    largest_size = 0
    z_max, y_max, x_max = active.shape
    offsets = [
        (dz, dy, dx)
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if not (dz == 0 and dy == 0 and dx == 0)
    ]

    for start_z, start_y, start_x in active_points:
        if visited[start_z, start_y, start_x]:
            continue
        n_components += 1
        size = 0
        stack = [(int(start_z), int(start_y), int(start_x))]
        visited[start_z, start_y, start_x] = True
        while stack:
            z, y, x = stack.pop()
            size += 1
            for dz, dy, dx in offsets:
                nz, ny, nx = z + dz, y + dy, x + dx
                if 0 <= nz < z_max and 0 <= ny < y_max and 0 <= nx < x_max and active[nz, ny, nx] and not visited[nz, ny, nx]:
                    visited[nz, ny, nx] = True
                    stack.append((nz, ny, nx))
        largest_size = max(largest_size, size)
    return n_components, largest_size


def spatio_spectral_features(subcube: np.ndarray) -> dict[str, float]:
    subcube = np.asarray(subcube, dtype=np.float64)
    if subcube.size == 0:
        return {}

    threshold = float(np.nanmean(subcube) + 2.0 * np.nanstd(subcube))
    active_3d = subcube > threshold
    active_voxels = int(np.count_nonzero(active_3d))
    n_components_3d, largest_component_size = connected_components_3d(active_3d)
    flux_z = np.nansum(subcube, axis=(1, 2))
    peak_z = np.nanmax(subcube, axis=(1, 2))

    area_values = []
    centroid_x = []
    centroid_y = []

    for slice_2d in subcube:
        active = slice_2d > threshold
        area = int(np.count_nonzero(active))
        area_values.append(area)

        if area == 0:
            centroid_x.append(np.nan)
            centroid_y.append(np.nan)
            continue

        weights = np.where(active, slice_2d, 0.0)
        weights = np.where(weights > 0, weights, 0.0)
        total_weight = float(np.nansum(weights))
        if total_weight <= 0:
            centroid_x.append(np.nan)
            centroid_y.append(np.nan)
            continue

        yy, xx = np.indices(slice_2d.shape)
        centroid_x.append(float(np.nansum(xx * weights) / total_weight))
        centroid_y.append(float(np.nansum(yy * weights) / total_weight))

    area_z = np.asarray(area_values, dtype=np.float64)
    centroid_x = np.asarray(centroid_x, dtype=np.float64)
    centroid_y = np.asarray(centroid_y, dtype=np.float64)
    valid_centroids = np.isfinite(centroid_x) & np.isfinite(centroid_y)
    active_channels = area_z > 0
    valid_points = np.column_stack([centroid_x[valid_centroids], centroid_y[valid_centroids]])
    centroid_steps = np.sqrt((np.diff(valid_points, axis=0) ** 2).sum(axis=1)) if len(valid_points) > 1 else np.asarray([], dtype=np.float64)
    area_mean = float(np.nanmean(area_z))
    flux_mean = float(np.nanmean(flux_z))
    flux_max = float(np.nanmax(flux_z))

    return {
        "flux_mean": flux_mean,
        "flux_std": float(np.nanstd(flux_z)),
        "flux_max": flux_max,
        "flux_min": float(np.nanmin(flux_z)),
        "flux_smoothness": float(np.nanmean(np.abs(np.diff(flux_z)))) if len(flux_z) > 1 else 0.0,
        "flux_num_local_peaks": count_local_peaks(flux_z),
        "flux_peak_to_mean": safe_divide(flux_max, abs(flux_mean)),
        "peak_mean": float(np.nanmean(peak_z)),
        "peak_max": float(np.nanmax(peak_z)),
        "active_fraction": float(np.mean(area_z > 0)),
        "active_voxels": active_voxels,
        "active_voxel_fraction": safe_divide(active_voxels, subcube.size),
        "n_active_channels": int(np.count_nonzero(active_channels)),
        "spectral_occupancy": safe_divide(int(np.count_nonzero(active_channels)), subcube.shape[0]),
        "longest_active_run_z": longest_true_run(active_channels),
        "area_mean": area_mean,
        "area_std": float(np.nanstd(area_z)),
        "area_max": float(np.nanmax(area_z)),
        "area_max_over_mean": safe_divide(float(np.nanmax(area_z)), area_mean),
        "area_smoothness": float(np.nanmean(np.abs(np.diff(area_z)))) if len(area_z) > 1 else 0.0,
        "centroid_x_std": float(np.nanstd(centroid_x)) if valid_centroids.any() else np.nan,
        "centroid_y_std": float(np.nanstd(centroid_y)) if valid_centroids.any() else np.nan,
        "centroid_path_length": centroid_path_length(centroid_x, centroid_y),
        "centroid_step_mean": float(np.nanmean(centroid_steps)) if len(centroid_steps) else 0.0,
        "centroid_step_std": float(np.nanstd(centroid_steps)) if len(centroid_steps) else 0.0,
        "centroid_step_max": float(np.nanmax(centroid_steps)) if len(centroid_steps) else 0.0,
        "valid_centroid_fraction": float(valid_centroids.mean()),
        "n_components_3d": n_components_3d,
        "largest_component_size": largest_component_size,
        "largest_component_fraction": safe_divide(largest_component_size, active_voxels),
        "component_fragmentation": safe_divide(n_components_3d, max(active_voxels, 1)),
    }


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


def label_candidate(row: pd.Series, truth_df: pd.DataFrame) -> tuple[int, object, dict[str, float]]:
    if MATCHING_MODE not in MATCHING_THRESHOLDS:
        raise ValueError(f"MATCHING_MODE invalido: {MATCHING_MODE}")

    dx = np.abs(truth_df["truth_x"].to_numpy() - float(row["x"]))
    dy = np.abs(truth_df["truth_y"].to_numpy() - float(row["y"]))
    dz = np.abs(truth_df["truth_z"].to_numpy() - float(row["z"]))
    dist_3d = np.sqrt(dx * dx + dy * dy + dz * dz)
    nearest_index = int(np.nanargmin(dist_3d))
    matching_diagnostics = {
        "min_abs_dx": float(dx[nearest_index]),
        "min_abs_dy": float(dy[nearest_index]),
        "min_abs_dz": float(dz[nearest_index]),
        "min_dist_3d": float(dist_3d[nearest_index]),
        "matching_mode": MATCHING_MODE,
    }

    max_dx, max_dy, max_dz = MATCHING_THRESHOLDS[MATCHING_MODE]
    nearest_matches = dx[nearest_index] <= max_dx and dy[nearest_index] <= max_dy and dz[nearest_index] <= max_dz
    if not nearest_matches:
        return 0, np.nan, matching_diagnostics

    if "id" in truth_df.columns:
        return 1, truth_df.iloc[nearest_index]["id"], matching_diagnostics
    return 1, truth_df.index[nearest_index], matching_diagnostics


def clean_training_label(matching_diagnostics: dict[str, float]) -> int:
    is_medium_match = (
        matching_diagnostics["min_abs_dx"] <= 20
        and matching_diagnostics["min_abs_dy"] <= 20
        and matching_diagnostics["min_abs_dz"] <= 50
    )
    if is_medium_match:
        return 1
    if matching_diagnostics["min_dist_3d"] > 100:
        return 0
    return -1


def candidate_features(row: pd.Series, data_shape: tuple[int, int, int]) -> tuple[int, int, int, int, int, int, dict[str, float]]:
    z_size, y_size, x_size = data_shape
    x0 = int(np.floor(row["x_min"]))
    x1 = int(np.ceil(row["x_max"])) + 1
    y0 = int(np.floor(row["y_min"]))
    y1 = int(np.ceil(row["y_max"])) + 1
    z0 = int(np.floor(row["z_min"]))
    z1 = int(np.ceil(row["z_max"])) + 1

    x0, x1 = crop_around_center(x0, x1, row["x"], MAX_X, x_size)
    y0, y1 = crop_around_center(y0, y1, row["y"], MAX_Y, y_size)
    z0, z1 = crop_around_center(z0, z1, row["z"], MAX_Z, z_size)

    direct = {
        "snr": row.get("snr", np.nan),
        "snr_max": row.get("snr_max", np.nan),
        "n_pix": row.get("n_pix", np.nan),
        "f_sum": row.get("f_sum", np.nan),
        "rms": row.get("rms", np.nan),
        "w20": row.get("w20", np.nan),
        "ell_maj": row.get("ell_maj", np.nan),
        "ell_min": row.get("ell_min", np.nan),
        "x_extent": x1 - x0,
        "y_extent": y1 - y0,
        "z_extent": z1 - z0,
        "ell_ratio": safe_divide(row.get("ell_min", np.nan), row.get("ell_maj", np.nan)),
        "flux_density": safe_divide(row.get("f_sum", np.nan), row.get("n_pix", np.nan)),
    }
    return z0, z1, y0, y1, x0, x1, direct


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    sofia_path = choose_sofia_catalog()
    sofia_df = read_sofia_catalog(sofia_path)
    truth_df = read_truth_catalog(TRUTH_PATH)

    rows = []
    with fits.open(FITS_PATH, memmap=True) as hdul:
        data = hdul[0].data
        header = hdul[0].header
        wcs = WCS(header)
        truth_df = add_truth_pixels(truth_df, wcs)

        for _, row in sofia_df.iterrows():
            z0, z1, y0, y1, x0, x1, features = candidate_features(row, data.shape)
            subcube = data[z0:z1, y0:y1, x0:x1]
            features.update(spatio_spectral_features(subcube))
            label, matched_truth_id, matching_diagnostics = label_candidate(row, truth_df)
            features.update(matching_diagnostics)
            clean_label = clean_training_label(matching_diagnostics)

            features.update({
                "name": row.get("name"),
                "id": row.get("id"),
                "x": row.get("x"),
                "y": row.get("y"),
                "z": row.get("z"),
                "label": label,
                "clean_label": clean_label,
                "is_ambiguous": clean_label == -1,
                "matched_truth_id": matched_truth_id,
            })
            rows.append(features)

    features_df = pd.DataFrame(rows)
    output_csv = OUTPUTS_DIR / "candidates_features.csv"
    features_df.to_csv(output_csv, index=False)

    print(f"Catalogo SoFiA usado: {sofia_path}")
    print(f"Candidatos leidos: {len(sofia_df)}")
    print(f"Features generadas: {len(features_df.columns)} columnas")
    print(f"TP: {int(features_df['label'].sum())}")
    print(f"FP: {int((features_df['label'] == 0).sum())}")
    print("10 candidatos mas cercanos al truth:")
    print(features_df.sort_values("min_dist_3d").head(10)[
        ["name", "id", "x", "y", "z", "label", "matched_truth_id", "min_abs_dx", "min_abs_dy", "min_abs_dz", "min_dist_3d"]
    ])
    print("Primeras filas:")
    print(features_df.head())
    print(f"CSV guardado en: {output_csv}")


if __name__ == "__main__":
    main()
