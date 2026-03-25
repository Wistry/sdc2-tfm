import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
from ska_sdc.sdc2.sdc2_scorer import Sdc2Scorer

# ==============================================================================
PREDICTION_FILE = os.environ.get('SDC2_PREDICTION_FILE', '/path/to/sdc2/results/master_test_cat.txt')
TRUTH_FILE = os.environ.get('SDC2_TRUTH_FILE', '/path/to/sdc2/data/sky_dev_truthcat_v2.txt')
# ==============================================================================

def main():
    print(f"\nCargando predicciones desde: {PREDICTION_FILE}")
    
    # Ponemos las 46 filas que escupe sofia y eliminamos filas fantasma
    sofia_cols = [
        'name', 'id', 'x', 'y', 'z', 'x_min', 'x_max', 'y_min', 'y_max', 
        'z_min', 'z_max', 'n_pix', 'f_min', 'f_max', 'f_sum', 'rel', 'flag', 
        'rms', 'w20', 'w50', 'wm50', 'z_w20', 'z_w50', 'z_wm50', 'ell_maj', 
        'ell_min', 'ell_pa', 'ell3s_maj', 'ell3s_min', 'ell3s_pa', 'kin_pa', 
        'err_x', 'err_y', 'err_z', 'err_f_sum', 'snr', 'snr_max', 'ra', 'dec', 
        'freq', 'x_peak', 'y_peak', 'z_peak', 'ra_peak', 'dec_peak', 'freq_peak'
    ]

    try:
        df_sofia = pd.read_csv(PREDICTION_FILE, sep=r'\s+', comment='#', names=sofia_cols)
        print(f" -> OK: {len(df_sofia)} galaxias detectadas por SoFiA cargadas.")
    except Exception as e:
        print(f"Error al leer la predicción: {e}")
        return

    print(f"\nCargando Truth Catalogue desde: {TRUTH_FILE}")
    try:
        df_truth = pd.read_csv(TRUTH_FILE, sep=r'\s+', comment='#')
        print(f" -> OK: Truth catalogue cargado.")
    except Exception as e:
        print(f"Error al leer Truth: {e}")
        return

    print("\nMapeando columnas al estándar SDC2...")
    df_submission = pd.DataFrame()
    
    df_submission['id'] = df_sofia['id']
    df_submission['ra'] = df_sofia['ra']               
    df_submission['dec'] = df_sofia['dec']             
    df_submission['central_freq'] = df_sofia['freq']   
    df_submission['line_flux_integral'] = df_sofia['f_sum'] 
    df_submission['hi_size'] = df_sofia['ell_maj'] * 4.0   
    df_submission['pa'] = df_sofia['kin_pa']           
    df_submission['i'] = df_sofia['ell_min']           
    df_submission['w20'] = df_sofia['w20']             

    # LIMPIEZA DE DATOS
    df_submission = df_submission.dropna(subset=['id'])
    df_submission = df_submission.fillna(0.0)

    # --- NUEVO: REORDENAMIENTO ESTRICTO ---
    EXPECTED_COLS = ['id', 'ra', 'dec', 'hi_size', 'line_flux_integral', 'central_freq', 'pa', 'i', 'w20']
    df_submission = df_submission[EXPECTED_COLS]
    # --------------------------------------

    print("Iniciando evaluación con SDC2 Scorer...")
    try:
        scorer = Sdc2Scorer(df_submission, df_truth)
        score = scorer.run()
        
        print("\n" + "="*40)
        print(" RESULTADOS DE LA EVALUACIÓN SDC2")
        print("="*40)
        print(f" Puntuación Final (Score): {score.value:.4f}")
        print(f" True Positives (TP)     : {getattr(score, 'n_match', 'N/A')}")
        print(f" False Positives (FP)    : {getattr(score, 'n_false', 'N/A')}")
        print("="*40)

    except Exception as e:
        print(f"\nError interno en el Scorer SDC2: {e}")

if __name__ == "__main__":
    main()