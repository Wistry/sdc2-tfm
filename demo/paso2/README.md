# Demo paso 2: filtrado de candidatos SoFiA con Ciencia de Datos

## Objetivo

Esta demo no detecta galaxias desde cero. Parte del catalogo bruto de candidatos generado por SoFiA y anade una capa de Ciencia de Datos para filtrar falsos positivos.

```text
Cubo FITS 3D
-> SoFiA
-> candidatos
-> subcubo local por candidato
-> features espacio-espectrales
-> clean_label TP/FP
-> RandomForest / XGBoost
-> probabilidad de fuente real
-> catalogo filtrable
```

## Aportacion dentro del TFM

SoFiA actua como generador inicial de candidatos. Mi aportacion es convertir cada candidato en una instancia tabular: para cada deteccion se extrae un subcubo local del FITS y se calculan features de coherencia espacial-espectral.

Con el truth catalogue se etiqueta cada candidato como TP claro, FP claro o ambiguo. Despues se entrena un clasificador supervisado para estimar `p(real | candidato)`.

**El objetivo no es sustituir SoFiA, sino refinar su catalogo bruto mediante una etapa supervisada de mineria de datos.**

## Scripts principales

### `01_extract_spatio_spectral_features.py`

- Lee catalogo SoFiA.
- Lee truth catalogue.
- Abre FITS con `memmap=True`.
- Extrae un subcubo local por candidato.
- Calcula features de intensidad, area, ocupacion espectral, componentes 3D y centroides.
- Genera `outputs/candidates_features.csv`.

### `02_train_candidate_classifier.py`

- Lee `candidates_features.csv`.
- Descarta ambiguos (`clean_label = -1`).
- Excluye columnas con data leakage.
- Entrena RandomForest y XGBoost.
- Compara experimentos con posicion y sin posicion.
- Evalua precision, recall, F1, ROC AUC, PR AUC y thresholds.
- Genera graficas de importancia de variables.

### `03_diagnose_candidates_and_matching.py`

- Calcula distancia candidato-truth.
- Compara criterios `strict`, `medium` y `loose`.
- Ayuda a entender si el problema viene del matching, SoFiA o las etiquetas.

## Features generadas

| Grupo | Variables ejemplo | Que miden |
| --- | --- | --- |
| Intensidad | `flux_mean`, `flux_std`, `flux_smoothness`, `flux_peak_to_mean` | evolucion del flujo a lo largo de `z` |
| Pico | `peak_mean`, `peak_max` | intensidad maxima por canal |
| Ocupacion espectral | `n_active_channels`, `spectral_occupancy`, `longest_active_run_z` | continuidad de senal en canales consecutivos |
| Area espacial | `area_mean`, `area_max`, `area_smoothness` | cuanto ocupa la senal en cada canal |
| Componentes 3D | `n_components_3d`, `largest_component_fraction`, `component_fragmentation` | fragmentacion o coherencia volumetrica |
| Centroide | `centroid_x_std`, `centroid_y_std`, `centroid_path_length`, `centroid_step_max` | movimiento espacial de la senal conforme cambia `z` |
| SoFiA | `snr`, `snr_max`, `n_pix`, `f_sum`, `w20`, `ell_ratio` | parametros originales del candidato |

Estas features buscan capturar si la senal forma una estructura coherente en el espacio 3D, en lugar de aparecer como ruido aislado o fragmentado.

## Mascara activa auxiliar

Para calcular varias features se crea una mascara binaria local. Se define de forma simple como voxels con intensidad superior a un umbral local:

```text
active[z,y,x] = subcube[z,y,x] > mean(subcube) + 2*std(subcube)
```

Esta mascara NO es el ground truth. Solo se usa para calcular features como area, canales activos, componentes conectadas y centroides.

**La mascara activa es una herramienta de extraccion de caracteristicas; la etiqueta de entrenamiento se obtiene aparte mediante el truth catalogue.**

## Etiquetado limpio

```text
clean_label = 1   -> positivo claro
clean_label = 0   -> negativo claro
clean_label = -1  -> ambiguo, descartado
```

- Positivos claros: candidatos suficientemente cercanos al truth catalogue.
- Negativos claros: candidatos suficientemente alejados.
- Ambiguos: casos intermedios que se descartan para no ensuciar el entrenamiento.

## Data leakage

Las columnas derivadas del truth catalogue no entran al modelo. Se excluyen columnas como:

- `label`
- `clean_label`
- `is_ambiguous`
- `matched_truth_id`
- `truth_x`
- `truth_y`
- `truth_z`
- `min_abs_dx`
- `min_abs_dy`
- `min_abs_dz`
- `min_dist_3d`
- `matching_mode`

Estas columnas sirven para etiquetar y diagnosticar, pero no se usan como entrada del modelo porque en una prediccion real no estarian disponibles.

## Ejecucion

```bash
cd demo/paso2

python 01_extract_spatio_spectral_features.py
python 03_diagnose_candidates_and_matching.py
python 02_train_candidate_classifier.py
```

## Outputs principales

- `outputs/candidates_features.csv`
- `outputs/matching_diagnostics.csv`
- `outputs/matching_summary.txt`
- `outputs/model_comparison_clean_labels.txt`
- `outputs/threshold_sweep_all.csv`
- `outputs/feature_importance_rf_with_position.png`
- `outputs/feature_importance_rf_no_position.png`
- `outputs/feature_importance_xgb_with_position.png`
- `outputs/feature_importance_xgb_no_position.png`

## Resultados preliminares

- Candidatos SoFiA: `1564`
- Ejemplos limpios usados: `312`
- TP limpios: `188`
- FP limpios: `124`
- Ambiguos descartados: `1252`

Mejor modelo preliminar:

- XGBoost con posicion
- ROC AUC: `0.8189`
- PR AUC: `0.8770`
- F1@0.50: `0.8034`
- Mejor F1: `0.8217`

Estos resultados son una prueba supervisada sobre candidatos etiquetados, no la puntuacion final del scorer SDC2.

## Limitaciones

- Split 70/30 sobre candidatos limpios, no sobre regiones independientes del cubo.
- Numero de ejemplos limpios limitado.
- La version con posicion funciona mejor, por lo que hay que estudiar generalizacion espacial.
- Falta generar el catalogo filtrado final y evaluarlo con el scorer SDC2.

## Documentacion detallada

Para detalles internos de funciones, outputs y logica de implementacion, consultar [`docs/explicacion_detallada.md`](docs/explicacion_detallada.md).
