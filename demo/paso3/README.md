# Demo paso 3: optimizacion automatica con Optuna

Optuna es una libreria para buscar hiperparametros automaticamente. Define un espacio de busqueda, prueba combinaciones y aprende que zonas parecen mejores. Permite guardar cada ensayo con sus metricas. Aqui se usa como una demo reproducible y pequena sobre el CSV generado en el paso 2.

Esta demo optimiza:

- hiperparametros de XGBoost.
- threshold de decision sobre `predict_proba`.

Entrada:

- `../paso2/outputs/candidates_features.csv`

Salida:

- mejores parametros.
- metricas del mejor ensayo.
- tabla de trials.
- graficas de evolucion, threshold vs F1 e importancia de features.

La demo usa por defecto `EXPERIMENT_MODE = "no_position"` para mostrar si las features morfologicas/espectrales aportan senal sin depender de coordenadas.

## Comandos

```bash
cd demo/paso3
python 01_optuna_xgboost_candidate_filter.py
```
