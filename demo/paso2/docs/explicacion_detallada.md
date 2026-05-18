# Explicacion completa de la demo paso 2

## Idea general

La demo `demo/paso2` representa una etapa de Ciencia de Datos posterior a SoFiA.

SoFiA ya ha hecho una primera deteccion sobre el cubo astronomico y ha generado un catalogo de candidatos. Cada candidato es una posible galaxia HI detectada en el cubo 3D. El objetivo de esta demo no es volver a detectar desde cero, sino analizar cada candidato y decidir si parece una deteccion real o un falso positivo.

En palabras sencillas:

- SoFiA dice: "aqui podria haber algo".
- La demo paso 2 mira con mas detalle alrededor de ese "algo".
- Extrae un pequeno bloque local del cubo FITS.
- Resume como se comporta la senal dentro de ese bloque.
- Usa esas variables para entrenar un clasificador.

El flujo conceptual es:

```text
Candidato SoFiA
-> subcubo local cube[z, y, x]
-> por cada canal z: flujo, area, centroide
-> resumen de coherencia espacial-espectral
-> etiqueta TP/FP aproximada con truth catalogue
-> RandomForest / XGBoost
```

## Que significa cube[z, y, x]

El FITS se interpreta como un cubo 3D:

```text
cube[z, y, x]
```

Donde:

- `z` es el eje espectral, relacionado con frecuencia/redshift.
- `y` es una coordenada espacial de la imagen.
- `x` es la otra coordenada espacial.
- cada valor `cube[z, y, x]` es una intensidad float.

Una galaxia HI real no aparece normalmente como un unico pixel aislado. Lo esperable es que tenga cierta continuidad:

- aparece en varios canales `z`;
- ocupa una pequena region espacial en `x,y`;
- su centroide no deberia saltar de forma caotica;
- su flujo deberia evolucionar con cierta coherencia.

Esa es la razon de extraer features locales.

## Ficheros principales

La demo tiene tres scripts:

```text
01_extract_spatio_spectral_features.py
02_train_candidate_classifier.py
03_diagnose_candidates_and_matching.py
```

Cada uno responde a una pregunta distinta.

### 01_extract_spatio_spectral_features.py

Pregunta que responde:

```text
Como convierto candidatos SoFiA en una tabla de features para Machine Learning?
```

Este script:

1. Lee el catalogo de candidatos SoFiA.
2. Lee el truth catalogue.
3. Abre el FITS con `memmap=True`.
4. Para cada candidato, extrae un subcubo local.
5. Calcula features tabulares y features espacio-espectrales.
6. Asigna etiquetas aproximadas TP/FP.
7. Guarda `outputs/candidates_features.csv`.

### 02_train_candidate_classifier.py

Pregunta que responde:

```text
Con esas features, un modelo puede separar candidatos reales de falsos positivos?
```

Este script:

1. Lee `outputs/candidates_features.csv`.
2. Usa `clean_label` como etiqueta de entrenamiento.
3. Descarta candidatos ambiguos.
4. Excluye columnas que causarian data leakage.
5. Entrena RandomForest y XGBoost si esta instalado.
6. Compara dos experimentos:
   - con posicion;
   - sin posicion.
7. Evalua thresholds de decision.
8. Guarda metricas y graficas de importancia.

### 03_diagnose_candidates_and_matching.py

Pregunta que responde:

```text
El problema esta en SoFiA, en el matching, en las features o en el modelo?
```

Este script no entrena. Diagnostica.

Calcula la distancia de cada candidato SoFiA al truth mas cercano y prueba varios criterios de matching:

- estricto: `dx<=10`, `dy<=10`, `dz<=25`;
- medio: `dx<=20`, `dy<=20`, `dz<=50`;
- laxo: `dx<=30`, `dy<=30`, `dz<=75`;
- por distancia 3D: `dist_3d <= 30`, `50`, `75`.

Esto sirve para ver si una mala metrica del modelo puede venir de etiquetas demasiado ruidosas o demasiado estrictas.

## Como se extrae cada subcubo local

La parte clave esta en `01_extract_spatio_spectral_features.py`.

Cada candidato SoFiA trae un bounding box 3D:

```python
x_min, x_max
y_min, y_max
z_min, z_max
```

El script transforma esos limites en indices enteros:

```python
x0 = int(np.floor(row["x_min"]))
x1 = int(np.ceil(row["x_max"])) + 1
y0 = int(np.floor(row["y_min"]))
y1 = int(np.ceil(row["y_max"])) + 1
z0 = int(np.floor(row["z_min"]))
z1 = int(np.ceil(row["z_max"])) + 1
```

Luego limita el tamano maximo del subcubo:

```python
MAX_Z = 128
MAX_Y = 64
MAX_X = 64
```

Si un candidato es demasiado grande, no se lee todo. Se recorta alrededor del centro del candidato:

```python
x0, x1 = crop_around_center(x0, x1, row["x"], MAX_X, x_size)
y0, y1 = crop_around_center(y0, y1, row["y"], MAX_Y, y_size)
z0, z1 = crop_around_center(z0, z1, row["z"], MAX_Z, z_size)
```

Finalmente se extrae el subcubo:

```python
subcube = data[z0:z1, y0:y1, x0:x1]
```

Esto significa:

```text
del cubo completo,
lee solo los canales z entre z0 y z1,
solo las filas y entre y0 y y1,
y solo las columnas x entre x0 y x1.
```

Importante: el FITS se abre asi:

```python
fits.open(FITS_PATH, memmap=True)
```

Eso evita cargar el FITS completo en RAM. Se trabaja con trozos locales.

## Funciones importantes del archivo 01

El archivo `01_extract_spatio_spectral_features.py` esta organizado como una pequena pipeline. Cada funcion tiene una responsabilidad concreta.

### `read_sofia_catalog(path)`

Lee el catalogo ASCII generado por SoFiA:

```python
df = pd.read_csv(path, sep=r"\s+", comment="#", names=SOFIA_COLUMNS, engine="python")
```

Esto significa que lee columnas separadas por espacios, ignora comentarios con `#` y asigna nombres usando `SOFIA_COLUMNS`.

Despues convierte a numericas todas las columnas excepto `name` y elimina filas que no tengan coordenadas basicas:

```text
x, y, z, x_min, x_max, y_min, y_max, z_min, z_max
```

En sencillo:

```text
Convierte el catalogo de SoFiA en una tabla pandas limpia.
```

### `read_truth_catalog(path)`

Lee el truth catalogue y comprueba que existan:

```text
ra, dec, central_freq
```

Estas columnas son necesarias para convertir posiciones astronomicas a pixeles del cubo.

En sencillo:

```text
Carga las posiciones reales de las fuentes simuladas.
```

### `choose_sofia_catalog()`

Elige el catalogo SoFiA disponible con este orden:

```text
master_test_many_candidates_cat.txt
master_test_medium_cat.txt
master_test_small_cat.txt
```

En sencillo:

```text
Usa primero el catalogo con mas candidatos; si no existe, usa uno mas pequeno.
```

### `clamp_interval(start, stop, max_size)`

Asegura que un intervalo no se salga de los limites del cubo.

Por ejemplo, si un candidato empezara en `x=-5`, esta funcion lo corrige para no leer indices negativos.

Tambien evita intervalos vacios:

```python
if stop <= start:
    stop = min(max_size, start + 1)
```

En sencillo:

```text
Evita que el codigo intente leer fuera del FITS.
```

### `crop_around_center(start, stop, center, limit, max_size)`

Recorta un intervalo si es demasiado grande.

Si el bounding box cabe dentro del limite, lo deja igual:

```python
if stop - start <= limit:
    return start, stop
```

Si no cabe, crea un intervalo nuevo centrado alrededor de `center`.

Ejemplo:

```text
MAX_X = 64
```

Si SoFiA da un candidato de 200 pixeles en `x`, la demo no extrae los 200. Extrae como maximo 64 alrededor del centro.

En sencillo:

```text
Mantiene los subcubos pequenos y manejables.
```

### `safe_divide(numerator, denominator)`

Hace divisiones evitando errores por cero.

Si el denominador es cero o NaN, devuelve NaN:

```python
if denominator == 0 or pd.isna(denominator):
    return np.nan
```

Se usa en ratios como:

- `ell_ratio`;
- `flux_density`;
- `largest_component_fraction`;
- `component_fragmentation`.

En sencillo:

```text
Evita que una division invalida rompa el script.
```

### `centroid_path_length(x_values, y_values)`

Calcula la distancia total recorrida por el centroide a lo largo del eje `z`.

Si los centroides validos son:

```text
z1 -> (x1, y1)
z2 -> (x2, y2)
z3 -> (x3, y3)
```

calcula:

```text
distancia(z1,z2) + distancia(z2,z3)
```

En sencillo:

```text
Mide si el centro de la fuente se desplaza poco o mucho entre canales.
```

### `longest_true_run(values)`

Calcula la racha mas larga de valores verdaderos consecutivos.

Se usa con:

```python
active_channels = area_z > 0
```

Ejemplo:

```text
[False, True, True, True, False, True] -> 3
```

En sencillo:

```text
Mide cuantos canales consecutivos contienen senal activa.
```

### `count_local_peaks(values)`

Cuenta picos locales en una serie.

Un punto es pico local si:

```text
valor_anterior < valor_actual > valor_siguiente
```

Se usa sobre `flux_z`.

En sencillo:

```text
Cuenta cuantos maximos locales tiene la curva de flujo.
```

### `connected_components_3d(active)`

Busca componentes conectadas dentro de la mascara activa 3D:

```python
active_3d = subcube > mean + 2 * std
```

Devuelve:

```text
n_components_3d, largest_component_size
```

Si `scipy.ndimage` esta disponible, usa `ndimage.label`. Si no, usa una busqueda manual con pila.

En sencillo:

```text
Mira si la senal activa forma una estructura continua o muchos trozos separados.
```

### `spatio_spectral_features(subcube)`

Es la funcion central del archivo.

Recibe un subcubo y devuelve un diccionario de features numericas.

Dentro hace:

1. calcula el umbral local `mean + 2*std`;
2. crea la mascara activa 3D;
3. calcula componentes conectados;
4. recorre canal por canal;
5. calcula `flux_z`, `peak_z`, `area_z` y centroides;
6. resume todo en features de coherencia.

En sencillo:

```text
Convierte un subcubo 3D en una fila de numeros utiles para Machine Learning.
```

Esta es la funcion mas importante para la extraccion de features.

### `add_truth_pixels(truth_df, wcs)`

Convierte las coordenadas astronomicas del truth:

```text
ra, dec, central_freq
```

a coordenadas pixel:

```text
truth_x, truth_y, truth_z
```

Usa:

```python
wcs.world_to_pixel_values(...)
```

En sencillo:

```text
Pasa las posiciones reales al mismo sistema de coordenadas que los candidatos SoFiA.
```

### `label_candidate(row, truth_df)`

Compara un candidato con el truth mas cercano.

Calcula:

```text
dx, dy, dz, dist_3d
```

Luego aplica:

```text
MATCHING_MODE = "medium"
```

Con:

```text
strict = dx<=10, dy<=10, dz<=25
medium = dx<=20, dy<=20, dz<=50
loose = dx<=30, dy<=30, dz<=75
```

Si el truth mas cercano cumple el criterio, `label = 1`; si no, `label = 0`.

Tambien guarda columnas diagnosticas:

```text
min_abs_dx, min_abs_dy, min_abs_dz, min_dist_3d, matching_mode
```

En sencillo:

```text
Decide si un candidato esta suficientemente cerca de una fuente real.
```

### `clean_training_label(matching_diagnostics)`

Crea una etiqueta mas limpia para entrenamiento:

```text
clean_label = 1  si cumple medium
clean_label = 0  si min_dist_3d > 100
clean_label = -1 si queda en zona ambigua
```

En sencillo:

```text
Evita entrenar con casos dudosos.
```

Esta funcion fue clave porque el modelo mejoro mucho al descartar ambiguos.

### `candidate_features(row, data_shape)`

Prepara los limites del subcubo de cada candidato.

Hace tres cosas:

1. lee `x_min/x_max`, `y_min/y_max`, `z_min/z_max`;
2. recorta si supera `MAX_X`, `MAX_Y`, `MAX_Z`;
3. devuelve `z0, z1, y0, y1, x0, x1`.

Tambien calcula features directas de SoFiA:

```text
snr, snr_max, n_pix, f_sum, rms, w20, ell_maj, ell_min, extensiones, ratios
```

En sencillo:

```text
Decide que trozo del cubo hay que leer para cada candidato.
```

### `main()`

Une todo el proceso.

Flujo:

```text
lee catalogo SoFiA
lee truth catalogue
abre FITS con memmap=True
convierte truth a pixeles
para cada candidato:
    calcula limites del subcubo
    extrae subcube = data[z0:z1, y0:y1, x0:x1]
    calcula features
    calcula label y clean_label
guarda outputs/candidates_features.csv
```

En sencillo:

```text
Ejecuta toda la cadena de extraccion de features.
```

## Funciones importantes del archivo 02

El archivo `02_train_candidate_classifier.py` es el encargado de entrenar y evaluar modelos a partir del CSV generado por el archivo 01.

Su entrada principal es:

```text
outputs/candidates_features.csv
```

Sus salidas principales son:

```text
outputs/model_comparison_clean_labels.txt
outputs/threshold_sweep_all.csv
outputs/feature_importance_*.png
```

En sencillo:

```text
El archivo 01 crea las features.
El archivo 02 prueba si esas features sirven para separar candidatos reales y falsos positivos.
```

### Configuracion inicial

El script usa `pandas` y `numpy` para manejar tablas, `sklearn` para modelos y metricas, `matplotlib` para graficas, y `xgboost` si esta instalado.

La importacion de XGBoost es opcional:

```python
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
```

Esto significa:

```text
Si XGBoost existe, lo usa.
Si no existe, el script sigue funcionando solo con RandomForest.
```

### `ALWAYS_EXCLUDE`

Esta constante contiene columnas que nunca deben entrar al modelo:

```python
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
```

La razon es evitar `data leakage`.

Por ejemplo, `min_dist_3d` viene de comparar con el truth catalogue. En un caso real no tendriamos el truth, asi que si el modelo usara esa columna estaria haciendo trampa.

En sencillo:

```text
Estas columnas sirven para diagnosticar y etiquetar, pero no para entrenar.
```

### `POSITION_COLUMNS`

Contiene columnas de posicion:

```python
POSITION_COLUMNS = {
    "x", "y", "z",
    "ra", "dec", "freq",
    "x_peak", "y_peak", "z_peak",
    "ra_peak", "dec_peak", "freq_peak",
}
```

Se usan para crear el experimento `no_position`.

La pregunta que responde es:

```text
El modelo aprende de la morfologia/espectro del candidato,
o aprende sobre todo de donde esta colocado en el cubo?
```

### `prepare_X(df, without_position)`

Prepara la matriz de features `X`.

Hace cuatro cosas:

1. excluye columnas de leakage;
2. si `without_position=True`, excluye tambien coordenadas;
3. se queda solo con columnas numericas;
4. rellena NaN con la mediana y, si hace falta, con 0.

Codigo clave:

```python
feature_df = df.drop(columns=[col for col in exclude if col in df.columns])
X = feature_df.select_dtypes(include=[np.number])
return X.fillna(X.median(numeric_only=True)).fillna(0.0)
```

En sencillo:

```text
Deja lista la tabla numerica que entra al modelo.
```

### `threshold_sweep(y_test, y_score, experiment, model_name)`

Prueba varios thresholds de decision:

```python
THRESHOLDS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
```

El modelo devuelve probabilidades:

```text
y_score = probabilidad de que el candidato sea real
```

Para cada threshold:

```python
y_pred = y_score >= threshold
```

Y calcula:

- precision;
- recall;
- F1;
- matriz de confusion;
- TP aceptados;
- FP aceptados.

En sencillo:

```text
Ayuda a decidir cuan estricto debe ser el filtro final.
```

Threshold bajo:

- acepta mas candidatos;
- recupera mas TP;
- tambien deja pasar mas FP.

Threshold alto:

- acepta menos candidatos;
- reduce FP;
- puede perder TP.

### `save_importance_png(features, importances, path, title)`

Guarda una grafica con las 15 features mas importantes.

Sirve para interpretar el modelo:

```text
Que variables esta usando mas para decidir?
```

Si aparecen features como `n_active_channels`, `area_max`, `centroid_x_std` o `n_components_3d`, significa que las features espacio-espectrales estan aportando senal.

### `evaluate_model(model, model_name, experiment, X, y, importance_path)`

Esta es la funcion central del archivo 02.

Hace:

1. divide datos en train/test;
2. entrena el modelo;
3. predice probabilidades;
4. evalua metricas;
5. prueba thresholds;
6. guarda importancia de features;
7. devuelve texto de resultados.

El split es:

```python
train_test_split(
    X,
    y,
    test_size=0.3,
    stratify=y,
    random_state=42,
)
```

Esto significa:

- 70% entrenamiento;
- 30% test;
- mantiene proporcion de TP/FP con `stratify`;
- es reproducible con `random_state=42`.

El modelo produce probabilidades:

```python
y_score = model.predict_proba(X_test)[:, 1]
```

`[:, 1]` es la probabilidad de clase positiva:

```text
candidato real
```

En sencillo:

```text
Entrena un modelo y comprueba que tal filtra candidatos.
```

### `matching_counts(df)`

Cuenta cuantos candidatos serian TP/FP bajo criterios:

```text
strict
medium
loose
```

Usa:

```text
min_abs_dx, min_abs_dy, min_abs_dz
```

En sencillo:

```text
Resume si cambiar el criterio de matching cambia mucho el numero de positivos.
```

### `automatic_conclusions(df, metrics, sweep_all)`

Genera conclusiones automaticas:

- cuantos TP/FP limpios se usaron;
- si el matching estricto parece demasiado duro;
- si el modelo depende de posicion;
- que threshold parece razonable;
- si las features espacio-espectrales parecen aportar senal.

En sencillo:

```text
Convierte los numeros en una primera interpretacion.
```

### `main()`

Une toda la pipeline de entrenamiento.

Primero lee:

```python
df = pd.read_csv(FEATURES_PATH)
```

Luego exige que exista:

```text
clean_label
```

Despues descarta ambiguos:

```python
train_df = df[df["clean_label"] != -1].copy()
```

La etiqueta usada para entrenar es:

```python
y = train_df["clean_label"].astype(int)
```

Luego crea dos experimentos:

```python
experiments = {
    "with_position": prepare_X(train_df, without_position=False),
    "no_position": prepare_X(train_df, without_position=True),
}
```

Y para cada experimento entrena:

```text
RandomForest
XGBoost, si esta instalado
```

Finalmente guarda:

```text
model_comparison_clean_labels.txt
threshold_sweep_all.csv
feature_importance_rf_with_position.png
feature_importance_rf_no_position.png
feature_importance_xgb_with_position.png
feature_importance_xgb_no_position.png
```

En sencillo:

```text
Compara modelos, compara usar/no usar posicion, evalua thresholds y guarda resultados.
```

## Como leer las metricas del archivo 02

### Precision

De todos los candidatos que el modelo acepta como reales, cuantos son realmente TP.

```text
precision alta = pocos falsos positivos
```

### Recall

De todos los candidatos reales, cuantos recupera el modelo.

```text
recall alto = se pierden pocos TP
```

### F1

Media equilibrada entre precision y recall.

```text
F1 alto = buen equilibrio entre recuperar TP y no aceptar demasiados FP
```

### ROC AUC

Mide capacidad general de separacion entre clases.

Es util, aunque en problemas desbalanceados puede ser menos informativa que PR AUC.

### PR AUC

Area bajo la curva precision-recall.

Es muy importante cuando hay muchos falsos positivos o clases desbalanceadas.

### Matriz de confusion

Tiene esta forma:

```text
[[TN FP]
 [FN TP]]
```

Donde:

- `TN`: falsos candidatos correctamente rechazados;
- `FP`: falsos positivos aceptados por error;
- `FN`: candidatos reales rechazados por error;
- `TP`: candidatos reales aceptados correctamente.

## Resumen rapido del archivo 02

```text
1. Lee candidates_features.csv.
2. Descarta clean_label = -1.
3. Usa clean_label como etiqueta.
4. Excluye columnas de leakage.
5. Crea experimento con posicion.
6. Crea experimento sin posicion.
7. Entrena RandomForest y XGBoost si esta instalado.
8. Evalua probabilidades con distintos thresholds.
9. Guarda metricas, sweep de thresholds e importancias.
10. Genera conclusiones automaticas.
```

## Explicacion de los archivos de outputs

La carpeta `demo/paso2/outputs/` contiene los resultados generados por los scripts de la demo. Cada archivo tiene una funcion distinta dentro del flujo.

### `candidates_features.csv`

Es el dataset principal de la demo.

Cada fila representa un candidato detectado por SoFiA.

Incluye:

- features originales del catalogo SoFiA;
- features extraidas del subcubo local;
- features de coherencia espacial-espectral;
- columnas de diagnostico de matching;
- `label`;
- `clean_label`;
- `is_ambiguous`.

Este archivo es la entrada principal del entrenamiento en `02_train_candidate_classifier.py`.

En sencillo:

```text
Es la tabla final de candidatos convertidos en variables de Machine Learning.
```

### `matching_diagnostics.csv`

Lo genera `03_diagnose_candidates_and_matching.py`.

Contiene, para cada candidato SoFiA:

- coordenadas del candidato;
- SNR, flujo, numero de pixeles y otras columnas basicas;
- distancia al truth mas cercano;
- `dx`, `dy`, `dz`;
- `min_dist_3d`;
- resultado bajo distintos criterios de matching.

Sirve para estudiar si el matching TP/FP es demasiado estricto, demasiado laxo o razonable.

En sencillo:

```text
Es la tabla para entender si los candidatos estan cerca o lejos del truth.
```

### `matching_summary.txt`

Es un resumen en texto del diagnostico de matching.

Incluye:

- catalogo SoFiA usado;
- numero de candidatos;
- numero de filas del truth;
- conteos TP/FP con varios criterios;
- los candidatos mas cercanos al truth.

Sirve para una lectura rapida sin abrir el CSV completo.

En sencillo:

```text
Es el resumen legible del archivo matching_diagnostics.csv.
```

### `model_comparison_clean_labels.txt`

Es el archivo principal de resultados de entrenamiento.

Lo genera `02_train_candidate_classifier.py`.

Incluye:

- total de filas;
- filas usadas para entrenamiento;
- TP limpios;
- FP limpios;
- ambiguos descartados;
- resultados con posicion;
- resultados sin posicion;
- resultados de RandomForest;
- resultados de XGBoost, si esta instalado;
- ROC AUC;
- PR AUC;
- precision;
- recall;
- F1;
- matriz de confusion;
- classification report;
- top features;
- conclusiones automaticas.

En sencillo:

```text
Es el informe principal para saber que modelo funciona mejor.
```

### `metrics.txt`

Es una copia/resumen de las metricas actuales del entrenamiento.

En esta demo normalmente contiene el mismo contenido que `model_comparison_clean_labels.txt`, porque el script tambien lo escribe ahi para tener una ruta generica de metricas.

En sencillo:

```text
Es el archivo rapido de metricas actuales.
```

### `threshold_sweep_all.csv`

Contiene la evaluacion de distintos thresholds para cada experimento y modelo.

Columnas importantes:

- `experiment`: `with_position` o `no_position`;
- `model`: RandomForest o XGBoost;
- `threshold`;
- `precision`;
- `recall`;
- `f1`;
- `tn`;
- `fp`;
- `fn`;
- `tp`;
- `tp_accepted`;
- `fp_accepted`.

Sirve para elegir el threshold segun el objetivo:

- maximizar F1;
- reducir falsos positivos;
- mantener recall alto;
- buscar precision minima.

En sencillo:

```text
Es la tabla para decidir donde cortar la probabilidad del modelo.
```

### `feature_importance_rf_with_position.png`

Grafica de importancia de features para:

```text
RandomForest + with_position
```

Incluye coordenadas como `x`, `y`, `z` si estan disponibles.

Sirve para ver si RandomForest esta usando mucho la posicion.

En sencillo:

```text
Muestra que variables usa mas RandomForest cuando puede ver coordenadas.
```

### `feature_importance_rf_no_position.png`

Grafica de importancia de features para:

```text
RandomForest + no_position
```

Aqui se excluyen coordenadas. Por tanto, el modelo tiene que apoyarse mas en:

- flujo;
- area;
- ocupacion espectral;
- centroides;
- componentes 3D;
- variables SoFiA no posicionales.

En sencillo:

```text
Muestra que aprende RandomForest cuando no puede usar posicion.
```

### `feature_importance_xgb_with_position.png`

Grafica de importancia de features para:

```text
XGBoost + with_position
```

En los resultados actuales fue uno de los modelos mas fuertes.

Sirve para ver que mezcla usa XGBoost:

- coordenadas;
- features directas de SoFiA;
- features 3D;
- features de coherencia.

En sencillo:

```text
Muestra que variables usa XGBoost cuando tiene toda la informacion permitida.
```

### `feature_importance_xgb_no_position.png`

Grafica de importancia de features para:

```text
XGBoost + no_position
```

Es especialmente interesante para el TFM porque muestra si el modelo aprende de la estructura del candidato, no solo de su localizacion.

Si aparecen features como:

- `area_max`;
- `longest_active_run_z`;
- `active_fraction`;
- `largest_component_fraction`;
- `centroid_path_length`;

entonces hay evidencia de que las features espacio-espectrales aportan senal.

En sencillo:

```text
Muestra que variables morfologicas/espectrales usa XGBoost sin coordenadas.
```

### `min_dist_3d_hist.png`

Histograma de la distancia 3D minima entre cada candidato y el truth mas cercano.

Sirve para ver si muchos candidatos estan:

- muy cerca del truth;
- muy lejos;
- en una zona intermedia ambigua.

En sencillo:

```text
Muestra como de cerca estan los candidatos SoFiA de las fuentes reales.
```

### `snr_vs_dist.png`

Scatter plot:

```text
snr vs min_dist_3d
```

Sirve para ver si los candidatos con mayor SNR tienden a estar mas cerca del truth.

Si no hay relacion clara, significa que SNR por si solo no basta para separar TP/FP.

En sencillo:

```text
Comprueba si mas SNR implica estar mas cerca de una fuente real.
```

### `flux_vs_dist.png`

Scatter plot:

```text
f_sum vs min_dist_3d
```

Sirve para ver si el flujo total del candidato esta relacionado con su cercania al truth.

En sencillo:

```text
Comprueba si los candidatos con mas flujo parecen mas reales.
```

### `snr_tp_fp.png`

Grafica de la distribucion de SNR para TP y FP segun el criterio medium.

Sirve para ver si los TP tienen SNR claramente distinto de los FP.

Si las distribuciones se solapan mucho, SNR no basta como filtro unico.

En sencillo:

```text
Compara el SNR de candidatos reales y falsos positivos aproximados.
```

## Ejemplo mental sencillo

Imagina que el cubo FITS es un edificio:

- cada planta del edificio es un canal `z`;
- cada planta tiene una imagen 2D `y,x`;
- una galaxia seria una mancha que aparece en varias plantas cercanas.

SoFiA marca una zona del edificio donde cree que hay algo. La demo paso 2 no revisa todo el edificio: solo recorta unas habitaciones alrededor de esa zona. Ese recorte es el subcubo.

Luego pregunta:

- cuanta luz hay en cada planta?
- en cuantas plantas aparece algo?
- la mancha se mueve suavemente o salta?
- esta concentrada o fragmentada?
- hay una componente 3D continua o muchos trozos sueltos?

Esas respuestas se convierten en columnas numericas.

# Features 

## Features directas de SoFiA

El script conserva variables que vienen del catalogo SoFiA, por ejemplo:

- `snr`;
- `snr_max`;
- `n_pix`;
- `f_sum`;
- `rms`;
- `w20`;
- `ell_maj`;
- `ell_min`;
- `x_extent`;
- `y_extent`;
- `z_extent`;
- `ell_ratio`;
- `flux_density`.

Estas features ya resumen informacion del candidato detectado por SoFiA.

Ejemplo sencillo:

- `snr`: relacion senal-ruido.
- `n_pix`: numero de pixeles asociados al candidato.
- `f_sum`: flujo total.
- `w20`: anchura espectral.
- `ell_ratio`: forma proyectada, si es alargada o compacta.

## Features por canal z

Dentro de cada subcubo, el script mira canal por canal.

Para cada slice 2D:

```text
slice_2d = subcube[z, :, :]
```

Primero define una mascara activa:

```python
active = slice_2d > mean(subcube) + 2 * std(subcube)
```

Es decir:

```text
pixel activo = pixel claramente por encima del fondo local
```

Para cada canal `z`, calcula:

- `flux_z`: suma de intensidades del canal;
- `peak_z`: valor maximo del canal;
- `area_z`: numero de pixeles activos;
- `x_centroid_z`, `y_centroid_z`: centroide ponderado de la zona activa.

En lenguaje sencillo:

- `flux_z` dice cuanta senal hay en ese canal.
- `area_z` dice cuanto ocupa la senal.
- el centroide dice donde esta el centro de la senal en la imagen.

## Resumen de la evolucion espectral

Una vez que se tiene una serie por canal `z`, el script la resume.

Por ejemplo, para el flujo:

- `flux_mean`;
- `flux_std`;
- `flux_max`;
- `flux_min`;
- `flux_smoothness`;
- `flux_num_local_peaks`;
- `flux_peak_to_mean`.

Interpretacion:

- si el flujo cambia suavemente, puede parecer una fuente real;
- si el flujo tiene saltos muy bruscos, puede ser ruido;
- si hay demasiados picos locales, puede indicar fragmentacion o artefactos.

Para el area:

- `area_mean`;
- `area_std`;
- `area_max`;
- `area_max_over_mean`;
- `area_smoothness`.

Interpretacion:

- una fuente real puede crecer y decrecer en area con cierta continuidad;
- un falso positivo puede aparecer en pixeles sueltos o con forma irregular.

## Features de ocupacion espectral

Estas features miden en cuantos canales aparece senal activa:

- `active_voxels`;
- `active_voxel_fraction`;
- `n_active_channels`;
- `spectral_occupancy`;
- `longest_active_run_z`.

Interpretacion sencilla:

- `active_voxels`: cuantos voxeles superan el umbral local.
- `n_active_channels`: en cuantos canales `z` hay actividad.
- `spectral_occupancy`: fraccion de canales con actividad.
- `longest_active_run_z`: tramo mas largo de canales consecutivos con senal.

Esto es importante porque una galaxia real no deberia aparecer solo como un pixel aislado en un unico canal, salvo casos extremos.

## Que significa coherencia espacial-espectral

Cuando hablamos de features con coherencia espacial-espectral nos referimos a variables que no miran solo un numero aislado del candidato, sino como se organiza la senal dentro del subcubo 3D.

La idea es comprobar si la senal tiene sentido como fuente fisica:

- coherencia espectral: la senal aparece de forma razonable a lo largo del eje `z`, no en canales aleatorios;
- coherencia espacial: la senal ocupa una region compacta o continua en `x,y`;
- coherencia 3D: la senal forma una estructura conectada en `(z,y,x)`, no muchos puntos sueltos;
- coherencia de movimiento: el centroide no salta caoticamente entre canales.

En sencillo:

```text
Una fuente real deberia parecer una mancha que evoluciona con cierta continuidad.
Un falso positivo suele parecer ruido: fragmentado, irregular o aislado.
```

### Features de flujo

Estas features resumen la serie `flux_z`, que es el flujo integrado por canal espectral. Para cada canal `z`, se suma la intensidad de su imagen 2D:

```text
flux_z = sum(slice_2d)
```

- `flux_mean`: flujo medio a lo largo de los canales del subcubo. Indica el nivel medio de senal del candidato.
- `flux_std`: variabilidad del flujo entre canales. Si es alta, el flujo cambia mucho con `z`.
- `flux_max`: maximo flujo encontrado en un canal. Indica el canal donde la senal integrada es mas fuerte.
- `flux_min`: minimo flujo encontrado en un canal. Ayuda a ver el rango de variacion.
- `flux_smoothness`: media de los cambios absolutos entre canales consecutivos. Se calcula como `mean(abs(diff(flux_z)))`. Si es bajo, el flujo evoluciona suavemente; si es alto, hay saltos bruscos.
- `flux_num_local_peaks`: numero de picos locales en la curva de flujo. Muchos picos pueden indicar ruido o estructura fragmentada.
- `flux_peak_to_mean`: relacion entre el flujo maximo y el flujo medio absoluto. Mide si la senal esta dominada por un unico canal o repartida en varios.

Lectura intuitiva:

```text
Si una fuente es real, esperamos una curva de flujo razonablemente continua.
Si solo hay un pico raro o saltos fuertes, puede ser ruido.
```

### Features de pico por canal

Estas features resumen `peak_z`, que es el valor maximo de intensidad en cada canal:

```text
peak_z = max(slice_2d)
```

- `peak_mean`: valor medio del pico maximo por canal. Mide la intensidad maxima tipica.
- `peak_max`: mayor pico encontrado en todo el subcubo. Detecta el voxel/canal mas intenso.

Lectura intuitiva:

```text
El pico dice donde esta la senal mas intensa.
Pero un pico muy alto aislado tambien puede ser ruido, por eso se combina con area, continuidad y componentes 3D.
```

### Features de ocupacion activa

Primero se define una mascara activa:

```python
active = subcube > mean(subcube) + 2 * std(subcube)
```

Es decir, se consideran activos los voxeles claramente por encima del fondo local.

- `active_fraction`: fraccion de canales `z` donde hay al menos un pixel activo. Es equivalente a preguntar: en que proporcion de canales aparece algo?
- `active_voxels`: numero total de voxeles activos en el subcubo.
- `active_voxel_fraction`: proporcion de voxeles activos respecto al tamano total del subcubo.
- `n_active_channels`: numero de canales `z` con area activa mayor que cero.
- `spectral_occupancy`: `n_active_channels / z_extent`. Mide que parte del eje espectral esta ocupada por senal.
- `longest_active_run_z`: tramo mas largo de canales consecutivos con senal activa.

Lectura intuitiva:

```text
Una fuente real suele ocupar varios canales cercanos.
Un falso positivo puede aparecer solo en canales sueltos.
```

Ejemplo:

```text
Canales activos: 0 0 1 1 1 0 1 0
longest_active_run_z = 3
```

Eso significa que hubo tres canales consecutivos con senal.

### Features de area espacial

Para cada canal se calcula:

```text
area_z = numero de pixeles activos en ese canal
```

Luego se resume esa serie:

- `area_mean`: area activa media por canal.
- `area_std`: variabilidad del area entre canales.
- `area_max`: area maxima alcanzada en un canal.
- `area_max_over_mean`: relacion entre area maxima y area media. Si es muy alta, puede indicar que un canal domina demasiado.
- `area_smoothness`: media de `abs(diff(area_z))`. Mide si el area cambia suavemente entre canales.

Lectura intuitiva:

```text
Si la fuente es coherente, el area activa no deberia aparecer y desaparecer de forma totalmente caotica.
```

Una galaxia puede tener mas area en los canales centrales y menos en los extremos, pero esa evolucion deberia tener cierta continuidad.

### Features de centroide

En cada canal con pixeles activos se calcula un centroide ponderado:

```text
x_centroid_z, y_centroid_z
```

El centroide es como el "centro de masa" de la senal activa. Los pixeles mas intensos pesan mas.

- `centroid_x_std`: dispersion del centroide en el eje `x`.
- `centroid_y_std`: dispersion del centroide en el eje `y`.
- `centroid_path_length`: distancia total recorrida por el centroide al avanzar en `z`.
- `centroid_step_mean`: salto medio del centroide entre canales consecutivos validos.
- `centroid_step_std`: variabilidad de esos saltos.
- `centroid_step_max`: mayor salto del centroide entre dos canales.
- `valid_centroid_fraction`: fraccion de canales donde se pudo calcular un centroide.

Lectura intuitiva:

```text
Una fuente real deberia tener un centroide relativamente estable.
Si el centroide salta mucho, puede ser que la mascara activa este capturando ruido en sitios distintos.
```

Ejemplo sencillo:

```text
Canal z=1: centroide en (10, 10)
Canal z=2: centroide en (11, 10)
Canal z=3: centroide en (12, 11)
```

Eso parece una evolucion suave.

Pero:

```text
Canal z=1: centroide en (10, 10)
Canal z=2: centroide en (40, 3)
Canal z=3: centroide en (5, 50)
```

Eso parece mucho menos coherente.

### Features de componentes conectados 3D

Aqui se analiza la mascara activa completa en 3D:

```text
active_3d[z, y, x]
```

Se buscan grupos de voxeles activos conectados entre si. Dos voxeles pertenecen a la misma componente si estan pegados o son vecinos en el espacio 3D.

- `n_components_3d`: numero de componentes conectadas 3D.
- `largest_component_size`: tamano de la componente conectada mas grande.
- `largest_component_fraction`: proporcion de voxeles activos que pertenecen a la componente mayor.
- `component_fragmentation`: `n_components_3d / active_voxels`. Mide cuan fragmentada esta la senal.

Lectura intuitiva:

```text
Fuente coherente: pocas componentes, una grande dominante.
Ruido fragmentado: muchas componentes pequenas.
```

Ejemplo:

```text
active_voxels = 100
largest_component_size = 80
largest_component_fraction = 0.80
```

Esto indica que la mayor parte de la senal activa esta conectada.

En cambio:

```text
active_voxels = 100
n_components_3d = 40
largest_component_size = 8
```

Esto indica una senal muy fragmentada.

## Features de centroide

Para los canales con pixeles activos, se calcula el centroide espacial.

Luego se resumen sus movimientos:

- `centroid_x_std`;
- `centroid_y_std`;
- `centroid_path_length`;
- `centroid_step_mean`;
- `centroid_step_std`;
- `centroid_step_max`;
- `valid_centroid_fraction`.

Interpretacion:

- si el centroide se desplaza poco y con continuidad, hay coherencia;
- si salta mucho de un canal a otro, puede ser ruido;
- `valid_centroid_fraction` indica en que fraccion de canales se pudo calcular un centroide.

## Componentes conectados 3D

La demo tambien mira la mascara activa completa en 3D:

```python
active_3d = subcube > mean + 2 * std
```

Luego calcula componentes conectados 3D:

- `n_components_3d`;
- `largest_component_size`;
- `largest_component_fraction`;
- `component_fragmentation`.

Interpretacion:

- si hay una unica componente grande, parece una estructura continua;
- si hay muchas componentes pequenas, parece fragmentado;
- `largest_component_fraction` mide cuanto domina la componente mayor;
- `component_fragmentation` mide cuan dispersa esta la senal activa.

Esta parte es potente porque una galaxia real deberia formar una estructura relativamente coherente en el espacio 3D `(z,y,x)`.

## Etiquetado TP/FP

Para entrenar hace falta una etiqueta.

La demo usa el truth catalogue. El truth trae posiciones reales en coordenadas astronomicas:

```text
ra, dec, central_freq
```

El script las transforma a pixeles:

```python
wcs.world_to_pixel_values(ra, dec, central_freq)
```

Asi obtiene:

```text
truth_x, truth_y, truth_z
```

Para cada candidato SoFiA se calcula la distancia al truth mas cercano:

- `min_abs_dx`;
- `min_abs_dy`;
- `min_abs_dz`;
- `min_dist_3d`.

Luego hay una etiqueta diagnostica `label` segun el modo actual:

```python
MATCHING_MODE = "medium"
```

Con:

```text
medium = dx <= 20, dy <= 20, dz <= 50
```

## Por que se creo clean_label

El primer entrenamiento salia mal porque el problema tenia muchos candidatos ambiguos.

Un candidato ambiguo es uno que:

- no esta suficientemente cerca del truth para ser TP claro;
- pero tampoco esta suficientemente lejos para ser FP claro.

Por eso se creo:

```text
clean_label
```

Reglas:

- `clean_label = 1` si cumple criterio medium;
- `clean_label = 0` si `min_dist_3d > 100`;
- `clean_label = -1` si es ambiguo.

Y:

```text
is_ambiguous = clean_label == -1
```

El entrenamiento usa solo:

```text
clean_label = 0 o 1
```

Y descarta:

```text
clean_label = -1
```

Esto fue un avance importante. Antes el modelo aprendia con etiquetas muy ruidosas. Despues se entreno con ejemplos mas claros.

## Data leakage

Data leakage significa usar durante el entrenamiento informacion que en la vida real no tendrias.

Ejemplo:

```text
min_dist_3d
```

Esta columna dice cuanto se parece el candidato al truth. Pero el truth solo existe en datos de entrenamiento/simulacion. En una prediccion real no sabriamos esa distancia.

Por eso estas columnas se excluyen siempre del modelo:

- `label`;
- `clean_label`;
- `is_ambiguous`;
- `matched_truth_id`;
- `truth_row`;
- `truth_x`;
- `truth_y`;
- `truth_z`;
- `min_abs_dx`;
- `min_abs_dy`;
- `min_abs_dz`;
- `min_dist_3d`;
- `matching_mode`;
- `name`;
- `id`.

Se pueden usar para diagnostico, pero no como features del modelo.

## Experimentos del script 02

El script `02_train_candidate_classifier.py` entrena dos variantes.

### Experimento A: with_position

Usa las features normales, incluyendo coordenadas como:

- `x`;
- `y`;
- `z`.

Sirve para ver el rendimiento maximo cuando el modelo puede aprovechar posicion.

### Experimento B: no_position

Excluye:

- `x`, `y`, `z`;
- `ra`, `dec`, `freq`;
- coordenadas peak equivalentes.

Sirve para probar si el modelo aprende de la morfologia y evolucion espectral, no solo de donde esta el candidato.

Este experimento es importante para el TFM porque demuestra si las features espacio-espectrales aportan informacion fisica.

## Modelos probados

El script prueba:

- RandomForest;
- XGBoost, si esta instalado.

### RandomForest

Es un conjunto de arboles de decision. Es robusto, sencillo de explicar y funciona bien con features tabulares.

### XGBoost

Tambien usa arboles, pero los entrena de forma secuencial corrigiendo errores anteriores. Suele funcionar muy bien en datos tabulares.

En esta demo XGBoost fue el mejor modelo.

## Threshold de decision

El modelo no devuelve directamente TP/FP. Devuelve una probabilidad:

```text
probabilidad de que el candidato sea real
```

Luego se aplica un threshold:

```text
si probabilidad >= threshold -> TP predicho
si probabilidad < threshold -> FP predicho
```

El script prueba varios:

```text
0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70
```

Un threshold bajo acepta muchos candidatos:

- sube el recall;
- pero tambien suben los falsos positivos.

Un threshold alto acepta menos candidatos:

- sube la precision;
- pero puede perder TP.

Por eso no basta con entrenar un modelo. Tambien hay que elegir threshold.

## Diagnostico con el script 03

El script `03_diagnose_candidates_and_matching.py` ayuda a responder:

```text
El problema es SoFiA, el matching, las features o el modelo?
```

Resultados principales obtenidos:

```text
strict: TP=28, FP=1536
medium: TP=188, FP=1376
loose: TP=500, FP=1064
```

Interpretacion:

- `strict` es demasiado duro: solo 28 TP.
- `medium` parece un compromiso razonable para la demo.
- `loose` acepta muchos mas TP, pero tambien puede mezclar mas candidatos dudosos.

Esto explica por que se introdujo `clean_label`: para entrenar con ejemplos mas claros.

## Resultados de entrenamiento principales

Con `clean_label`, features 3D y XGBoost/RandomForest, se obtuvo:

```text
Total filas: 1564
Filas usadas para entrenamiento: 312
TP limpios: 188
FP limpios: 124
Ambiguos descartados: 1252
```

Esto significa que la mayor parte de candidatos no se usaron para entrenar porque eran ambiguos. Eso no es malo: era justamente el objetivo de limpiar etiquetas.

## Mejor resultado con XGBoost

El mejor resultado en paso 2 fue:

```text
with_position / XGBoost
ROC AUC: 0.8189
PR AUC: 0.8770
F1@0.50: 0.8034
Mejor threshold por F1: 0.20
F1 en threshold 0.20: 0.8217
```

Interpretacion:

- el modelo con XGBoost aprende bastante bien con etiquetas limpias;
- el threshold 0.20 recupera mas TP;
- el threshold 0.70 da precision mayor o igual a 0.8, pero pierde recall.

## Resultado sin posicion

Tambien se probo `no_position`.

Esto elimina coordenadas y obliga al modelo a usar features morfologicas y espectrales.

Resultado aproximado:

```text
no_position / XGBoost
ROC AUC: 0.5728
PR AUC: 0.6955
F1@0.50: 0.6721
```

Interpretacion:

- al quitar posicion baja el rendimiento;
- aun asi, hay senal en las features morfologicas/espectrales;
- las features 3D aparecen en las importancias del modelo.

## Que features aparecieron como importantes

En XGBoost con posicion aparecieron features como:

- `n_active_channels`;
- `area_max`;
- `n_components_3d`;
- `flux_density`;
- `n_pix`;
- `centroid_x_std`.

En XGBoost sin posicion aparecieron:

- `n_pix`;
- `area_max`;
- `longest_active_run_z`;
- `active_fraction`;
- `centroid_x_std`;
- `largest_component_fraction`;
- `centroid_path_length`.

Esto es una buena senal para el TFM: el modelo no esta usando solo columnas originales, tambien aprovecha features construidas desde el subcubo.

