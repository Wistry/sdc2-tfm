# AGENTS.md — Instrucciones para Codex

## Estilo de trabajo
- Responder breve y directo salvo que se pida detalle.
- No explorar todo el repositorio si el usuario indica una carpeta concreta.
- Antes de modificar código, identificar solo los archivos necesarios.
- Usar comandos concretos y rápidos (`rg`, `find`, `ls`) en vez de inspecciones masivas.
- No abrir ni procesar archivos grandes salvo petición explícita.
- No regenerar outputs grandes salvo que el usuario lo pida.
- Priorizar scripts pequeños, reproducibles y fáciles de ejecutar.

## Carpetas que NO deben leerse ni tocarse salvo petición explícita
- `data/`
- `results/`
- `repos/`
- `docs/`
- `notebooks/`
- `.git/`

## Archivos grandes a evitar
- `*.fits`
- `*.fits.gz`
- `*.npy`
- `*.npz`
- `*.h5`
- `*.hdf5`
- `*.pkl`
- `*.parquet`
- `*.log`
- `*.png`
- `*.jpg`
- `*.pdf`

## Proyecto actual
El proyecto trata sobre SKA SDC2: detección de galaxias HI en cubos espectrales 3D.

Representación principal:
- FITS cube: `cube[z, y, x]`
- `z`: frecuencia / eje espectral
- `y`: Dec
- `x`: RA
- valores: intensidad float

Objetivo:
- convertir candidatos detectados por SoFiA en un dataset de Ciencia de Datos
- extraer features tabulares y espacio-espectrales
- etiquetar candidatos como TP/FP usando truth catalogue
- entrenar modelos como Random Forest o XGBoost
- evitar deep learning salvo petición explícita

## Demo actual
Trabajar principalmente en:

- `demo/paso2/`

Objetivo de `demo/paso2`:
- leer catálogo SoFiA
- extraer subcubos locales de candidatos
- calcular series/features a lo largo de `z`
- etiquetar con truth catalogue
- entrenar clasificador simple
- guardar outputs pequeños en `demo/paso2/outputs/`

## Respuestas finales
- Máximo 5-8 líneas salvo que se pida explicación larga.
- Incluir comandos concretos.
- No repetir contexto innecesario.