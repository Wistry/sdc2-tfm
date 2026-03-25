# SDC2 TFM — HI Source Detection Pipeline (SKA SDC2)

> ⚠️ **Work in progress** — Baseline reproducible en desarrollo.

Repositorio de trabajo para el Trabajo de Fin de Máster (TFM) centrado en el **SKA Science Data Challenge 2 (SDC2)**, aplicando **Ciencia de Datos a pipelines de detección de fuentes HI**.

El objetivo es reproducir, analizar y mejorar pipelines existentes (SoFiA-2 / HI-FRIENDS) mediante la incorporación de técnicas de optimización, análisis de datos y machine learning ligero.

---

## 🎯 Objetivo del Proyecto

Construir un pipeline reproducible y evaluable:

`FITS cube` ➔ `Detección (SoFiA / HI-FRIENDS)` ➔ `Catálogo` ➔ `Scoring SDC2`

**Fases de análisis y mejora:**
* Analizar resultados (True Positives / False Positives / False Negatives).
* Optimizar los hiperparámetros del detector.
* Aplicar técnicas de Data Science (detección de outliers / modelos predictivos).
* Mejorar la robustez general del pipeline.

---

## 🧠 Enfoque del TFM

Este trabajo no se centra en diseñar un detector desde cero, sino en **mejorar pipelines existentes mediante técnicas de ciencia de datos**.

**Pipeline final propuesto:**
1.  **Detección base:** SoFiA-2 / HI-FRIENDS.
2.  **Feature extraction:** Generación de un catálogo enriquecido.
3.  **Modelado:** Machine learning ligero y *outlier detection*.
4.  **Filtrado físico:** Limpieza basada en propiedades astronómicas.
5.  **Optimización:** Búsqueda y ajuste automático de parámetros con Optuna.
6.  **Evaluación:** Cálculo del score oficial SDC2.

---

## 📁 Estructura del Proyecto

```text
sdc2/
├── .gitignore
├── environment.yml
├── README.md
├── configs/
│   ├── hifriends/
│   └── sofia/
│       └── test_dev.par
├── data/                    # Excluido en git (archivos pesados)
│   ├── sky_dev_truthcat_v2.txt
│   └── sky_dev_v2.fits
├── docs/
├── notes/
│   ├── experiments.md
│   └── hifriends_strategy.md
├── repos/                   # Clones de repositorios externos, necesarios descargarlos
│   ├── hi-friends_analysis/
│   ├── ska-sdc-2/
│   └── SoFiA-2/
├── results/
│   ├── master_test_cat.txt
│   └── master_test_logfile.log
└── scripts/
    └── score_eval.py
