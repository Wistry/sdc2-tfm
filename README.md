# SDC2 TFM

Proyecto de trabajo para reproducir y analizar pipelines del **SKA Science Data Challenge 2 (SDC2)**, con foco inicial en **SoFiA-2** y **HI-FRIENDS** como baseline reproducible, y posterior exploración de estrategias de ensemble y mejora de pipeline.

## Objetivo inmediato

Construir un baseline reproducible:

FITS cube -> SoFiA / HI-FRIENDS -> catálogo -> score local

A partir de ahí:

- entender el pipeline completo
- evaluar resultados en development dataset
- analizar posibles mejoras
- preparar una base sólida para el TFM

---

## Estructura del proyecto

```text
sdc2/
├── README.md
├── .gitignore
├── environment.yml
├── notes/
├── configs/
│   ├── sofia/
│   └── hifriends/
├── scripts/
├── repos/
│   ├── SoFiA-2/
│   └── hifriends/
├── data/
└── results/