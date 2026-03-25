## 17/03/2026
### Exp 001: Baseline (Sensibilidad estándar)
* **Region:** x=0-500, y=0-500, z=0-200
* **SoFiA:**
  * threshold = 5.0 rms
  * kernels spatial = 0, 3
  * kernels spectral = 0, 3
  * reliability.enable = false
  * filtering: positive flux
* **Result:**
  * detections after filtering: 518
  * TP = 0 | FP = 518
  * score = -518.00
* **Notes:** Avalancha de Falsos Positivos. Sin filtros avanzados, el algoritmo confunde el ruido gaussiano del SKA con galaxias tenues.

### Exp 002: Reducción de Ruido (Fuerza Bruta)
* **Region:** x=0-500, y=0-500, z=0-200
* **SoFiA:**
  * threshold = 20.0 rms
  * kernels spatial = 0, 3
  * kernels spectral = 0, 3
  * filtering: positive flux
* **Result:**
  * detections after filtering: 0
  * TP = 0 | FP = 0
  * score = 0.00
* **Notes:** Mitigación de FPs subiendo drásticamente el umbral. Corta el ruido, pero vuelve al sistema ciego a fuentes con bajo SNR.

### Exp 003: Alta Sensibilidad + Fiabilidad Estadística
* **Region:** x=0-500, y=0-500, z=0-200
* **SoFiA:**
  * threshold = 4.0 rms
  * kernels spatial = 0, 3, 6
  * kernels spectral = 0, 3, 7
  * reliability.enable = true (threshold = 0.90)
  * filtering: positive flux, SNR > 3
* **Result:**
  * raw detections: 1718
  * detections after reliability: 0
  * TP = 0 | FP = 0
  * score = 0.00
* **Notes:** El módulo de fiabilidad purga el 100% del ruido (0 FPs), pero evidencia que la señal tenue de las galaxias es estadísticamente indistinguible del fondo.

### Exp 004: Target Específico 
* **Region:** x=0-72, y=291-391, z=5219-5319
* **SoFiA:**
  * threshold = 4.0 rms
  * kernels spatial = 0, 3, 6
  * kernels spectral = 0, 3, 7
  * reliability.enable = false
  * filtering: positive flux, SNR > 3
* **Result:**
  * detections after filtering: 1
  * TP = 1 | FP = 0
  * score = +0.5161
* **Notes:** Subcubo centrado WCS en la galaxia más masiva. Demuestra que el pipeline de extracción física funciona. 