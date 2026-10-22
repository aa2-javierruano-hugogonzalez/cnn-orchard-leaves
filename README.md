# cnn-orchard-leaves

Clasificación de hojas de árboles frutales con redes neuronales convolucionales en PyTorch. Práctica 1 de
**Aprendizaje Automático II (curso 2026-27)**.

El repositorio contiene una CNN propia configurable, una búsqueda manual de hiperparámetros (exp01–exp05), una
búsqueda automática con Optuna (exp06–exp07) y una evaluación final con varias semillas (exp08) y transferencia de
conocimiento con EfficientNet-B0 (exp09). Se usa un subconjunto de imágenes, generado por nosotros, alojado en 
[LeafScans Orchard](https://www.kaggle.com/datasets/javierruanohdez/leafscans-orchard-balanced-fruit-tree-leaves)
de Kaggle, para facilitar la descarga. El análisis de los resultados se recoge en la memoria de la práctica.

---

## Índice

1. [Estructura del repositorio](#1-estructura-del-repositorio)
2. [Instalación](#2-instalación)
3. [Descarga del conjunto de datos](#3-descarga-del-conjunto-de-datos)
4. [Ejecución](#4-ejecución)
5. [Reproducibilidad](#5-reproducibilidad)
6. [Autores](#6-autores)

---

## 1. Estructura del repositorio

```
cnn-orchard-leaves/
├── requirements.txt                     # Dependencias del proyecto
├── experiments/
│   ├── manual/                          # Búsqueda manual de hiperparámetros
│   │   ├── exp01_baseline.py
│   │   ├── exp02_capacity.py
│   │   ├── exp03_activation.py
│   │   ├── exp04_resolution_batch.py
│   │   └── exp05_pooling_batchnorm.py
│   ├── tuning/                          # Búsqueda automática con Optuna
│   │   ├── exp06_optuna_tuning.py
│   │   └── exp07_refinement.py
│   └── final/                           # Evaluación final
│       ├── exp08_multi_seed_evaluation.py
│       └── exp09_transfer_learning.py
├── src/
│   ├── data/
│   │   ├── download_dataset.py          # Descarga del conjunto desde Kaggle
│   │   └── make_dataset.py              # Partición, preprocesado, caché, balanceo, aumentado y DataLoaders
│   ├── models/
│   │   ├── cnn.py                       # Arquitectura BaselineCNN configurable
│   │   └── train.py                     # Bucle de entrenamiento y parada anticipada
│   ├── utils/
│   │   ├── evaluation.py                # Métricas, matriz de confusión, curvas y predicciones
│   │   ├── rebuild_predictions.py       # Regenera predictions.csv de experimentos antiguos
│   │   ├── mcnemar_table.py             # Tests de McNemar por pares (tabla LaTeX)
│   │   └── check_tuning_progress.py     # Estado de un estudio Optuna
│   └── visualization/
│       ├── plot_distribution.py         # Figura de la distribución de imágenes por clase y partición
│       └── plot_optuna_study.py         # Figuras del historial de Optuna e importancia fANOVA (exp06)
├── data/                                # (no versionado) imágenes originales y tensores preprocesados
└── results/                             # (no versionado) una carpeta por experimento
```

## 2. Instalación

Requisitos: Python 3.10 o superior. Si hay una GPU con CUDA, el código la usa automáticamente.

Para entrenar en GPU, instala antes PyTorch con soporte CUDA usando el comando que indica
<https://pytorch.org/get-started/locally/>, y después ejecuta `pip install -r requirements.txt`.

```bash
git clone https://github.com/aa2-javierruano-hugogonzalez/cnn-orchard-leaves.git
cd cnn-orchard-leaves

python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows

pip install -r requirements.txt
```



## 3. Descarga del conjunto de datos

1. Crea un token de API en <https://www.kaggle.com/settings> (*API → Create Legacy Api Key*) y guarda `kaggle.json` en
   `~/.kaggle/` (Linux/macOS) o en `C:\Users\<usuario>\.kaggle\` (Windows).
2. Desde la raíz del repositorio, ejecuta:

```bash
python -m src.data.download_dataset
```

También se puede descargar el ZIP a mano desde Kaggle y descomprimirlo en `data/raw/`. Los scripts esperan esta
estructura:

```
data/raw/leafscans-orchard-compressed/<clase>/*.tif
```

Si la carpeta descomprimida tiene otro nombre, renómbrala a `leafscans-orchard-compressed`.

## 4. Ejecución

Ejecuta todos los scripts desde la raíz del repositorio y como módulos (`python -m ...`), para que funcionen las
importaciones de `src`.

```bash
# Comprobar la carga de datos (la primera vez genera la caché en data/processed/)
python -m src.data.make_dataset

# Búsqueda manual
python -m experiments.manual.exp01_baseline
python -m experiments.manual.exp02_capacity
python -m experiments.manual.exp03_activation
python -m experiments.manual.exp04_resolution_batch
python -m experiments.manual.exp05_pooling_batchnorm

# Búsqueda automática (se puede interrumpir y reanudar)
python -m experiments.tuning.exp06_optuna_tuning
python -m experiments.tuning.exp07_refinement

# Evaluación final
python -m experiments.final.exp08_multi_seed_evaluation
python -m experiments.final.exp09_transfer_learning

# Utilidades
python -m src.utils.check_tuning_progress --study exp06_optuna_tuning
python -m src.utils.rebuild_predictions
python -m src.utils.mcnemar_table > results/mcnemar_table.tex

# Figuras (se guardan en results/figures/)
python -m src.visualization.plot_distribution
python -m src.visualization.plot_optuna_study      # requiere haber ejecutado exp06
```

Los hiperparámetros de cada experimento se definen como constantes al principio de su script. Cada experimento
guarda en `results/<experimento>/`:

- los pesos del mejor modelo (`best_model.pt`);
- un resumen con hiperparámetros y métricas (`summary.json`);
- las curvas de entrenamiento y la matriz de confusión (`.pdf` y `.png`);
- las predicciones sobre test (`predictions.csv`).

Si la carpeta del experimento ya existe, el script se detiene para no sobrescribirla. Los estudios de Optuna son la
excepción: guardan su estado en `tuning.db` y continúan desde donde se quedaron.

## 5. Reproducibilidad

- Las semillas de `random`, `numpy` y `torch` se fijan en cada ejecución.
- La partición 70/15/15, estratificada por clase, usa siempre la semilla 42, así que todos los experimentos se
  evalúan sobre las mismas imágenes de test.
- En GPU puede haber pequeñas diferencias entre ejecuciones, porque algunas operaciones de cuDNN no son
  deterministas.

## 6. Autores

- Javier Ruano Hernández
- Hugo González Fernández

Aprendizaje Automático II · Curso 2026-27
