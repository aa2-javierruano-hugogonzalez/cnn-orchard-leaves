# CNN-Orchard-Leaves

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Optuna](https://img.shields.io/badge/Optuna-5C4EE5?style=for-the-badge&logo=optuna&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=plotly&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)


Clasificación de hojas de árboles frutales con redes neuronales convolucionales en PyTorch.

El repositorio contiene una CNN propia configurable, una búsqueda manual de hiperparámetros (exp01–exp05), una
búsqueda automática con Optuna (exp06–exp07) y una evaluación final con varias semillas (exp08) y transferencia de
conocimiento con EfficientNet-B0 (exp09). Se usa un subconjunto de imágenes procedente de [LeafScans-Orchard](https://zenodo.org/records/20187966) (9708 imágenes) confeccionado por nosotros mismos y alojado en [Kaggle](https://www.kaggle.com/datasets/javierruanohdez/leafscans-orchard-balanced-fruit-tree-leaves) (2138 imágenes)
para facilitar su descarga. 

La memoria de la práctica, disponible en la raíz del repositorio, recoge el desarrollo completo del trabajo: la contextualización del problema, los objetivos, los fundamentos teóricos y las arquitecturas de referencia, la metodología empleada, el diseño y desarrollo de los experimentos, así como los resultados obtenidos. Incluye, además, gráficos, tablas comparativas, análisis y discusión de resultados, y las conclusiones derivadas de los distintos experimentos, documentando el proceso seguido desde la preparación de los datos hasta la evaluación de los modelos y la comparación de las estrategias de entrenamiento.

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
|       ├── plot_letterbox_comparison.py # Figura comparativa del preprocesamiento con y sin letterbox
│       └── plot_optuna_study.py         # Figuras del historial de Optuna e importancia fANOVA (exp06)
├── project_report.pdf                   # Memoria completa del proyecto
├── data/                                # (no versionado) imágenes originales y tensores preprocesados
└── results/                             # (no versionado) una carpeta por experimento
```

## 2. Instalación

Requisitos: Python 3.10 o superior. Si existe GPU con CUDA, el código la usa automáticamente.

```bash
git clone https://github.com/aa2-javierruano-hugogonzalez/cnn-orchard-leaves.git
cd cnn-orchard-leaves

python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

Para entrenar en GPU, instala antes PyTorch con soporte CUDA usando el comando que indica
<https://pytorch.org/get-started/locally/>, y después ejecuta `pip install -r requirements.txt`.

## 3. Descarga del conjunto de datos

1. Crea una API token en <https://www.kaggle.com/settings> (*API → Create Legacy API Key*) y guarda `kaggle.json` en
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
python -m src.utils.mcnemar_table

# Figuras (se guardan en results/figures/)
python -m src.visualization.plot_distribution
python -m src.plot_letterbox_comparison
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

- [Javier Ruano Hernández](https://github.com/javierruanohdez)
- [Hugo González Fernández](https://github.com/hugogzfz)

_Aprendizaje Automático II · Curso 2026-27_
