![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24+-F7931E?logo=scikit-learn&logoColor=white)
![sktime](https://img.shields.io/badge/sktime-0.5+-blue)
![SciPy](https://img.shields.io/badge/SciPy-1.7+-8CAAE6?logo=scipy&logoColor=white)

# Anomaly Mining for Jet Engine Diagnostics

> 📄 **[View full results & conclusions in English →](RESULTS.md)**

This project applies two distinct anomaly detection strategies to turbofan engine sensor data from NASA's C-MAPSS benchmark to identify engines approaching failure, a real-world predictive maintenance problem. A deep learning **Autoencoder** (AE) learns a compressed representation of healthy engine behaviour and flags operating cycles that deviate from it, while a statistical **Mahalanobis Distance** (MD) approach measures how far sensor readings stray from the training distribution. Both methods extract a per-engine anomaly signal which is then fed into a **TimeSeriesForestClassifier** to predict whether an engine has fewer than 40 remaining useful life (RUL) cycles, a threshold that represents the onset of imminent failure. The comparison provides insight into when a lightweight statistical baseline can match or outperform a neural network, and vice versa.

---

## Problem Statement

Aircraft engine degradation is gradual and multi-dimensional: wear and thermal stress cause subtle shifts across dozens of sensor channels before failure. Early detection (predictive maintenance) can prevent accidents and dramatically reduce unplanned downtime. This project frames the problem as **binary anomaly classification**: given a multivariate time series of engine sensor readings, determine whether the engine is in a **healthy state** (RUL ≥ 40 cycles, Class 0) or in an **anomalous / near-failure state** (RUL < 40 cycles, Class 1).

---

## Dataset

The experiments use NASA's **C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)** dataset, a widely used benchmark for prognostics and health management research.

| Property | Value |
|---|---|
| Source | [NASA Prognostics Center of Excellence](https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6) |
| Subset | FD001 (single fault mode, sea-level operating conditions) |
| Test engines | 100 |
| Raw sensors | 21 measurements + 3 operating settings |
| Fault type | High-pressure compressor degradation |

Place the downloaded files under `CMAPSSData/`:

```
CMAPSSData/
├── test_FD001.txt    # sensor readings for 100 engines
└── RUL_FD001.txt     # ground-truth remaining useful life at last cycle
```

**Features selected** after dropping the constant `EPRA` signal:

| Feature | Physical meaning |
|---|---|
| P30 | High-pressure compressor discharge pressure |
| T30 | High-pressure compressor discharge temperature |
| TGT (T50) | Turbine gas temperature |
| FF | Fuel flow (derived: throttle × static pressure, `φ × Ps30`) |

**Labels** are constructed from the RUL ground truth: **Class 1** (anomalous) if RUL < 40 cycles, **Class 0** (healthy) otherwise.

---

## Methodology

Both scripts share a common preprocessing pipeline before diverging into their respective anomaly detectors.

### Shared Preprocessing

1. **Feature selection**: four health-indicator signals retained; constant `EPRA` column dropped.
2. **Normalisation**: Min-Max scaling to [0, 1] per feature.
3. **Per-engine chronological split**: the first 70% of each engine's life is used for anomaly modelling; the final 30% for evaluation.
4. **Uniform feature length**: each engine's anomaly signal is interpolated to a fixed 50-point sequence so all 100 engines share the same input dimensionality for the downstream classifier.

> A Savitzky-Golay smoothing utility (`smooth()`) is also provided and can optionally be applied before the modelling step.

---

### `NASA_AE.py` — Autoencoder (Deep Learning)

An undercomplete autoencoder is trained independently on the healthy portion of each engine's sensor data and then scored on the full lifecycle:

```
Input (4) → Dense(10, ELU) → Dense(2, ELU)  ← bottleneck
          → Dense(10, ELU) → Output (4)
```

- **Training loss**: Mean Squared Error (MSE) reconstruction loss, Adam optimiser, 100 epochs, batch size 10.
- **Anomaly criterion**: MAE reconstruction error > 0.3 → anomaly flag.
- **Why AE?** Forces the network to learn a compact representation of normal behaviour; high reconstruction error signals a departure from that learned regime without requiring labelled anomaly examples.

---

### `NASA_MD.py` — Mahalanobis Distance (Statistical)

Mahalanobis Distance measures how many standard-deviation-equivalents a point lies from the training distribution, accounting for feature correlations. PCA (2 components) is applied first to reduce dimensionality and decorrelate features:

- **Anomaly criterion**: MD > 3 × mean training distance → anomaly flag.
- **Why MD?** Parameter-free, interpretable, and computationally lightweight. Provides a strong statistical baseline, especially effective when the healthy-state distribution is approximately Gaussian.

---

### Downstream Classification

The per-engine anomaly signal (reconstruction error or Mahalanobis distance, interpolated to 50 points) forms a feature matrix **X** of shape 100 × 50. A **TimeSeriesForestClassifier** (`sktime`) is trained on this representation to predict the binary RUL label. Evaluation uses:

- Hold-out test set (25% of engines, `train_test_split`)
- 5-fold `TimeSeriesSplit` cross-validation on the training set

---

## Results

| Method | Hold-out Accuracy | CV Mean Accuracy | CV Std |
|---|---|---|---|
| Autoencoder (AE) | — | — | — |
| Mahalanobis Distance (MD) | — | — | — |

> Run each script to populate the table. Full quantitative analysis and discussion are available in the thesis manuscript (`TFG_LauraGlez.pdf`).

---

## Key Findings

- Both methods successfully capture the characteristic degradation signature that emerges in the final 30% of an engine's operational life.
- The Autoencoder's bottleneck layer forces a 2-dimensional latent space that naturally separates healthy from anomalous operating states.
- The Mahalanobis Distance approach requires no hyperparameter tuning and no training time beyond covariance estimation, making it attractive for real-time deployment in resource-constrained environments.
- Per-engine modelling (training a separate anomaly model on each engine's early life) provides a personalised baseline that accounts for unit-to-unit manufacturing variation.

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the dataset

Download the [NASA C-MAPSS dataset](https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6) and place `test_FD001.txt` and `RUL_FD001.txt` inside a `CMAPSSData/` directory at the project root.

### 3. Run the Autoencoder approach

```bash
python NASA_AE.py
```

### 4. Run the Mahalanobis Distance approach

```bash
python NASA_MD.py
```

Both scripts print per-engine progress, hold-out accuracy, and cross-validation scores to stdout.

---

## Tech Stack

| Library | Purpose |
|---|---|
| TensorFlow / Keras | Autoencoder model |
| scikit-learn | Preprocessing, PCA, train/test split |
| sktime | TimeSeriesForestClassifier |
| SciPy | Savitzky-Golay smoothing |
| NumPy / Pandas | Data manipulation |
| Matplotlib / Seaborn | Visualisation |

---

## Reference

Bachelor's thesis — *Minería de Anomalías Aplicada al Diagnóstico de Motores a Reacción*, Escuela Politécnica de Gijón. Full manuscript: `TFG_LauraGlez.pdf`.
