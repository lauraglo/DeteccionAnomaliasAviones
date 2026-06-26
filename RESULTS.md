# Results & Conclusions
### Anomaly Mining for Jet Engine Diagnostics — Bachelor's Thesis Summary

> Full methodology and theoretical background available in `TFG_LauraGlez.pdf` (Spanish).  
> This document summarises the experimental results and conclusions in English.

---

## Experimental Setup

Two anomaly detection models were compared across two datasets:

| Model | Approach |
|---|---|
| **Mahalanobis Distance (MD)** | Statistical — flags cycles whose sensor readings deviate beyond 3× the mean distance from the training distribution |
| **Autoencoder (AE)** | Deep learning — flags cycles with high reconstruction error (MAE > 0.3 threshold) |

Each model processes the sensor data of an individual engine, identifies anomalous operating cycles, and feeds the resulting anomaly signal into a **TimeSeriesForestClassifier** that predicts whether the engine is healthy or near failure. Results are reported as classification accuracy using two validation strategies:
- **Train accuracy** — `accuracy_score` on a held-out test split
- **CV accuracy** — 5-fold cross-validation mean

The anomaly threshold (what % of an engine's life counts as "anomalous") was varied across experiments: **40%, 50%, 60%, 70%, 80%**.

---

## Dataset 1 — NASA C-MAPSS (Simulated)

100 simulated turbofan engines. Ground-truth RUL provided. Health signals monitored: P30, T30, TGT, FF.

### Mahalanobis Distance

| Anomaly threshold | Train accuracy | CV accuracy |
|---|---|---|
| 40% | **90%** | **87%** |
| 50% | 87% | 81% |
| 60% | 70% | 74% |
| 70% | 93% | 85% |
| 80% | 100% | 94% |

The MD model performs well across all thresholds, reaching ~90% accuracy at the practically relevant 40% threshold and near-perfect accuracy when a larger fraction of the engine life is labelled anomalous.

### Autoencoder — 100 Epochs

| Anomaly threshold | Train accuracy | CV accuracy |
|---|---|---|
| 40% | 63% | **87%** |
| 50% | 76% | 78% |
| 60% | 53% | 75% |
| 70% | 73% | 80% |
| 80% | 100% | 84% |

### Autoencoder — 200 Epochs

| Anomaly threshold | Train accuracy | CV accuracy |
|---|---|---|
| 40% | 83% | 85% |
| 50% | **87%** | 76% |
| 60% | 66% | 78% |
| 70% | 100% | 83% |
| 80% | 100% | 84% |

Doubling the training epochs improves the Autoencoder significantly (e.g., from 63% to 83% at the 40% threshold), but it still falls slightly short of the Mahalanobis Distance across most configurations. Both models converge to 100% accuracy at the 80% threshold, where the anomaly signal is strongest.

---

## Dataset 2 — Rolls-Royce (Real-World)

Real engine data classified by turbine and compressor health state (Good, Good-to-Normal, etc.). More complex: missing values, noisy readings, no ground-truth RUL.

The number of initial "normal" cycles used to build the baseline was varied: **50, 100, 200, 300**.

### Mahalanobis Distance

|Normal cycles used | 40% anomalies — Train | 40% anomalies — CV | 90% anomalies — Train | 90% anomalies — CV |
|---|---|---|---|---|
| 50 | 56% | 58% | 89% | 88% |
| 100 | 60% | 61% | **93%** | 84% |
| 200 | 53% | 55% | 86% | 90% |
| 300 | 61% | 56% | 88% | 88% |

### Autoencoder

| Configuration | 40% anomalies — Train | 40% anomalies — CV | 90% anomalies — Train | 90% anomalies — CV |
|---|---|---|---|---|
| 100 epochs | 66% | 61% | 87% | **90%** |
| 200 epochs | 64% | 58% | 86% | 90% |
| 20n–2n–20n | 50% | 60% | 90% | 85% |
| 5n–2n–20n | 61% | 58% | 87% | 89% |

At the **40% anomaly threshold** both models average ~57–63% accuracy, barely above random chance, indicating neither technique generalises successfully to the real-world dataset under that label configuration. At **90% anomalies** results improve substantially (86–93%), but this reflects an artificially easy classification problem rather than a solved one.

---

## Conclusions

1. **Primary objective achieved on simulated data.** Both Mahalanobis Distance and the Autoencoder successfully detect anomalies in the NASA C-MAPSS benchmark, correctly classifying engines as healthy or near-failure with reasonable accuracy.

2. **Mahalanobis Distance outperforms the Autoencoder on the NASA dataset.** MD achieves slightly higher accuracy across most threshold configurations, particularly at the 40% anomaly level (~90% vs ~63–83% for AE depending on epochs). Being parameter-free also makes it faster to deploy.

3. **More training epochs improve the Autoencoder.** Going from 100 to 200 epochs yields a notable improvement (e.g., +20 pp at the 40% threshold). However, over-training must be avoided to prevent the model from memorising noise.

4. **Neither technique resolves the real-world Rolls dataset.** The increased complexity of real sensor data (noise, missing values, no ground-truth RUL) makes the problem significantly harder. Results at the 40% threshold are close to random, suggesting the learned anomaly signatures do not transfer directly from simulated to real operating conditions.

5. **The anomaly threshold has a strong impact on accuracy.** Both models perform best when a large fraction of cycles is labelled anomalous (80–90%), where the classification task is inherently easier. The 40% threshold represents the operationally meaningful regime and is the most challenging.

---

## Future Work

- Explore additional techniques (e.g., LSTM networks, Isolation Forest, Local Outlier Factor) to improve results on the real-world dataset.
- Incorporate domain-specific preprocessing for the Rolls data to reduce noise and handle missing values more robustly.
- Extend the output from binary classification to **RUL estimation**, predicting the remaining useful life as a continuous value based on the density and timing of detected anomalous cycles.
