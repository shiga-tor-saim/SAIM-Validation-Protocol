# SAIM Engine v9.3: Spinal Active Inference Model Analysis Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/Status-Gold%20Master-green.svg)]()

**Official Implementation for Manuscript Submission.**

## 📖 Overview

The **Spinal Active Inference Model (SAIM) Engine v9.3** is a computational pipeline designed to quantify neuro-somatic reorganization following chiropractic adjustments. Based on the Free Energy Principle (Friston, 2010), it integrates multi-modal physiological signals into a unified state space to evaluate the dynamics of "Destruction and Reorganization."

This version (**v9.3**) is the **"Reviewer-Proof" Gold Master edition**, specifically engineered to eliminate analyzer bias and ensure statistical robustness for high-impact journal submission.

---

## 🚀 Key Features in v9.3

### 1. Bias Neutralization (Equal Weighting)
Unlike previous versions that allowed manual weight adjustment, v9.3 employs a strict **Equal Weighting Strategy**. The Integration Score is calculated as the mathematical mean of all available functional metrics (`FSI`, `SOM`, `AUT`, `HEMO`), preventing "cherry-picking" of favorable indicators.

### 2. Robust Statistics (Median/MAD)
To mitigate the impact of physiological artifacts (e.g., sneezing, body movement), all internal normalizations now use **Robust Z-scores** based on the Median and Median Absolute Deviation (MAD), rather than Mean and Standard Deviation.
> $Z_{robust} = 0.6745 \cdot (x - \text{median}) / \text{MAD}$

### 3. Fail-Safe Baseline Check
The engine performs a critical pre-flight check. If the baseline data (`01_Pre`) is missing, the analysis **aborts immediately** to prevent the generation of scientifically invalid comparisons.

### 4. Full Data Transparency
Generates a comprehensive `FullData` CSV containing raw calculated values for every single time window, ensuring complete reproducibility and auditability.

---

## 📊 Calculated Metrics

| Metric | Full Name | Description |
| :--- | :--- | :--- |
| **NCI** | Neural Complexity Index | Primary outcome. Global system integrity (0.0-1.0). |
| **PE** | Prediction Error | Instability of the internal model (EEG Alpha volatility). |
| **HEMO** | Neurovascular Flexibility | Metabolic resource capacity (fNIRS HbO variance). |
| **F** | Free Energy Proxy | The cost function the system seeks to minimize. |
| **FSI** | Frontal Stability Index | EEG Gamma/Delta ratio (Cognitive binding). |
| **SOM** | Micro-Kinematic Stability | Inverse of body acceleration variance. |
| **AUT** | Autonomic Flexibility | Shannon entropy of Heart Rate distribution. |

---

## 📦 Installation & Dependencies

Requires **Python 3.8+**. Install dependencies via pip:

```bash
pip install pandas numpy matplotlib seaborn scipy
