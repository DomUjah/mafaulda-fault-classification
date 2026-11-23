# Machine Learning-based Hybrid Approach for Improved Rotating Machinery Fault Classification using MAFAULDA dataset

**Source code for the paper submitted to *Computers* (MDPI), Manuscript ID: computers-4631154**

> *A Two-Stage CNN-Based Stacking Ensemble Using the MAFAULDA Benchmark Dataset*

---

## Overview

This repository contains the full Python source code for all experiments reported in the manuscript:

- **Stage 1**: 1D-CNN with the proposed Fractional Activation Function (FAF: f(x) = sin(x) + tanh(x)) for feature extraction from vibration windows.
- **Stage 2**: Independent evaluation of two ensemble classifiers — Random Forest (CNN-RF) and XGBoost (CNN-XGBoost) — trained on the 256-dimensional CNN feature vectors.

---

## Repository Structure

```
.
├── mafaulda_fault_classifier.py   # Main pipeline (Stages 1 and 2)
├── download_data.py               # Downloads MAFAULDA
├── preprocess_mafaulda.py         # Converts raw recordings to windowed CSV
├── README.md                      # This file
├── LICENSE                        # CC BY 4.0
└── data/                          # Created at runtime — not tracked by git
    ├── MAFAULDA_raw/              # Raw recordings (from download_data.py)
    └── MAFAULDA_preprocessed.csv  # Windowed feature matrix (from preprocess_mafaulda.py)
```

---

## Dataset

The experiments use the **MAFAULDA (Machinery Fault Database)** benchmark dataset:

- **Source**: Ribeiro, M.F. et al. (2016). MAFAULDA dataset.
- **Official archive**: [https://www02.smt.ufrj.br/~offshore/mfs/](https://www02.smt.ufrj.br/~offshore/mfs/page_01.html#SEC2)
- **Licence**: The dataset is distributed by its original authors under its own terms. It is **not included** in this repository. Download it using the script provided (see Step 2 below).

The dataset comprises 1,951 vibration recordings from a SpectraQuest Machinery Fault Simulator, covering 6 fault classes (Normal, Horizontal Misalignment, Vertical Misalignment, Imbalance, Overhang, Underhang) across 8 sensor channels. After windowing (1,024 samples, 50 % overlap), approximately 1,000,000 samples are produced.

---

## Requirements

```
python >= 3.9
tensorflow >= 2.10
scikit-learn >= 1.2
xgboost >= 1.7
pandas >= 1.5
numpy >= 1.23
matplotlib >= 3.6
seaborn >= 0.12
```

Install all dependencies:

```bash
pip install tensorflow scikit-learn xgboost pandas numpy matplotlib seaborn
```

In Google Colab, TensorFlow, NumPy, pandas, matplotlib and seaborn are pre-installed. Only install the extras:

```bash
!pip install xgboost scikit-learn --upgrade
```

---

## Quickstart

### Option A — Google Colab (recommended)

1. Open a new Colab notebook at [colab.research.google.com](https://colab.research.google.com).
2. Mount your Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Upload all `.py` files to your Drive or clone this repository:
   ```bash
   !git clone https://github.com/<your-username>/mafaulda-fault-classification.git
   %cd mafaulda-fault-classification
   ```
4. Download and preprocess the dataset:
   ```bash
   !python download_data.py
   !python preprocess_mafaulda.py
   ```
5. Run the full pipeline:
   ```python
   from mafaulda_fault_classifier import run_pipeline
   output = run_pipeline('data/MAFAULDA_preprocessed.csv')
   ```

### Option B — Local Python

```bash
# Clone the repository
git clone https://github.com/<your-username>/mafaulda-fault-classification.git
cd mafaulda-fault-classification

# Install dependencies
pip install tensorflow scikit-learn xgboost pandas numpy matplotlib seaborn

# Download and preprocess MAFAULDA
python download_data.py
python preprocess_mafaulda.py

# Run the pipeline
python mafaulda_fault_classifier.py
```

---

## Hyperparameters

All hyperparameters are defined at the top of `mafaulda_fault_classifier.py` and match Table X of the paper (Section 2.6 — Experimental Setup).

| Component | Hyperparameter | Value |
|---|---|---|
| Data | Window size | 1,024 samples |
| Data | Channels | 8 |
| Data | Overlap | 50 % |
| Data | Split | 70 % train / 15 % val / 15 % test |
| Data | Batch size | 64 |
| CNN | Conv1D filters | 64, 128 (kernel size 3) |
| CNN | Activation | FAF: sin(x) + tanh(x) |
| CNN | Dense layer | 256 neurons |
| CNN | Dropout | 0.50 |
| CNN | Optimizer | Adam (lr=0.001, β₁=0.9, β₂=0.999) |
| CNN | Loss | Sparse categorical cross-entropy |
| CNN | Max epochs | 100 (EarlyStopping patience=10) |
| Random Forest | Estimators | 100 |
| Random Forest | Max features | sqrt(n_features) |
| Random Forest | Criterion | Gini impurity |
| XGBoost | Estimators | 100 |
| XGBoost | Learning rate | 0.1 |
| XGBoost | Max depth | 6 |
| XGBoost | Objective | multi:softprob |
| XGBoost | Regularisation (λ) | 1 |

---

## Cross-Validation Folds

Three cross-validation folds are defined by fault-class composition:

| Fold | Classes | Description |
|---|---|---|
| Fd1 | N, H, V, I, O, U | Full 6-class |
| Fd2 | N, O, U, I | 4-class — no misalignment |
| Fd3 | H, I, N, V | 4-class — includes misalignment |

To replicate a specific fold, filter the preprocessed CSV to include only the relevant classes before calling `run_pipeline()`.

---

## Licence

The source code in this repository is released under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** licence. You are free to share and adapt the code for any purpose, provided appropriate credit is given.

See [LICENSE](LICENSE) for the full licence text.

---

## Citation

If you use this code in your research, please cite the manuscript:

> Ujah, D.A.; Okafor, K.C. *Machine Learning-based Hybrid Approach for Improved Fault Classification and Predictive Maintenance in Industry 4.0 Era*. *Computers* (MDPI), 2026. Manuscript ID: computers-4631154.

---

## Contact
Omowunmi Mary Longe
Correspondence: omowunmil@uj.ac.za 
