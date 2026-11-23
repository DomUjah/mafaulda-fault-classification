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
## the python code
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive
import shap

def process_csv_file(file_path):
    print(f"Attempting to read file: {file_path}")
    try:
        data = pd.read_csv(file_path, header=None)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        raise

    print("Data loaded. Shape:", data.shape)
    if data.empty:
        raise ValueError("The CSV file is empty or couldn't be read properly.")

    data.columns = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'feature7', 'feature8', 'target']
    X = data.drop('target', axis=1)
    y = data['target']

    return X, y

def process_data(X, y):
    feature_encoders = {}
    target_encoder = LabelEncoder()

    # Sample 500 rows for normalization statistics
    if len(X) >= 500:
        sample_data = X.sample(n=500, random_state=42)
    else:
        sample_data = X

    # Calculate sample statistics and normalize
    sample_mean = sample_data.mean()
    sample_std_dev = sample_data.std(ddof=1)
    X_normalized = (X - sample_mean) / sample_std_dev

    # Handle categorical features
    for column in X_normalized.columns:
        if X_normalized[column].dtype == 'object':
            feature_encoders[column] = LabelEncoder()
            X_normalized[column] = feature_encoders[column].fit_transform(X_normalized[column].astype(str))

    # Fill any remaining NaN values with column means
    X_normalized = X_normalized.fillna(X_normalized.mean())

    # Encode target variable
    y = target_encoder.fit_transform(y.astype(str))

    return X_normalized, y, feature_encoders, target_encoder

def fourier_activation(x):
    term1 = tf.sin(x)
    term2 = tf.sin(2 * x) / 2
    term3 = tf.sin(3 * x) / 3
    return term1 + term2 + term3

def create_cnn(input_shape, num_classes):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    model.add(tf.keras.layers.Dense(64, activation=fourier_activation))
    model.add(tf.keras.layers.Dense(32, activation=fourier_activation))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    return model

def plot_metrics(history, title=""):
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title(f'{title} - Accuracy vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{title} - Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def plot_roc_curve(y_true, y_pred_proba, num_classes, model_name):
    y_true_bin = tf.keras.utils.to_categorical(y_true, num_classes=num_classes)

    plt.figure(figsize=(10, 7))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'Class {i} ROC curve (area = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')
    plt.show()

def plot_ml_accuracy(model_name, train_acc, test_acc):
    plt.figure(figsize=(8, 6))
    plt.bar(['Train', 'Test'], [train_acc, test_acc], color=['blue', 'green'])
    plt.title(f'{model_name} Accuracy')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.show()

def main():
    print("Starting main function...")
    try:
        # Mount Google Drive
        drive.mount('/content/drive')
        file_path = '/content/drive/MyDrive/VIBRATIONS/VIBRATIONS_4U.csv'

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        # Process the data
        X, y = process_csv_file(file_path)
        X, y, feature_encoders, target_encoder = process_data(X, y)

        print("Data processing completed successfully.")
        print(f"Feature names: {X.columns.tolist()}")
        print(f"Target classes: {np.unique(y)}")

        # Get number of classes
        num_classes = len(np.unique(y))

        # Split the data
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        # Train and evaluate CNN
        cnn_model = create_cnn(input_shape=(X.shape[1],), num_classes=num_classes)
        cnn_model.compile(optimizer='adam',
                         loss='sparse_categorical_crossentropy',
                         metrics=['accuracy'])
        cnn_history = cnn_model.fit(X_train, y_train,
                                  epochs=50,
                                  batch_size=32,
                                  validation_data=(X_val, y_val),
                                  verbose=1)

        # Train Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)

        # Train XGBoost
        xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        xgb_model.fit(X_train, y_train)

        # Get predictions
        y_pred_cnn = np.argmax(cnn_model.predict(X_test), axis=1)
        y_pred_cnn_proba = cnn_model.predict(X_test)
        y_pred_rf = rf_model.predict(X_test)
        y_pred_xgb = xgb_model.predict(X_test)

        # Calculate accuracies
        cnn_train_acc = accuracy_score(y_train, np.argmax(cnn_model.predict(X_train), axis=1))
        cnn_test_acc = accuracy_score(y_test, y_pred_cnn)
        rf_train_acc = accuracy_score(y_train, rf_model.predict(X_train))
        rf_test_acc = accuracy_score(y_test, y_pred_rf)
        xgb_train_acc = accuracy_score(y_train, xgb_model.predict(X_train))
        xgb_test_acc = accuracy_score(y_test, y_pred_xgb)

        # Print classification reports
        print("\nCNN Classification Report:")
        print(classification_report(y_test, y_pred_cnn))
        print("\nRandom Forest Classification Report:")
        print(classification_report(y_test, y_pred_rf))
        print("\nXGBoost Classification Report:")
        print(classification_report(y_test, y_pred_xgb))

        # Plot metrics
        plot_metrics(cnn_history, "CNN")
        plot_ml_accuracy('Random Forest', rf_train_acc, rf_test_acc)
        plot_ml_accuracy('XGBoost', xgb_train_acc, xgb_test_acc)

        # Plot confusion matrices
        plot_confusion_matrix(y_test, y_pred_cnn, classes=np.unique(y), model_name="CNN")
        plot_confusion_matrix(y_test, y_pred_rf, classes=np.unique(y), model_name="Random Forest")
        plot_confusion_matrix(y_test, y_pred_xgb, classes=np.unique(y), model_name="XGBoost")

        # Plot ROC curves
        plot_roc_curve(y_test, y_pred_cnn_proba, num_classes, "CNN")

        # SHAP Analysis for Random Forest
        explainer_rf = shap.TreeExplainer(rf_model)
        shap_values_rf = explainer_rf.shap_values(X_test)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values_rf, X_test, feature_names=X.columns)

        # SHAP Analysis for XGBoost
        explainer_xgb = shap.Explainer(xgb_model)
        shap_values_xgb = explainer_xgb(X_test)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values_xgb, X_test, feature_names=X.columns)

        # Save models
        cnn_model.save('/content/drive/MyDrive/VIBRATIONS/cnn_model.h5')
        print("\nModels trained and evaluated successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
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
