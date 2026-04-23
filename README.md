# EEG Mental Workload Classification

Developed and compared two machine learning approaches (a **from-scratch logistic regression** model and a **1D Convolutional Neural Network (CNN)**) to classify mental workload (high vs. low) from 62-channel EEG recordings.


---

## Overview

Mental workload estimation from EEG signals is a key challenge in brain–computer interfaces and cognitive neuroscience. Raw EEG data is high-dimensional (62 electrodes × 512 time points = 31,744 features per sample), making direct classification impractical for small datasets.

This project addresses that by applying **FFT-based spectral feature extraction** across five standard EEG frequency bands, compressing each sample down to **310 features** while preserving the most discriminative neural information. Two classifiers are then trained and evaluated using **5-fold cross-validation**:

| Model | Mean Accuracy | Std Dev |
|---|---|---|
| Logistic Regression (from scratch) | **91.39%** | 5.72% |
| 1D CNN (TensorFlow/Keras) | **93.89%** | 8.45% |

---

## Dataset

- **Source:** `WLDataCW.mat` (MATLAB format)
- **Dimensions:** 62 electrodes × 512 time points × 360 samples
- **Sampling rate:** 256 Hz (2-second recordings)
- **Labels:** Binary — 0 (low workload) / 1 (high workload)

> **Note:** The dataset is not included in this repository due to its file size.

---

## Feature Extraction Pipeline

Each sample is transformed through the following steps:

1. **Segment**: Split each electrode's 512-point signal into two 1-second segments (256 points each)
2. **FFT**: Apply Fast Fourier Transform to each segment to obtain the power spectrum
3. **Band Power Averaging**: Compute mean power within five standard EEG frequency bands:
   - Delta (1–4 Hz)
   - Theta (4–8 Hz)
   - Alpha (8–13 Hz)
   - Beta (13–30 Hz)
   - Gamma (30–128 Hz)
4. **Average** across both segments → 5 features per electrode
5. **Result:** 62 electrodes × 5 bands = **310 features per sample**

---

## Models

### Logistic Regression (From Scratch)

A fully custom implementation using only NumPy — no ML frameworks. Includes:

- Sigmoid activation with numerical stability clipping
- Forward propagation and cross-entropy loss computation
- Manual gradient calculation and parameter updates via gradient descent
- One-hot encoded output (2-class softmax-style)

### 1D Convolutional Neural Network

Built with TensorFlow/Keras. The best-performing architecture uses:

- Input shape: (62, 5) — electrodes × frequency bands
- Two Conv1D layers (64 and 128 filters) with ReLU activation
- BatchNormalization and MaxPooling1D
- Dropout (0.5) for regularisation
- Dense output layer with softmax activation
- Adam optimiser (learning rate = 0.001)
- EarlyStopping callback to prevent overfitting

Four CNN configurations were tested, varying filter counts (32/64 vs. 64/128), learning rate, and dropout rate.

---

## Project Structure

```
eeg-workload-classification/
│
├── logistic_model.py          # Logistic regression classifier
├── deep_learning_model.py     # 1D CNN classifier (TensorFlow/Keras)
├── README.md
├── .gitignore
├── requirements.txt
└── WLDataCW.mat               # Dataset (not included — place here to run)
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
git clone https://github.com/hameedsoyemi/eeg-workload-classification.git
cd eeg-workload-classification
pip install -r requirements.txt
```

### Running the Models

Place `WLDataCW.mat` in the project root directory, then:

```bash
# Run the logistic regression model
python logistic_model.py

# Run the 1D CNN model
python deep_learning_model.py
```

Both scripts will output fold-by-fold accuracy and the overall mean accuracy.

---

## Requirements

```
numpy
scipy
tensorflow>=2.10
matplotlib
```

---

## Key Findings

- FFT-based spectral feature extraction effectively reduced dimensionality by **99%** (31,744 → 310) while retaining discriminative information.
- The logistic model achieved competitive performance (91.39%) despite having no hidden layers, demonstrating the value of well-engineered features.
- The best CNN configuration (64/128 filters, LR=0.001, dropout=0.5) achieved 93.89% accuracy, showing that deeper architectures can capture additional patterns.
- Increasing convolutional filter count had the most significant positive impact on CNN performance.
- Both models consistently struggled on Fold 3, suggesting inherently difficult samples in that partition (likely due to subject variability or recording noise).
- For small EEG datasets, strong feature engineering can make simpler models competitive with deep learning approaches.

---

## Technologies

Python · NumPy · SciPy · TensorFlow / Keras · FFT · Matplotlib

---

## Author

**Hameed Soyemi**

---

## Licence

This project is released under the [MIT Licence](https://opensource.org/licenses/MIT).
