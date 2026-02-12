# DANTE.v2: Scenario-Specific Insider Threat Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)

Official implementation of **"DANTE.v2: A Scenario-Specific Breakdown of an LSTM-Based Approach to Predict Insider Threat on Daily System Logs"**

**Accepted at:** 41st Computers and Their Applications (CATA), 2026

**Authors:** Alabhya Pahari, Aryan Adhikari, Oluseyi Olukola, Nick Rahimi  
**Affiliation:** University of Southern Mississippi

---

## 🎯 Overview

DANTE.v2 is an optimized LSTM-Based deep learning system for detecting insider threats in organizational system logs. Unlike existing approaches that report only aggregate performance, we provide the **first comprehensive scenario-specific evaluation** on the CMU CERT r5.2 dataset.

### Key Contributions

- **Scenario-Specific Analysis:** First work to evaluate insider threat detection performance across all 4 CERT threat scenarios individually
- **High Detection Rates:** Achieved F1-scores of 0.71 (IP Theft) and 0.74 (Long-term Theft) despite extreme class imbalance (<1% malicious)
- **Computational Efficiency:** Demonstrated LSTM viability as a fast, resource-efficient alternative to Transformer models for real-time detection
- **Optimized Architecture:** Introduced weighted loss functions and extended sequence windows to improve minority class detection

---

## 📊 Results Summary

| Scenario | Threat Type | F1-Score | Key Characteristics |
|----------|-------------|----------|---------------------|
| Scenario 1 | Data Exfiltration | 0.55 | Sudden after-hours activity, USB usage |
| Scenario 2 | IP Theft | **0.71** | Extended disgruntlement, massive data transfer |
| Scenario 3 | IT Sabotage | 0.375 | Admin credential theft, rapid execution |
| Scenario 4 | Long-term Theft | **0.74** | Unauthorized workstation access, 3-month escalation |

---

## 🏗️ Architecture

DANTE.v2 employs a hybrid LSTM-CNN architecture optimized for sequential log analysis:

```
Daily System Logs → Action ID Encoding → LSTM Encoder (Frozen) → 
Sigmoid Activation → CNN Classifier → Binary Classification (Benign/Malicious)
```

### Components

1. **LSTM Encoder** (Pre-trained, frozen during classification):
   - 3-layer bidirectional LSTM
   - 40-dimensional embeddings and hidden states
   - Trained via unsupervised sequence reconstruction

2. **CNN Classifier** (Trained end-to-end):
   - 2D convolutions to capture spatial patterns in LSTM hidden states
   - Max-pooling for dimensionality reduction
   - Weighted Cross-Entropy loss (49:1 malicious-to-benign ratio)

## 🚀 Quick Start

### Prerequisites

```bash
Python >= 3.8
PyTorch >= 1.12
CUDA >= 11.3 (optional, for GPU acceleration)
```

### Installation

```bash
# Clone the repository
git clone https://github.com/alabhya-ai/DANTE.v2_source.git
cd DANTE.v2_source

# Install dependencies
pip install -r requirements.txt
```

### Dataset Preparation

1. Download the [CMU CERT Insider Threat Dataset r5.2](https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247)
2. Extract
3. Run additional/pre_process_data.ipynb
4. Run additional/merge_processed.ipynb
5. Run extractXY.ipynb, train_test_splitter.py files

This will generate:
- `X_train.pkl`, `y_train.pkl` (Training data)
- `X_test.pkl`, `y_test.pkl` (Test data)

### Training

**Step 1: Pre-train the LSTM Encoder**

```bash
# Open train.ipynb and run the encoder training cell:
train_lstm_encoder(EPOCHS=2, LSTM_CHECKPOINT_PATH='lstm_encoder.pt')
```

**Step 2: Train the CNN Classifier**

```bash
# Run the classifier training cell:
train_cnn_classifier(EPOCHS=10, 
                     OUTPUT_FILENAME='model.pkl', 
                     LSTM_CHECKPOINT_PATH='lstm_encoder.pt')
```

Training takes approximately: ~44 minutes
- **LSTM Encoder:** 2 epochs, NVIDIA RTX 4060
- **CNN Classifier:** 10 epochs, NVIDIA RTX 4060

### Evaluation

Evaluate the trained model on the test set:

```bash
# Open evaluate.ipynb and run:
evaluate_model(X_SUBSET_PATH='X_test.pkl',
               Y_SUBSET_PATH='y_test.pkl',
               MODEL_WEIGHTS_PATH='model.pkl',
               LSTM_CHECKPOINT_PATH='lstm_encoder.pt')
```

Output includes:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix visualization
- Inference time statistics

**Scenario-Specific Evaluation:**

To evaluate on individual scenarios, first filter the test set by scenario ID, then run:

```bash
evaluate_model(X_SUBSET_PATH='X_test_scenario1.pkl',
               Y_SUBSET_PATH='y_test_scenario1.pkl',
               ...)
```

---


## 📝 Citation

If you use this code or reference our work, please cite the work appropriately.


## 🙏 Acknowledgments

This work builds upon the original DANTE architecture by Ma et al. (2020). We thank the CMU CERT team for providing the benchmark insider threat dataset.

**Advisor:** Dr. Nick Rahimi - Nick.Rahimi@usm.edu  
**Institution:** School of Computing Sciences & Computer Engineering, University of Southern Mississippi

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🔗 Related Work

- [Original DANTE Paper](https://ieeexplore.ieee.org/document/9152693) (Ma et al., 2020)
- [CMU CERT Dataset](https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247)
- [DeepLog: Anomaly Detection from System Logs](https://dl.acm.org/doi/10.1145/3133956.3134015) (Du et al., 2017)

---

**Last Updated:** Feb 12, 2026 
