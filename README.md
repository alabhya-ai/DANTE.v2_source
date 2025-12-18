# DANTE.v2: Scenario-Specific Insider Threat Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)

Official implementation of **"DANTE.v2: A Scenario-Specific Breakdown of an LSTM-based Approach to Predict Insider Threat on Daily System Logs"**

**Accepted at:** 90th Mississippi Academy of Sciences Annual Meeting (March 19-20, 2026)

**Authors:** Alabhya Pahari, Aryan Adhikari  
**Affiliation:** University of Southern Mississippi

---

## 🎯 Overview

DANTE.v2 is an optimized LSTM-based deep learning system for detecting insider threats in organizational system logs. Unlike existing approaches that report only aggregate performance, we provide the **first comprehensive scenario-specific evaluation** on the CMU CERT r5.2 dataset.

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

**Global Accuracy:** 97.5% (Baseline F1: 0.35 on general test set)

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

### Hyperparameters

| Component | Parameter | Value |
|-----------|-----------|-------|
| **LSTM Encoder** |
| | Embedding Dimension | 40 |
| | Hidden Size | 40 |
| | Number of Layers | 3 |
| | Dropout Rate | 0.5 |
| | Vocabulary Size | 251 |
| **CNN Classifier** |
| | Conv1 Filters | 32 (kernel: 5×5) |
| | Conv2 Filters | 64 (kernel: 5×5) |
| | Pooling | MaxPool 2×2 |
| **Training** |
| | Batch Size | 64 |
| | Learning Rate | 1×10⁻⁴ |
| | Optimizer | Adam |
| | LSTM Pre-training Epochs | 2 |
| | CNN Training Epochs | 10 |
| | Train/Test Split | 80/20 (Stratified) |

---

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
2. Extract and place in `./data/cert-r5.2/`
3. Run preprocessing:

```bash
python extractXY.ipynb  # Converts raw logs to action sequences
python train_test_splitter.py  # Creates stratified train/test split
```

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

Training takes approximately:
- **LSTM Encoder:** ~30 minutes (2 epochs, NVIDIA RTX 4060)
- **CNN Classifier:** ~2 hours (10 epochs, NVIDIA RTX 4060)

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

## 📁 Repository Structure

```
DANTE.v2_source/
│
├── model.py                    # Model architecture definitions
│   ├── LSTM_Encoder           # Sequence encoder (frozen during classification)
│   ├── CNN_Classifier         # Malicious activity classifier
│   └── InsiderClassifier      # Combined model wrapper
│
├── extractXY.ipynb            # Raw log preprocessing pipeline
├── train_test_splitter.py     # Stratified dataset splitting
├── train.ipynb                # Training scripts (encoder + classifier)
├── evaluate.ipynb             # Evaluation and metrics computation
│
├── data/                      # (User must download CERT r5.2 dataset)
├── checkpoints/               # Saved model weights
└── results/                   # Evaluation outputs (confusion matrices, metrics)
```

---

## 🔬 Reproducing Paper Results

To reproduce the scenario-specific F1-scores reported in Table I:

1. **Prepare Scenario-Specific Test Sets:**
   ```python
   # Filter test data by scenario (based on insiders.csv metadata)
   # Save as X_test_s1.pkl, y_test_s1.pkl, etc.
   ```

2. **Evaluate Each Scenario:**
   ```bash
   for scenario in {1..4}; do
       python evaluate.ipynb --test_X X_test_s${scenario}.pkl \
                             --test_y y_test_s${scenario}.pkl \
                             --output results_scenario${scenario}.json
   done
   ```

3. **Aggregate Results:**
   ```python
   # Compile metrics from results_scenario*.json into Table I format
   ```

---

## 💡 Key Design Decisions

### Why LSTM over Transformers?

While Transformers excel at capturing global dependencies, our work prioritizes:

1. **Inference Speed:** LSTMs process sequences in O(n) time vs. O(n²) for Transformers
2. **Resource Efficiency:** Critical for real-time deployment on edge devices
3. **Sufficient Performance:** LSTM hidden states effectively capture the temporal patterns in system logs

### Weighted Loss Function

Standard cross-entropy fails on imbalanced datasets (<1% malicious). We assign a 49:1 weight ratio, forcing the model to prioritize detecting rare insider threats over benign activity.

### Extended Sequence Windows

Original DANTE: 50-200 actions/day  
DANTE.v2: **10-250 actions/day**

This expansion captures both stealthy low-volume probes and high-intensity exfiltration events, particularly critical for Scenarios 2 and 4 (extended attacks).

---

## 📝 Citation

If you use this code or reference our work, please cite:

```bibtex
@inproceedings{pahari2025dante,
  title={DANTE.v2: A Scenario-Specific Breakdown of an LSTM-based Approach to Predict Insider Threat on Daily System Logs},
  author={Pahari, Alabhya and Adhikari, Aryan},
  booktitle={Proceedings of the 90th Mississippi Academy of Sciences Annual Meeting},
  year={2025}
}
```

---

## 🛠️ Troubleshooting

**Issue:** `RuntimeError: CUDA out of memory`  
**Solution:** Reduce `BATCH_SIZE` in `train.ipynb` (try 32 or 16)

**Issue:** `FileNotFoundError: X_train.pkl not found`  
**Solution:** Ensure you've run `extractXY.ipynb` and `train_test_splitter.py` first

**Issue:** Low F1-scores during training  
**Solution:** Verify class weights are set correctly in `train_cnn_classifier()` (should be [1.0, 49.0])

---

## 📧 Contact

**Alabhya Pahari** - Alabhya.Pahari@usm.edu  
**Aryan Adhikari** - AryanAdhikari@usm.edu

**Advisor:** Dr. Nick Rahimi - nick.rahimi@usm.edu  
**Institution:** School of Computing Sciences & Computer Engineering, University of Southern Mississippi

---

## 🙏 Acknowledgments

This work builds upon the original DANTE architecture by Ma et al. (2020). We thank the CMU CERT team for providing the benchmark insider threat dataset.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🔗 Related Work

- [Original DANTE Paper](https://ieeexplore.ieee.org/document/9152693) (Ma et al., 2020)
- [CMU CERT Dataset](https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247)
- [DeepLog: Anomaly Detection from System Logs](https://dl.acm.org/doi/10.1145/3133956.3134015) (Du et al., 2017)

---

**Last Updated:** December 18, 2025  
**Status:** ✅ Accepted
