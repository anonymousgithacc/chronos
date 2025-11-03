# 🕒 CHRONOS: Predicting Training Time and Convergence in Deep Learning

<p align="center">
  <img src="./docs/chronos.png" alt="Chronos logo" width="60%" height="40%">
</p>
<p align="center" style="font-size: 11px;">
  [This logo was generated using DALL·E 3 by OpenAI]
</p>

[![Paper (Under Review)](https://img.shields.io/badge/Paper-MLSys%202026-lightblue)](./CHRONOS_MLSys2026.pdf)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

**CHRONOS** is an end-to-end framework that predicts both **training time** and **epochs to convergence** of deep neural networks *before* training.  
It combines analytical modeling and early training signals to provide zero-shot estimates of convergence and cost across architectures and GPUs.

---

## 🚀 Overview

Training modern neural networks is expensive. CHRONOS predicts total training time and cost *without* running full training runs.  
It integrates:
- 🔧 **Computational features** (FLOPs, memory, arithmetic intensity)
- 🧠 **Early probes** (gradient norm, NTK trace)
- ⚙️ **Hardware-aware modeling** for GPU-specific accuracy

**Average errors:**  
- < 11% for iteration-level runtime  
- < 21% for convergence prediction  
- Up to **60% improvement** over PreNeT

---

## 📦 Installation

### 🧰 From source (development mode)
```bash
git clone https://github.com/anonymousgithacc/chronos.git
cd chronos
pip install -e .
```

### 🔢 Python version
CHRONOS officially supports **Python 3.11**.

---

## 🧩 Example Usage

```python
import chronos

print("Initializing Chronos Training Time Predictor...")

predictor = chronos.TrainingTimePredictor(model=chronos.Models.VGG16,
                                          dataset=chronos.Datasets.Cifar100,
                                          optimizer=chronos.Optimizers.SGD,
                                          batch_size=32,
                                          learning_rate=0.001,
                                          precision=16,
                                          gpu=chronos.Devices.T4)

predicted_time = predictor.predict_epoch_time()
print(f"Predicted epoch time: {predicted_time:.2f} seconds")

predicted_epochs = predictor.predict_number_of_epochs()
print(f"Predicted epochs to converge: {predicted_epochs} epochs")

total_time_hours = (predicted_time * predicted_epochs) / 3600
print(f"Predicted total training time: {total_time_hours:.2f} hours")
```
---

## ✅ Example Output
```bash
Initializing Chronos Training Time Predictor...
Predictor configured for: VGG16 on T4
Optimizer: SGD, Dataset: CIFAR100
Files already downloaded and verified
Predicted epoch time: 251.29 seconds
Predicted epochs to converge: 20.955 epochs
Predicted total training time: 1.46 hours
```

---

## 📊 Extra Scripts

| Module | Purpose |
|:--------|:---------|
| `epoch_time_collection/` | Benchmarks and collects per-epoch training times |
| `epoch_time_prediction/` | Trains a per-epoch execution time model across GPUs |
| `epoch_convergence_prediction/` | Trains a epochs-to-convergence model using probe features |

---

## ⚖️ License

This project is released under the **MIT License**.

---

## 📘 Citation

If you find this work useful, please cite the CHRONOS paper once it is published.
