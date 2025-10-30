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
```bash
git clone https://github.com/anonymousgithacc/chronos.git
cd chronos
pip install -r requirements.txt
```

## 🧩 Example Usage

```python
import chronos

predictor = chronos.TrainingTimePredictor(
    model=chronos.Models.ViT,
    dataset=chronos.Datasets.STL10,
    batch_size=16,
    learning_rate=0.001,
    optimizer=chronos.Optimizers.Adam,
    precision=16,
    gpu=chronos.Devices.V100,
)

epoch_time = predictor.predict_epoch_time()
epochs = predictor.predict_number_of_epochs()
total_time = (epoch_time * epochs) / 3600

print(f"Epoch time: {epoch_time:.2f}s")
print(f"Predicted epochs: {epochs}")
print(f"Total training time: {total_time:.2f}h")
```
---

The code for collecting training epoch times, predicting per-epoch execution time, and estimating convergence epochs can be found in the `epoch_time_collection`, `epoch_time_prediction`, and `epoch_convergence_prediction` directories, respectively.