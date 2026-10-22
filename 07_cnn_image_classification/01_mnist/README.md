# 🧠 MNIST Handwritten Digit Classification

A CNN image classification project using **PyTorch** to recognize handwritten digits (0–9) from the MNIST dataset.

## 🎯 Project Overview

- **Dataset:** MNIST
- **Classes:** 10 (digits 0–9)
- **Image Size:** 28 × 28 (Grayscale)
- **Training Images:** 60,000
- **Test Images:** 10,000

## 🔄 Project Workflow

**Dataset → Model → Training → Evaluation → Prediction**

## 📂 Project Files

| File | Description |
|---|---|
| `dataset.py` | Loads and prepares MNIST images |
| `model.py` | Builds the CNN architecture |
| `train.py` | Trains and saves the CNN model |
| `evaluate.py` | Evaluates model accuracy |
| `predict.py` | Predicts handwritten digits |

## ⚙️ Training Configuration

- **Optimizer:** Adam
- **Loss Function:** CrossEntropyLoss
- **Learning Rate:** 0.001
- **Batch Size:** 64
- **Epochs:** 3

## 🛠️ Technologies

Python · PyTorch · TorchVision · Matplotlib · TensorBoard

## 📚 Reference

[TorchVision MNIST Dataset](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html)