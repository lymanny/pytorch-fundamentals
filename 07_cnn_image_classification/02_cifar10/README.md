# 🧠 CIFAR-10 Object Image Classification

A CNN image classification project using **PyTorch** to recognize 10 categories of RGB images from the CIFAR-10 dataset.

## 🎯 Project Overview

- **Dataset:** CIFAR-10
- **Classes:** 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Image Size:** 32 × 32 (RGB)
- **Training Images:** 50,000
- **Test Images:** 10,000

## 🔄 Project Workflow

**Dataset → Model → Training → Evaluation → Prediction**

## 📂 Project Files

| File | Description |
|---|---|
| `dataset.py` | Loads and prepares CIFAR-10 images |
| `model.py` | Builds the CNN architecture |
| `train.py` | Trains and saves the CNN model |
| `evaluate.py` | Evaluates model accuracy |
| `predict.py` | Predicts object categories |

## ⚙️ Training Configuration

- **Optimizer:** Adam
- **Loss Function:** CrossEntropyLoss
- **Learning Rate:** 0.001
- **Batch Size:** 64
- **Epochs:** 5

## 🛠️ Technologies

Python · PyTorch · TorchVision · Matplotlib · TensorBoard

## 📚 Reference

[TorchVision CIFAR-10 Dataset](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html)