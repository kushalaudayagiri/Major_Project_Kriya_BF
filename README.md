# Major_Project_Kriya_BF
# Major Project – Bed Form Classification

## 📌 Project Overview
This project focuses on predicting **Bed Form classification** using Neural Networks.

The work is divided into two main tasks:

1️⃣ Neural Network Normalization Comparison  
2️⃣ ANN and PINN Implementation  

Dataset used: `Dataset_(Task-1).xlsx`

---

# 🔹 Task 1: Neural Network Normalization Comparison

File:
nn_normalization_comparison.py

## 🎯 Objective
To compare the performance of a Neural Network:
- Without Normalization
- With StandardScaler Normalization

## ⚙️ Model Used
MLPClassifier (Multi-Layer Perceptron)

Hidden Layers:
(32, 16)

Activation:
ReLU

Optimizer:
Adam

## 📊 Evaluation Metrics
- Accuracy
- Precision (Weighted)
- Recall (Weighted)
- F1-Score (Weighted)
- Confusion Matrix

## ✅ Conclusion
The performance of the model is compared using a final comparison table to determine whether normalization improves accuracy.

---

# 🔹 Task 2: ANN and PINN Implementation

File:
ANN&PNN_TASK_1.ipynb

## 🔵 Artificial Neural Network (ANN)

A feedforward neural network built using TensorFlow/Keras.

Architecture:
- Dense (32 neurons, ReLU)
- Dense (16 neurons, ReLU)
- Output layer (Softmax)

Loss Function:
Sparse Categorical Crossentropy

Optimizer:
Adam

Purpose:
To classify Bed Form using only data-driven learning.

---

## 🔴 Physics-Informed Neural Network (PINN)

PINN extends ANN by incorporating a physics-based constraint into the loss function.

Total Loss = Data Loss + Physics Loss

Where:
- Data Loss → Classification error
- Physics Loss → Constraint based on physical relationship between variables

Purpose:
To improve generalization by embedding domain knowledge.

---

