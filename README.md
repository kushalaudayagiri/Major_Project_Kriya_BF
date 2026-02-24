**Bed Form Classification**

This project compares the performance of a Neural Network model trained:
1. Without feature normalization
2. With StandardScaler normalization

The objective is to analyze how normalization affects training stability and classification performance on an imbalanced dataset.

**Project files:**
before_norm.ipynb → Neural Network without normalization
after_norm.ipynb → Neural Network with StandardScaler

**Dataset Overview:**

The dataset contains hydraulic and sediment-related features used to predict:

Target Variable: Bed Form
Features:
Flow Depth
Slope
Charge
Channel Width
Particle Size

The dataset is imbalanced, with one dominant class.

**Model 1: Without Normalization (before_norm.ipynb)**

Steps Performed

1. Data loading
2. Basic preprocessing
3. Train-Test Split
4. Training MLPClassifier directly on raw feature values
5. Model evaluation using:
     a. Accuracy
     b. Precision
     c. Recall
     d. F1-Score

**Observations:**

Model trained successfully.
However, since features had different numerical ranges:
  -- Training was less stable.
  -- Some features dominated gradient updates.
Performance on minority classes was weaker.

**Model 2: With Normalization (after_norm.ipynb)**

Additional Step Added
  -- After Train-Test Split: StandardScaler()

1. Scaling applied only on training data
2. Test data transformed using the same scaler
3. Prevented data leakage

**Updated pipeline**

Load Data
   ↓
Data Cleaning
   ↓
Train-Test Split
   ↓
Feature Scaling (StandardScaler)
   ↓
Train MLPClassifier
   ↓
Evaluate Model

**Observations**

1. Improved training stability
2. Better gradient convergence
3. More balanced class prediction
4. Improved macro F1-score
5. Better balanced accuracy
