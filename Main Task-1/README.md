# Main Task-1: PINN + XGBoost Hybrid Project

## 1) Project Summary
This folder implements `PINN_Fixed_Hybrid.py`, a production-style hybrid ML pipeline that combines:
- A Physics-Informed Neural Network (PINN) using PyTorch
- An XGBoost classifier
- A weighted ensemble to improve robustness

Goal: classify river bed forms from `Dataset (Task-2).xlsx`, preserving minority classes and adding physics constraints.

---

## 2) Project Structure (IT Standards)
```
Main Task-1/
  ├─ PINN_Fixed_Hybrid.py   # Main hybrid training/evaluation code
  ├─ Dataset (Task-2).xlsx  # Input dataset (keep local)
  └─ README.md             # This documentation
```

### Coding & design standards used
- Single-file end-to-end pipeline for quick research reproducibility.
- Modular helper functions with clear responsibilities.
- Explicit seed-setting for reproducibility.
- Logging/prints for dataset, training progress, and metrics.
- Class imbalance-safe split and class-weight handling.

---

## 3) How the code works
### 3.1 Data preprocessing
1. Load dataset from Excel.
2. Add synthetic feature `I` with values -1 (first 1312 rows) and 1 (remaining rows).
3. Use selected features: `I`, `Fr`, `tau_b_star`, `Y_star`, `d_star`, `G`.
4. Normalize feature values with `StandardScaler`.
5. Convert labels from `Bed form` to zero-based numeric classes.

### 3.2 Minority-safe split
`safe_train_test_split()` does:
- Identify single-sample classes.
- Keep singleton class instances in training (to prevent stratify errors).
- Stratified split for multi-sample classes where possible.

### 3.3 Physics-informed neural network (PINN)
- `PINN(nn.Module)` architecture: Linear-Tanh-Dropout-Linear-Tanh-Dropout-Linear.
- Additional physics loss constraints (ReLU penalties):
  - `tau_b_star` >= 0.05
  - `Fr` <= 2.0
  - `d_star` >= 1.0
  - `Y_star` >= 0.1
- Total loss: cross-entropy + lambda * physics_loss.

### 3.4 Enhanced XGBoost
- Uses `XGBClassifier` with regularization/hyperparameters.
- Computes sample weights from class weights to account for imbalance.

### 3.5 Weighted ensemble
- Combines PINN softmax output and XGBoost probabilities.
- Default weight: 40% PINN, 60% XGBoost.

### 3.6 Evaluation
- Evaluates each method with macro F1, accuracy, and classification report.
- Prints XGBoost feature importance ranking.
- Chooses best model by highest macro F1.

---

## 4) How to run (dev/test)
1. Open terminal in `Main Task-1`.
2. Install dependencies (example):
```bash
pip install pandas numpy torch scikit-learn xgboost openpyxl
```
3. Run:
```bash
py PINN_Fixed_Hybrid.py
```
4. Inspect logs and final model performance summary.

---

## 5) Current run results (from latest execution)
### Dataset summary
- Total samples: 2548
- Features: 6
- Classes: 4
- Class distribution: 2447, 97, 3, 1

### Safe split
- Train: 1957, 78, 2, 1
- Test: 490, 19, 1

### Model performance
| Model | Macro F1 | Weighted F1 | Accuracy | Test Support |
|---|---|---|---|---|
| Enhanced PINN | 0.9214 | 0.98 | 0.9784 | 510 |
| Enhanced XGBoost | 0.6581 | 1.00 | 0.9980 | 510 |
| Weighted Ensemble | 0.6581 | 1.00 | 0.9980 | 510 |

- Best: Enhanced PINN

### Feature importance (XGBoost)
1. `Fr` (50.25%)
2. `tau_b_star` (14.95%)
3. `d_star` (14.02%)
4. `Y_star` (13.61%)
5. `I` (7.17%)
6. `G` (0.00%)

---

## 8) Evaluation parameters (model hyperparameters)

### 8.1 Shared training settings
- Random seed: 42 (NumPy and PyTorch)
- Test size: 20% (safe stratified split)
- Label mapping: zero-based from unique classes

### 8.2 Enhanced PINN configuration
- Input dim: 6
- Hidden dim: 128
- Output dim: number of unique classes
- Architecture: Linear -> Tanh -> Dropout(0.2) -> Linear -> Tanh -> Dropout(0.2) -> Linear
- Loss: `CrossEntropyLoss(weight=class_weights)` + physics penalty
- Physics constraints:
  - `tau_b_star >= 0.05`
  - `Fr <= 2.0`
  - `d_star >= 1.0`
  - `Y_star >= 0.1`
- Physics weight (`lambda_physics`): 0.15
- Optimizer: `AdamW` (lr=0.001, weight_decay=1e-4)
- LR scheduler: `ReduceLROnPlateau` (patience=20, factor=0.7)
- Early stopping: patience=40 epochs
- Training epochs: up to 250
- Gradient clipping: max_norm=1.0

### 8.3 Enhanced XGBoost configuration
- Model: `xgb.XGBClassifier`
- n_estimators: 200
- max_depth: 6
- learning_rate: 0.1
- subsample: 0.8
- colsample_bytree: 0.8
- reg_alpha: 0.1
- reg_lambda: 0.1
- scale_pos_weight: mean sample weights
- random_state: 42
- n_jobs: -1
- eval_metric: `mlogloss`
- sample weights: based on balanced class weights from training labels

### 8.4 Ensemble settings
- Weighted combination of predicted class probabilities:
  - PINN weight: 0.4
  - XGBoost weight: 0.6

### 8.5 Evaluation metrics reported
- Macro F1 score
- Accuracy
- Classification report (precision, recall, f1-score, support)
- Feature importance ranking (XGBoost)

---

## 9) IT project standards and best practices
- Keep dataset and code in the same folder for local experiments.
- Use `requirements.txt` in future for dependency management.
- Use Azure/GitHub CI for reproducible runs and quality checks.
- For production, separate model training, evaluation, and inference into modules.

### Suggested next improvements
- Add `requirements.txt` and `venv` support.
- Add unit tests for `safe_train_test_split` and physics constraints.
- Add a script for prediction/inference and one for hyperparameter tuning.
- Log results to a CSV/JSON for experiment tracking.

- Keep dataset and code in the same folder for local experiments.
- Use `requirements.txt` in future for dependency management.
- Use Azure/GitHub CI for reproducible runs and quality checks.
- For production, separate model training, evaluation, and inference into modules.

### Suggested next improvements
- Add `requirements.txt` and `venv` support.
- Add unit tests for `safe_train_test_split` and physics constraints.
- Add a script for prediction/inference and one for hyperparameter tuning.
- Log results to a CSV/JSON for experiment tracking.

---

## 7) Quick troubleshooting
- If split fails due to single-sample classes, verify class counts before split.
- If XGBoost shows `ValueError: n_classes` mismatch, re-open and ensure the same label mapping is used across train/test.
- If model underperforms, try adjusting `lambda_physics`, hidden layers, or XGBoost hyperparameters.

