# Physics-Informed Neural Network (PINN) for Bedform Classification

## Overview
This implementation uses a Physics-Informed Neural Network to classify river bedforms by combining data-driven learning with physical constraints from sediment transport theory.

## Dataset
- **File**: `Dataset (Task-2).xlsx`
- **Features**: 6 input features
  - `I`: Flow type indicator (-1 for flume, 1 for natural)
  - `Fr`: Froude number (flow regime)
  - `tau_b_star`: Shields parameter (dimensionless shear stress)
  - `Y_star`: Dimensionless flow depth
  - `d_star`: Dimensionless particle size
  - `G`: Sediment gradation coefficient
- **Target**: Bed form classification (6 classes after remapping)

## Model Architecture
- **Type**: 3-layer feedforward neural network
- **Input Layer**: 6 features
- **Hidden Layers**: 2 layers with 64 neurons each
- **Activation**: Tanh (smooth gradients for physics calculations)
- **Output Layer**: 6 classes (softmax via CrossEntropyLoss)

## Physics-Informed Loss
The model incorporates three physical constraints:

1. **Shields Parameter Constraint**: `τ_b* > 0.05`
   - Ensures sufficient shear stress for sediment movement

2. **Froude Number Constraint**: `Fr < 2.0`
   - Maintains subcritical to transitional flow regime typical for bedforms

3. **Particle Size Constraint**: `d_star > 1.0`
   - Ensures particles are large enough for bedform formation

**Total Loss**: `Loss = Data_Loss + λ × Physics_Loss`
- λ = 0.1 (physics weight)

## Class Imbalance Handling
- **Method**: Class weights (balanced)
- **Approach**: Automatically computes weights inversely proportional to class frequencies
- **Advantage**: Retains all classes including those with few samples
- **Implementation**: Applied in CrossEntropyLoss

## Training Configuration
- **Train/Test Split**: 80/20
- **Epochs**: 200
- **Optimizer**: Adam (lr=0.001)
- **Batch Processing**: Full batch training
- **Preprocessing**: StandardScaler normalization

## Usage
```bash
python PINN.py
```

## Output
- Training progress (every 50 epochs)
- Classification report (precision, recall, F1-score)
- Confusion matrix
- Test accuracy

## Dependencies
```
pandas
numpy
torch
scikit-learn
openpyxl
```

## Key Features
1. **Physics-informed learning**: Incorporates domain knowledge
2. **Class balancing**: Handles imbalanced datasets without data loss
3. **Automatic class remapping**: Converts to consecutive indices
4. **Comprehensive evaluation**: Multiple metrics for model assessment
