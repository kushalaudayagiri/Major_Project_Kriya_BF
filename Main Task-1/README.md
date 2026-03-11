# Physics-Informed Neural Networks (PINNs) for Sediment Transport and Bedform Classification

## 🌊 Project Overview
This project implements two Physics-Informed Neural Network approaches for sediment transport modeling and bedform classification, combining data-driven learning with fundamental sediment transport physics.

## 📁 Implementation Variants

### 1. PINN.py - Basic Implementation
**Purpose**: Simple PINN with basic physics constraints  
**Data Source**: External dataset (`Dataset (Task-2).xlsx`)  
**Approach**: Minimal physics integration with existing data

### 2. PINN_physics_based.py - Advanced Implementation
**Purpose**: Comprehensive physics-informed modeling  
**Data Source**: Physics-based synthetic data generation  
**Approach**: Full sediment transport physics integration

---

## 🔬 PINN.py - Basic Implementation

### Dataset
- **File**: `Dataset (Task-2).xlsx`
- **Features**: 6 input features
  - `I`: Flow type indicator (-1 for flume, 1 for natural)
  - `Fr`: Froude number (flow regime)
  - `tau_b_star`: Shields parameter (dimensionless shear stress)
  - `Y_star`: Dimensionless flow depth
  - `d_star`: Dimensionless particle size
  - `G`: Sediment gradation coefficient
- **Target**: Bed form classification (6 classes after remapping)

### Model Architecture
- **Type**: 3-layer feedforward neural network
- **Hidden Layers**: 2 layers with 64 neurons each
- **Activation**: Tanh
- **Output**: 6 classes

### Physics Constraints
1. **Shields Parameter**: `τ_b* > 0.05` (minimum shear stress)
2. **Froude Number**: `Fr < 2.0` (flow regime limit)
3. **Particle Size**: `d_star > 1.0` (minimum grain size)

---

## 🌊 PINN_physics_based.py - Advanced Implementation

### Comprehensive Physics Integration

#### Core Physical Parameters
- **Shields Parameter (θ)**: Dimensionless bed shear stress
- **Particle Reynolds Number**: Flow-particle interaction
- **Froude Number**: Flow regime classification
- **Dimensionless Grain Size (D*)**: Fundamental sediment parameter

#### Advanced Physics Models
1. **Hjulström-Sundborg Relationships**
   - Critical velocity for sediment entrainment
   - Grain size dependent erosion/deposition

2. **Van Rijn Bed Form Classification**
   - Physics-based bed form prediction
   - Multi-parameter classification logic
   - Accounts for flow intensity and grain size

3. **Sediment Transport Physics**
   - Continuity and momentum conservation
   - Physical consistency constraints
   - Multi-physics loss function

### Model Architecture
- **Type**: 4-layer deep neural network
- **Architecture**: 128 → 128 → 64 → 4 neurons
- **Regularization**: Dropout (0.1)
- **Activation**: Tanh (physics-friendly)
- **Output**: 4 bed form classes

### Physics-Informed Loss Function
```python
Total_Loss = Data_Loss + λ_physics × Physics_Loss
```

**Physics Loss Components**:
1. Shields parameter constraints
2. Froude number bounds
3. Critical velocity relationships
4. Bed form transition physics
5. Flow continuity conservation

### Synthetic Data Generation
- **Physics-based sampling**: Realistic parameter ranges
- **Dependent variable calculation**: Manning's equation for shear stress
- **Noise injection**: Realistic variability
- **Sample size**: 4000 training, 1000 test samples

---

## 📊 Comparison Matrix

| Feature | PINN.py | PINN_physics_based.py |
|---------|---------|----------------------|
| **Complexity** | ✅ Simple | ⚠️ Advanced |
| **Physics Integration** | ❌ Basic (3 constraints) | ✅ Comprehensive (15+ relationships) |
| **Data Source** | ❌ External dataset required | ✅ Physics-based generation |
| **Architecture** | 3-layer (64 neurons) | 4-layer (128→64 neurons) |
| **Training Speed** | ✅ Fast | ⚠️ Moderate |
| **Generalization** | ❌ Limited to dataset | ✅ Physics-constrained |
| **Scientific Accuracy** | ❌ Minimal | ✅ High |
| **Use Case** | Learning/Prototyping | Production/Research |

---

## 🚀 Usage

### Basic Implementation
```bash
python PINN.py
```

### Advanced Physics-Based Implementation
```bash
python PINN_physics_based.py
```

---

## 📈 Output Metrics

### Both Implementations Provide:
- Training progress monitoring
- Classification reports (precision, recall, F1-score)
- Confusion matrices
- Test accuracy
- Physics loss tracking (advanced version)

### Advanced Implementation Additional Outputs:
- Physics parameter validation
- Bed form transition analysis
- Sediment transport regime classification
- Physical consistency metrics

---

## 🛠️ Dependencies

```txt
torch>=1.9.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
openpyxl>=3.0.0
```

---

## 🎯 Recommendations

### For Learning & Experimentation:
**Use PINN.py**
- Simple to understand and modify
- Quick prototyping
- Basic PINN concept demonstration

### For Research & Production:
**Use PINN_physics_based.py**
- Scientifically accurate sediment transport modeling
- Comprehensive physics integration
- Better generalization to new conditions
- Suitable for engineering applications

### Hybrid Approach:
Combine the architectural simplicity of PINN.py with selected physics components from PINN_physics_based.py for balanced complexity and performance.

---

## 🔬 Scientific Foundation

The advanced implementation incorporates established sediment transport theory:
- **Shields (1936)**: Critical shear stress theory
- **Hjulström-Sundborg**: Velocity-grain size relationships
- **Van Rijn (1984)**: Bed form classification system
- **Manning's Equation**: Flow resistance relationships

---

## 📝 Key Features

### PINN.py Features:
1. ✅ Simple physics-informed learning
2. ✅ Class imbalance handling
3. ✅ Automatic class remapping
4. ✅ Comprehensive evaluation metrics

### PINN_physics_based.py Features:
1. 🌊 Advanced sediment transport physics
2. 🔬 Synthetic data generation
3. 📊 Multi-physics loss optimization
4. 🎯 Van Rijn bed form classification
5. ⚖️ Physical consistency validation
6. 🚀 Deep architecture with regularization
