import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle

class PINN:
    """Physics-Informed Neural Network with custom loss"""
    def __init__(self, hidden_sizes=(100, 50), alpha=0.1):
        self.hidden_sizes = hidden_sizes
        self.alpha = alpha  # Weight for physics loss
        self.weights = []
        self.biases = []
        
    def _init_weights(self, input_dim):
        """Initialize network weights"""
        layers = [input_dim] + list(self.hidden_sizes) + [1]
        for i in range(len(layers) - 1):
            w = np.random.randn(layers[i], layers[i+1]) * 0.1
            b = np.zeros((1, layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def _tanh(self, x):
        return np.tanh(x)
    
    def _tanh_derivative(self, x):
        return 1 - np.tanh(x)**2
    
    def forward(self, X):
        """Forward pass"""
        self.activations = [X]
        for i in range(len(self.weights)):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            if i < len(self.weights) - 1:
                a = self._tanh(z)
            else:
                a = z  # Linear output
            self.activations.append(a)
        return self.activations[-1]
    
    def physics_loss(self, X, y_pred):
        """Physics-informed loss: enforce conservation law"""
        # Example: Permeability should be positive and bounded
        physics_penalty = np.mean(np.maximum(0, -y_pred))  # Penalize negative predictions
        
        # Add smoothness constraint (gradient penalty)
        if len(X) > 1:
            diff = np.diff(y_pred.flatten())
            smoothness_penalty = np.mean(diff**2)
        else:
            smoothness_penalty = 0
            
        return physics_penalty + 0.01 * smoothness_penalty
    
    def compute_loss(self, X, y_true):
        """Total loss = Data loss + Physics loss"""
        y_pred = self.forward(X)
        data_loss = np.mean((y_pred - y_true.reshape(-1, 1))**2)
        phys_loss = self.physics_loss(X, y_pred)
        return data_loss + self.alpha * phys_loss, data_loss, phys_loss
    
    def backward(self, X, y_true, lr=0.001):
        """Backward pass with gradient descent"""
        m = X.shape[0]
        y_pred = self.activations[-1]
        
        # Output layer gradient
        dz = 2 * (y_pred - y_true.reshape(-1, 1)) / m
        
        # Backpropagate
        for i in range(len(self.weights) - 1, -1, -1):
            dw = np.dot(self.activations[i].T, dz)
            db = np.sum(dz, axis=0, keepdims=True)
            
            if i > 0:
                da = np.dot(dz, self.weights[i].T)
                dz = da * self._tanh_derivative(self.activations[i])
            
            self.weights[i] -= lr * dw
            self.biases[i] -= lr * db
    
    def fit(self, X, y, epochs=500, lr=0.001, verbose=True):
        """Train the PINN"""
        self._init_weights(X.shape[1])
        losses = []
        
        for epoch in range(epochs):
            total_loss, data_loss, phys_loss = self.compute_loss(X, y)
            self.backward(X, y, lr)
            losses.append(total_loss)
            
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Total Loss: {total_loss:.4f}, "
                      f"Data Loss: {data_loss:.4f}, Physics Loss: {phys_loss:.4f}")
        
        return losses
    
    def predict(self, X):
        """Make predictions"""
        return self.forward(X).flatten()

# Load MAT file
print("Loading dataset...")
mat_data = scipy.io.loadmat('TRUE_PERM_64by220.mat')
var_names = [k for k in mat_data.keys() if not k.startswith('__')]
data = mat_data[var_names[0]]
print(f"Data shape: {data.shape}")

# Prepare data
X = data[:, :-1]
y = data[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize data
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

# Train PINN model
print("\nTraining PINN model with physics-informed loss...")
model = PINN(hidden_sizes=(100, 50), alpha=0.1)
losses = model.fit(X_train_scaled, y_train_scaled, epochs=500, lr=0.01, verbose=True)

# Evaluate model
print("\nEvaluating model...")
y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\nResults:")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# Save model
with open('pinn_model.pkl', 'wb') as f:
    pickle.dump({'model': model, 'scaler_X': scaler_X, 'scaler_y': scaler_y}, f)
print("\nModel saved as 'pinn_model.pkl'")

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(losses)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Total Loss (Data + Physics)')
ax1.set_title('PINN Training Loss')
ax1.grid(True)

ax2.scatter(y_test, y_pred, alpha=0.5)
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax2.set_xlabel('Actual')
ax2.set_ylabel('Predicted')
ax2.set_title('PINN: Actual vs Predicted')

plt.tight_layout()
plt.savefig('pinn_results.png')
print("Results plot saved as 'pinn_results.png'")
plt.show()
