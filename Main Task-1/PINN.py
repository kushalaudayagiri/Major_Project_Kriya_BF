import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

file_path = "Dataset (Task-2).xlsx"
df = pd.read_excel(file_path)

df["I"] = 1
df.loc[0:1311, "I"] = -1

features = ["I", "Fr", "tau_b_star", "Y_star", "d_star", "G"]
X = df[features].values
y = (df["Bed form"] - 1).values

unique_classes = np.unique(y)
class_mapping = {old: new for new, old in enumerate(unique_classes)}
y = np.array([class_mapping[label] for label in y])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

class PINN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)
    
    def physics_loss(self, x):
        Fr = x[:, 1]
        tau_b = x[:, 2]
        Y_star = x[:, 3]
        d_star = x[:, 4]
        
        physics1 = torch.relu(0.05 - tau_b)
        physics2 = torch.relu(Fr - 2.0)
        physics3 = torch.relu(1.0 - d_star)
        
        return (physics1.mean() + physics2.mean() + physics3.mean())

class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weights = torch.FloatTensor(class_weights)

model = PINN(input_dim=6, hidden_dim=64, output_dim=len(np.unique(y)))
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 200
lambda_physics = 0.1

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    outputs = model(X_train)
    loss_data = criterion(outputs, y_train)
    loss_physics = model.physics_loss(X_train)
    loss = loss_data + lambda_physics * loss_physics
    
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Data Loss: {loss_data.item():.4f}, Physics Loss: {loss_physics.item():.4f}")

model.eval()
with torch.no_grad():
    y_pred = model(X_test).argmax(dim=1).numpy()
    y_test_np = y_test.numpy()
    
    print("\nClassification Report:")
    print(classification_report(y_test_np, y_pred, zero_division=0))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test_np, y_pred))
    
    accuracy = (y_pred == y_test_np).mean()
    print(f"\nTest Accuracy: {accuracy:.4f}")
