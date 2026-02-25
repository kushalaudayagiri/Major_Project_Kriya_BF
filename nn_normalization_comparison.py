import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

# ==============================
# 1) LOAD DATASET
# ==============================
file_path = "Dataset_(Task-1).xlsx"   # Keep the Excel file in same folder
df = pd.read_excel(file_path)

print("\n✅ Dataset Loaded Successfully")
print("Shape:", df.shape)
print(df.head())

# ==============================
# 2) DATA PREPROCESSING
# ==============================

# Convert CHARGE column to numeric (sometimes it comes as object type)
df["CHARGE"] = pd.to_numeric(df["CHARGE"], errors="coerce")

# Drop missing values
df = df.dropna().reset_index(drop=True)

print("\n✅ After Cleaning")
print("Shape:", df.shape)

# ==============================
# 3) BASIC EDA (SUMMARY STATS)
# ==============================
print("\n📌 Summary Statistics:")
print(df.describe())

# ==============================
# 4) SPLIT FEATURES & TARGET
# ==============================
X = df.drop(columns=["Bed Form"])
y = df["Bed Form"]

# NOTE:
# Stratify not used because some classes have only 1 sample
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print("\n✅ Train-Test Split Done")
print("Train size:", X_train.shape)
print("Test size :", X_test.shape)

# ==============================
# TASK 1: WITHOUT NORMALIZATION
# ==============================

print("\n==============================")
print("TASK 1: WITHOUT NORMALIZATION")
print("==============================")

mlp_no_norm = MLPClassifier(
    hidden_layer_sizes=(32, 16),
    activation="relu",
    solver="adam",
    alpha=1e-4,
    learning_rate_init=0.001,
    max_iter=500,
    random_state=42
)

mlp_no_norm.fit(X_train, y_train)
pred1 = mlp_no_norm.predict(X_test)

# Metrics
acc1 = accuracy_score(y_test, pred1)
prec1, rec1, f11, _ = precision_recall_fscore_support(
    y_test, pred1, average="weighted", zero_division=0
)

print("\n📌 Metrics (Without Normalization):")
print("Accuracy :", acc1)
print("Precision:", prec1)
print("Recall   :", rec1)
print("F1-score :", f11)

print("\n📌 Classification Report:")
print(classification_report(y_test, pred1, zero_division=0))

print("\n📌 Confusion Matrix:")
print(confusion_matrix(y_test, pred1))


# ==============================
# TASK 2: WITH NORMALIZATION
# ==============================

print("\n==============================")
print("TASK 2: WITH NORMALIZATION")
print("==============================")

# Pipeline ensures scaler is fit only on training data
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=0.001,
        max_iter=500,
        random_state=42
    ))
])

pipe.fit(X_train, y_train)
pred2 = pipe.predict(X_test)

# Metrics
acc2 = accuracy_score(y_test, pred2)
prec2, rec2, f12, _ = precision_recall_fscore_support(
    y_test, pred2, average="weighted", zero_division=0
)

print("\n📌 Metrics (With Normalization - StandardScaler):")
print("Accuracy :", acc2)
print("Precision:", prec2)
print("Recall   :", rec2)
print("F1-score :", f12)

print("\n📌 Classification Report:")
print(classification_report(y_test, pred2, zero_division=0))

print("\n📌 Confusion Matrix:")
print(confusion_matrix(y_test, pred2))


# ==============================
# FINAL COMPARISON TABLE
# ==============================

print("\n==============================")
print("FINAL COMPARISON")
print("==============================")

comparison_table = pd.DataFrame({
    "Metric": ["Accuracy", "Precision (weighted)", "Recall (weighted)", "F1-score (weighted)"],
    "Without Normalization": [acc1, prec1, rec1, f11],
    "With StandardScaler": [acc2, prec2, rec2, f12]
})

print("\n📊 Comparison Table:")
print(comparison_table)

print("\n✅ Conclusion:")
if acc2 > acc1:
    print("Normalization improved performance. StandardScaler model performed better.")
else:
    print("Normalization did not improve performance in this case.")
