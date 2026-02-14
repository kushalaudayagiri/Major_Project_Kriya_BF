Neural Network Classification using MLP Project Overview

This project implements a Multilayer Perceptron (MLP) neural network to solve a multi-class classification problem using a structured tabular dataset. The goal is to analyze how data preprocessing (normalization) impacts the performance of a neural network.

The work is divided into two tasks:

Task 1: MLP without feature scaling

Task 2: MLP with feature scaling (Standardization)

Dataset

Input: Numerical feature columns

Output: Target class (last column in the dataset)

Missing values are removed before training.

Dataset is randomly shuffled and split into training and testing sets.

Assumption: All features are numerical and suitable for neural network input.

Task 1 – MLP without Normalization Objective

Train an MLP classifier directly on raw feature values and evaluate its performance.

Methodology

Drop missing values

Randomly sample 40% of the dataset (to reduce size)

Split data into:

75% Training

25% Testing

Train an MLPClassifier with:

1 hidden layer (16 neurons)

ReLU activation

Adam optimizer

200 training iterations

Evaluation Metrics

Accuracy

Precision, Recall, F1-score (via Classification Report)

Key Limitation

Neural networks are sensitive to feature scale. Training on unscaled data can:

Slow convergence

Bias learning toward high-magnitude features

Reduce overall accuracy

Task 2 – MLP with Normalization Objective

Improve model performance by normalizing input features before training.

Methodology

Same dataset and train–test split as Task 1

Apply StandardScaler

Fit only on training data

Transform both training and test data

Train the same MLP architecture to ensure a fair comparison

Evaluation Metrics

Accuracy

Precision, Recall, F1-score

Why This Matters

Feature scaling:

Stabilizes gradient updates

Improves convergence

Leads to better generalization

Comparison Between Task 1 and Task 2

Task 1 and Task 2 use the same dataset, same train–test split, and identical MLP architecture, ensuring a fair comparison. The only difference is feature normalization. In Task 1, the model is trained on raw, unscaled data, which makes learning sensitive to feature magnitude and leads to unstable gradients and weaker performance. In Task 2, features are standardized using StandardScaler, allowing the neural network to converge faster, learn balanced weights, and generalize better. Any improvement in accuracy and classification metrics in Task 2 is therefore solely attributable to proper data preprocessing, not changes in model complexity or parameters.

Technologies Used

Python

Pandas

Scikit-learn

Multilayer Perceptron (MLP)

Conclusion

This project demonstrates that data preprocessing is not optional in neural networks. Even with the same architecture and hyperparameters, normalization alone can significantly improve performance.

About
No description, website, or topics provided.
Resources
 Readme
 Activity
Stars
 0 stars
Watchers
 0 watching
Forks
 0 forks
Releases
No releases published
Create a new release
Packages
No packages published
Publish your first package
Languages
Jupyter Notebook
100.0%
Footer
© 2026 
