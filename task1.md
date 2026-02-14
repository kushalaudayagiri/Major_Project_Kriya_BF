1\. **Neural Network Architecture Details**



\- \*\*Input Layer\*\*: 5 features (Flow Depth, SLOPE, CHARGE, Channel Width, Particle size)

\- \*\*Hidden Layer 1\*\*: 50 neurons

\- \*\*Hidden Layer 2\*\*: 25 neurons  

\- \*\*Output Layer\*\*: 4 classes (Bed Form: 2, 3, 5, 6)

\- \*\*Activation Function\*\*: ReLU (hidden layers), Softmax (output layer)

\- \*\*Optimizer\*\*: Adam

\- \*\*Loss Function\*\*: Cross-Entropy Loss

\- \*\*Max Iterations\*\*: 1000

\- \*\*Random State\*\*: 42



2\. **Evaluation Metrics Comparison**



Metric	                     Without Normalization	With Normalization (MinMax)

Accuracy	                  96.47%	             97.84%

Precision(Weighted Avg)	           0.93                       0.97

Recall (Weighted Avg)	           0.96	                      0.98

F1-Score (Weighted Avg)	           0.95	                      0.97



3\. **Comparison table of metrics**



Model	                Accuracy

Without Normalization	0.9647 (96.47%)

With Normalization	0.9784 (97.84%)

Improvement	        +1.37%



**Weighted Average Metrics**



Metric	            Without Normalization	With Normalization	Improvement

Precision	            0.93	             0.97	          +0.04

Recall	                    0.96	             0.98	          +0.02

F1 Score	            0.95	             0.97	          +0.02



4\. **Why Normalization Improved Performance**



**Feature Scale Difference**



CHARGE: 0.13 to 1,641,478 (very large range)

Particle Size: 0.00026 to 0.177 (very small range)

Because the feature scales were very different, the model gave more importance to large-value features like CHARGE. This caused biased learning and poor prediction of minority classes.

After normalization, all features were scaled to a similar range, allowing the model to learn from every feature equally.



**Better Gradient Optimization**



Balanced gradients

Faster and more stable training

Improved convergence

With normalization, the optimizer (Adam) updated weights more efficiently. This improved overall accuracy (96.47% → 97.84%) and increased macro F1-score.



**Stable Weight Updates**



Equal contribution from all 5 features

Better minority class detection

Improved generalization

Normalization prevented any single feature from dominating the learning process.







