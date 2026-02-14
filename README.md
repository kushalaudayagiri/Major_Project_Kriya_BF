STEP:1-->DataSet Loaded
      -->Loaded the datset(Dataset-2xlsx)
      -->This dataset contains:Flow depth
                               Slope
                               Charge
                               Channel Width
                               Particle Size
                               Bed Form(Target)
                   Total Records~2548


 STEP:2-->Data Preprocessing
       -->The dataset had 4 missing values in some feature columns.I used df=df.dropna() because 
          neural networks cannot handle Nan values.
       -->In the dataset CHARGE column is object(string),neural networkks need numeric inputs.So converted
          that column into numeric.
       -->The dataset had duplicates.So,after removing that the dataset rows are(2548-2179).this prevents bias in training.

       The processs performed in Data Preprocessing is:

                   Checked missing values
                           |
                   Removed Missing values
                           |
                   Converted CHARGE(col) to numeric
                           |
                   Removed rows if conversion created Nan
                            |
                    Removed Duplicate rows
                            |
                    Separated features and targets
                            |
                    Train-Test-Split
                            |
                     Feature Scaling


  STEP:3-->EDA
        -->Dataset information showcased
        -->Summary statistics:Count
                              Mean
                              Standard Deviation
                              Min 
                              Max
        -->Missing values
        -->Target Distribution
                bedform 
                2->2447
                3->97
                6->5
                5->3

  STEP:4-->Train a neural network
        -->Used MLP classifier,because that suits best for the dataset as it contains numerical features,has class
           imbalance and has different feature scales.
        -->The dataset is medium-sized,so taken two hidden layers that are enough to learn nonlinear patterns.
        -->ReLU activation function for fast computation and mostly used and also works better for numerical and tabular data.
           So,tha dataset is numerical i choosed ReLU.
           And the Adam is safest general_purpose choice.


  STEP:5-->Evaluating model using appropriate metrics
        -->The model was evaluated using Accuracy,Precision,Recall,F1-Score.Due to severe
           class imbalance,Balanced_accuracy and macro F1_score  were considered more appropriate
           metrics than simple Accuracy.SMOTE was applied to balance the training data. 


Final:-->PIPELINE

            Load Data
               ↓
              EDA
               ↓
            Cleaning
               ↓
        Remove rare classes
               ↓
          Train-Test Split
               ↓
            Scaling
              ↓
            SMOTE (training only)
              ↓
           Train MLP
              ↓
           Evaluate   




Model Comparison and Performance Analysis

Two neural network models were developed to analyze the impact of normalization on performance. 
The first model was trained without normalization, while the second model used StandardScaler normalization applied after train-test splitting.
Both models used identical neural network architecture and hyperparameters to ensure fair comparison.
The normalized model demonstrated improved learning stability and better class balance handling. Although overall accuracy remained similar, 
the normalized model achieved better balanced accuracy and macro F1-score, indicating improved performance on minority class prediction.
Normalization ensured that features with different numerical ranges contributed equally during training, leading to more effective gradient optimization and improved generalization capability.
Therefore, the neural network trained with normalization performed better and is considered the preferred model for this dataset.

                                                                    
                                


