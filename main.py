import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,precision_score,recall_score,f1_score,balanced_accuracy_score
from imblearn.over_sampling import SMOTE



df=pd.read_excel("Dataset-2.xlsx")
print("Original Shape:",df.shape)
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())


print("Dataset Shape:",df.shape)
print("Columns:",df.columns)

print("\nMissing Values:\n",df.isnull().sum())

print("\nTarget Variable Distribution (Bed Form):")
print(df["Bed Form"].value_counts())

df=df.dropna()
print("\nShape After Removing Missing Values:",df.shape)

df["CHARGE"]=pd.to_numeric(df["CHARGE"],errors='coerce')


df=df.dropna()
print("\nData Types After Conversion:\n",df.dtypes)

df = df.drop_duplicates()
print("\nShape After Removing Duplicates:",df.shape)

df = df[df["Bed Form"].isin([2, 3])]

X=df.drop("Bed Form",axis=1)
y=df["Bed Form"]

print("\nFeature Columns:\n",X.columns)
print("\nTarget Distribution:\n",y.value_counts())

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

print("\nTraining Shape:",X_train.shape)
print("Testing Shape:",X_test.shape)

scaler=StandardScaler()
X_train_scaled=X_train.values
X_test_scaled=X_test.values

smote = SMOTE(k_neighbors=1,random_state=42)

X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)

print("Before SMOTE:", y_train.value_counts())
print("After SMOTE:", pd.Series(y_train_bal).value_counts())

#print("\nData Preprocessing Completed Successfully!")

#Without Normalization
print("\n================ TASK 2:WITHOUT NORMALIZATION ================\n")


mlp=MLPClassifier(
    hidden_layer_sizes=(64,32), 
    activation='relu',
    solver='adam',
    max_iter=1000,
    learning_rate='adaptive',
    early_stopping=True,
    random_state=42
)

mlp.fit(X_train_bal, y_train_bal)

#print("\nModel Training Completed!")

y_pred=mlp.predict(X_test_scaled)

print("\nAccuracy:",accuracy_score(y_test,y_pred))

balanced_accuracy=balanced_accuracy_score(y_test, y_pred)
print("Balanced Accuracy:", balanced_accuracy)

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test,y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test,y_pred))

f1_macro=f1_score(y_test,y_pred,average='macro')
f1_weighted=f1_score(y_test,y_pred,average='weighted')

print("Macro F1 Score:",f1_macro)
print("Weighted F1 Score:",f1_weighted)



#with Normalization

print("\n================ TASK 2:WITH NORMALIZATION ================\n")

X_train2,X_test2,y_train2,y_test2=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)


scaler2=StandardScaler()

X_train_norm=scaler2.fit_transform(X_train2)
X_test_norm=scaler2.transform(X_test2)

from imblearn.over_sampling import SMOTE

smote2 = SMOTE(k_neighbors=1,random_state=42)

X_train_norm_bal,y_train_norm_bal=smote2.fit_resample(X_train_norm,y_train2)

print("Before SMOTE:\n",y_train2.value_counts())
print("After SMOTE:\n",pd.Series(y_train_norm_bal).value_counts())



mlp_norm = MLPClassifier(
    hidden_layer_sizes=(64,32),  
    activation='relu',
    solver='adam',
    max_iter=1000,
    learning_rate='adaptive',
    early_stopping=True,
    random_state=42
)

mlp_norm.fit(X_train_norm_bal,y_train_norm_bal)

print("Normalized Model Training Completed!")


y_pred_norm=mlp_norm.predict(X_test_norm)

print("\nAccuracy:",accuracy_score(y_test2,y_pred_norm))

balanced_accuracy_norm=balanced_accuracy_score(y_test2,y_pred_norm)
print("Balanced Accuracy:",balanced_accuracy_norm)

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test2,y_pred_norm))

print("\nClassification Report:\n")
print(classification_report(y_test2,y_pred_norm,zero_division=0))

f1_macro_norm = f1_score(y_test2,y_pred_norm,average='macro')
f1_weighted_norm = f1_score(y_test2,y_pred_norm,average='weighted')

print("Macro F1 Score:",f1_macro_norm)
print("Weighted F1 Score:",f1_weighted_norm)


print("\n================ FINAL MODEL COMPARISON ================\n")

print("WITHOUT NORMALIZATION")
print("----------------------")
print("Accuracy:",accuracy_score(y_test,y_pred))
print("Balanced Accuracy:",balanced_accuracy)
print("Macro F1:",f1_macro)
print("Weighted F1:",f1_weighted)

print("\nWITH NORMALIZATION")
print("----------------------")
print("Accuracy:",accuracy_score(y_test2,y_pred_norm))
print("Balanced Accuracy:",balanced_accuracy_norm)
print("Macro F1:",f1_macro_norm)
print("Weighted F1:",f1_weighted_norm)


# Decide best model automatically
if balanced_accuracy_norm>balanced_accuracy:
    print("\n✅ Normalized Model Performs Better")
else:
    print("\n✅ Non-Normalized Model Performs Better")