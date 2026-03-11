import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight


# ===============================
# Load Dataset
# ===============================

file_path = "Dataset (Task-2).xlsx"

df = pd.read_excel(file_path)

df["I"] = 1
df.loc[0:1311, "I"] = -1

features = ["I","Fr","tau_b_star","Y_star","d_star","G"]

X = df[features].values
y = df["Bed form"].values


# ===============================
# Fix label indexing automatically
# ===============================

unique_classes = np.unique(y)

class_mapping = {old:new for new,old in enumerate(unique_classes)}

y = np.array([class_mapping[i] for i in y])

num_classes = len(unique_classes)

print("Detected classes:", unique_classes)
print("Total classes:", num_classes)


# ===============================
# Physics Guided Data Augmentation
# ===============================

def physics_guided_augmentation(X,y,n_aug=3):

    X_aug=[]
    y_aug=[]

    for i in range(len(X)):

        sample=X[i]
        label=y[i]

        for _ in range(n_aug):

            new_sample=sample.copy()

            new_sample[1]*=np.random.uniform(0.9,1.1)
            new_sample[2]*=np.random.uniform(0.9,1.1)
            new_sample[3]*=np.random.uniform(0.95,1.05)
            new_sample[4]*=np.random.uniform(0.95,1.05)
            new_sample[5]*=np.random.uniform(0.95,1.05)

            tau_b=new_sample[2]
            d_star=new_sample[4]

            shields=tau_b/(1.65*9.81*(d_star+1e-6))

            if shields>0.001 and shields<10:

                X_aug.append(new_sample)
                y_aug.append(label)

    X_aug=np.array(X_aug)
    y_aug=np.array(y_aug)

    X=np.vstack((X,X_aug))
    y=np.concatenate((y,y_aug))

    return X,y


print("Original dataset size:",len(X))

X,y=physics_guided_augmentation(X,y,n_aug=3)

print("Augmented dataset size:",len(X))


# ===============================
# Train Test Split
# ===============================

X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42
)


# ===============================
# Feature Scaling
# ===============================

scaler=StandardScaler()

X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

X_train=torch.FloatTensor(X_train)
X_test=torch.FloatTensor(X_test)

y_train=torch.LongTensor(y_train)
y_test=torch.LongTensor(y_test)


# ===============================
# PINN Model
# ===============================

class PINN(nn.Module):

    def __init__(self,input_dim,hidden_dim,output_dim):

        super(PINN,self).__init__()

        self.net=nn.Sequential(

            nn.Linear(input_dim,hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim,hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim,hidden_dim//2),
            nn.Tanh(),

            nn.Linear(hidden_dim//2,output_dim)

        )

        self.g=9.81
        self.rho_w=1000
        self.rho_s=2650
        self.nu=1e-6
        self.s=self.rho_s/self.rho_w


    def forward(self,x):

        return self.net(x)


    # ===============================
    # Physics Loss
    # ===============================

    def shields_parameter(self,tau_b,d_star):

        return tau_b/((self.rho_s-self.rho_w)*self.g*(d_star+1e-6))


    def physics_loss(self,x,predictions):

        Fr=x[:,1]
        tau_b=x[:,2]
        d_star=x[:,4]

        theta=self.shields_parameter(tau_b,d_star)

        physics_losses=[]

        loss1=torch.relu(0.001-theta)
        loss2=torch.relu(theta-10)

        physics_losses.append(loss1.mean()+loss2.mean())

        loss3=torch.relu(-Fr)
        loss4=torch.relu(Fr-5)

        physics_losses.append(loss3.mean()+loss4.mean())

        bed_expected=torch.zeros_like(theta)

        ripple=(theta>=0.05)&(theta<0.5)&(Fr<0.8)
        dune=(theta>=0.1)&(theta<1.0)&(Fr>=0.3)
        upper=(theta>=0.8)|(Fr>=1.0)

        bed_expected[ripple]=1
        bed_expected[dune]=2
        bed_expected[upper]=3

        probs=torch.softmax(predictions,dim=1)

        expected_onehot=torch.zeros_like(probs)

        expected_onehot.scatter_(1,bed_expected.long().unsqueeze(1),1)

        kl_loss=nn.KLDivLoss()(torch.log(probs+1e-8),expected_onehot)

        physics_losses.append(kl_loss)

        return sum(physics_losses)


# ===============================
# Class Weights
# ===============================

class_weights=compute_class_weight(
    'balanced',
    classes=np.unique(y),
    y=y
)

class_weights=torch.FloatTensor(class_weights)


# ===============================
# Model Setup
# ===============================

model=PINN(input_dim=6,hidden_dim=128,output_dim=num_classes)

criterion=nn.CrossEntropyLoss(weight=class_weights)

optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    patience=20,
    factor=0.5
)


# ===============================
# Training
# ===============================

epochs=300
lambda_physics=0.2

for epoch in range(epochs):

    model.train()

    optimizer.zero_grad()

    outputs=model(X_train)

    data_loss=criterion(outputs,y_train)

    physics_loss=model.physics_loss(X_train,outputs)

    loss=data_loss+lambda_physics*physics_loss

    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)

    optimizer.step()

    scheduler.step(loss)

    if (epoch+1)%50==0:

        print(f"\nEpoch {epoch+1}/{epochs}")
        print("Total Loss:",loss.item())
        print("Data Loss:",data_loss.item())
        print("Physics Loss:",physics_loss.item())


# ===============================
# Evaluation
# ===============================

model.eval()

with torch.no_grad():

    y_pred=model(X_test).argmax(dim=1).numpy()
    y_true=y_test.numpy()

    print("\nClassification Report\n")

    print(classification_report(y_true,y_pred))

    print("\nConfusion Matrix\n")

    print(confusion_matrix(y_true,y_pred))

    accuracy=(y_pred==y_true).mean()

    print("\nTest Accuracy:",accuracy)