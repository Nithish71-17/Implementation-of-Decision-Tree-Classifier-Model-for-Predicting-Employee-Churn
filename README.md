# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

1. Import the required libraries.

2.Upload and read the dataset.

3.Check for any null values using the isnull() function.

4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.


## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Dharshini S
RegisterNumber:  212224040074

# Employee Churn Prediction using Decision Tree Classifier

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

# -----------------------------
# Step 1: Load Dataset
# -----------------------------
data = pd.read_csv("Employee.csv")
print("Dataset shape:", data.shape)
print(data.head())

# Dataset info (like your screenshot)
print(data.info())

# -----------------------------
# Step 2: Preprocessing
# -----------------------------
# Clean column names (remove hidden spaces if any)
data.columns = data.columns.str.strip()

# One-hot encode categorical features
data = pd.get_dummies(data, columns=["Departments", "salary"], drop_first=True)

# -----------------------------
# Step 3: Split Data
# -----------------------------
X = data.drop("left", axis=1)
y = data["left"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# -----------------------------
# Step 4: Train Decision Tree
# -----------------------------
clf = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=42)
clf.fit(X_train, y_train)

# -----------------------------
# Step 5: Evaluate Model
# -----------------------------
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# -----------------------------
# Step 6: Plot Decision Tree
# -----------------------------
plt.figure(figsize=(20,10))
plot_tree(clf,
          feature_names=X.columns,
          class_names=["Stay", "Leave"],
          filled=True,
          rounded=True,
          fontsize=10)
plt.show()

*/
```

## Output:
<img width="1378" height="671" alt="Screenshot 2025-09-10 112641" src="https://github.com/user-attachments/assets/12a05bc2-b9a2-4cd1-91ca-84cabe6a0c58" />

<img width="1917" height="1003" alt="Screenshot 2025-09-10 112503" src="https://github.com/user-attachments/assets/19f2e50b-c94c-4583-b12a-fa3ca9bebe67" />

## Result:

Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
