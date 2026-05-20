# HEART DISEASE PREDICTION USING MACHINE LEARNING

# STEP 1 — IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)

# STEP 2 — LOAD DATASET

df = pd.read_csv("heart.csv")

# STEP 3 — VIEW DATASET

print("FIRST 5 ROWS")
print(df.head())
print("\nDATASET SHAPE")
print(df.shape)
print("\nCOLUMN NAMES")
print(df.columns)
print("\nMISSING VALUES")
print(df.isnull().sum())

# STEP 4 — REMOVE DUPLICATES

print("\nDUPLICATES:", df.duplicated().sum())
df.drop_duplicates(inplace=True)
print("DUPLICATES REMOVED")

# STEP 5 — HANDLE MISSING VALUES

df.fillna(df.mean(numeric_only=True), inplace=True)

# STEP 6 — DATA VISUALIZATION

# HEART DISEASE COUNT

plt.figure(figsize=(6,5))
sns.countplot(x='target', data=df)
plt.title("Heart Disease Count")
plt.xlabel("Target")
plt.ylabel("Count")
plt.show()

# AGE DISTRIBUTION

plt.figure(figsize=(8,5))
plt.hist(df['age'], bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# CORRELATION HEATMAP

plt.figure(figsize=(14,10))
sns.heatmap(
    df.corr(),
    annot=True,
    cmap='coolwarm'
)
plt.title("Correlation Heatmap")
plt.show()

# STEP 7 — SPLIT FEATURES & TARGET

X = df.drop('target', axis=1)
y = df['target']

# STEP 8 — TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
print("\nTRAINING DATA SIZE:", X_train.shape)
print("TESTING DATA SIZE:", X_test.shape)

# STEP 9 — LOGISTIC REGRESSION MODEL

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_prediction = lr_model.predict(X_test)

# STEP 10 — DECISION TREE MODEL

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_prediction = dt_model.predict(X_test)

# STEP 11 — RANDOM FOREST MODEL

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf_model.fit(X_train, y_train)
rf_prediction = rf_model.predict(X_test)

# STEP 12 — ACCURACY SCORES

lr_accuracy = accuracy_score(y_test, lr_prediction)
dt_accuracy = accuracy_score(y_test, dt_prediction)
rf_accuracy = accuracy_score(y_test, rf_prediction)

print("MODEL ACCURACY")

print("Logistic Regression Accuracy:",round(lr_accuracy * 100, 2), "%")

print("Decision Tree Accuracy:",round(dt_accuracy * 100, 2), "%")

print("Random Forest Accuracy:",round(rf_accuracy * 100, 2), "%")

# STEP 13 — CONFUSION MATRIX

cm = confusion_matrix(y_test, rf_prediction)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# STEP 14 — CLASSIFICATION REPORT

print("CLASSIFICATION REPORT")
print(classification_report(y_test, rf_prediction))

# STEP 15 — ROC CURVE

rf_probability = rf_model.predict_proba(X_test)[:,1]

fpr, tpr, threshold = roc_curve(
    y_test,
    rf_probability
)

auc_score = roc_auc_score(
    y_test,
    rf_probability
)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label="AUC = " + str(round(auc_score, 2)))
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# STEP 16 — FEATURE IMPORTANCE

importance = rf_model.feature_importances_
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importance
})

feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)
print("FEATURE IMPORTANCE")
print(feature_importance)

# FEATURE IMPORTANCE GRAPH

plt.figure(figsize=(10,6))
sns.barplot(
    x='Importance',
    y='Feature',
    data=feature_importance
)
plt.title("Feature Importance")
plt.show()

# STEP 17 — SAVE CLEANED DATA

df.to_csv("cleaned_heart_data.csv", index=False)
print("\nCLEANED DATASET SAVED")
print("PROJECT COMPLETED SUCCESSFULLY")