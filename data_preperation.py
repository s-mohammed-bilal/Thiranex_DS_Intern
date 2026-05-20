# HEART DISEASE DATA CLEANING & VISUALIZATION

# STEP 1 — Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# STEP 2 — LOAD DATASET

df = pd.read_csv("Heart_disease.csv")

# STEP 3 — VIEW DATASET

print("FIRST 5 ROWS")
print(df.head())

print("\nDATASET SHAPE")
print(df.shape)

print("\nCOLUMN NAMES")
print(df.columns)

print("\nDATA TYPES")
print(df.dtypes)

print("\nDATASET INFO")
print(df.info())

# STEP 4 — CHECK MISSING VALUES

print("\nMISSING VALUES")
print(df.isnull().sum())

# Fill missing numerical values with mean

numeric_columns = [
    'age', 'trestbps', 'chol',
    'thalach', 'oldpeak', 'ca'
]

for col in numeric_columns:
    df[col].fillna(df[col].mean(), inplace=True)

# Fill categorical columns with mode

categorical_columns = [
    'sex', 'cp', 'fbs', 'restecg',
    'exang', 'slope', 'thal', 'target'
]

for col in categorical_columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

print("\nMISSING VALUES AFTER CLEANING")
print(df.isnull().sum())

# STEP 5 — REMOVE DUPLICATES

print("\nDUPLICATE ROWS")
print(df.duplicated().sum())

df.drop_duplicates(inplace=True)

print("\nDUPLICATES REMOVED")

# STEP 6 — STATISTICAL SUMMARY

print("\nSTATISTICAL SUMMARY")
print(df.describe())

# STEP 7 — OUTLIER DETECTION

outlier_columns = [
    'age', 'trestbps', 'chol',
    'thalach', 'oldpeak'
]

for col in outlier_columns:

    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

# STEP 8 — DATA VISUALIZATION

# AGE DISTRIBUTION

plt.figure(figsize=(8,5))
plt.hist(df['age'], bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# HEART DISEASE COUNT

plt.figure(figsize=(6,5))
sns.countplot(x='target', data=df)
plt.title("Heart Disease Count")
plt.xlabel("Target")
plt.ylabel("Count")
plt.show()

# CHEST PAIN TYPE ANALYSIS

plt.figure(figsize=(8,5))
sns.countplot(x='cp', hue='target', data=df)
plt.title("Chest Pain Type vs Heart Disease")
plt.xlabel("Chest Pain Type")
plt.ylabel("Count")
plt.show()

# GENDER ANALYSIS

plt.figure(figsize=(6,5))
sns.countplot(x='sex', hue='target', data=df)
plt.title("Gender vs Heart Disease")
plt.xlabel("Sex (0 = Female, 1 = Male)")
plt.ylabel("Count")
plt.show()

# CHOLESTEROL DISTRIBUTION

plt.figure(figsize=(8,5))
sns.histplot(df['chol'], kde=True)
plt.title("Cholesterol Distribution")
plt.xlabel("Cholesterol")
plt.ylabel("Frequency")
plt.show()

# MAX HEART RATE ANALYSIS

plt.figure(figsize=(8,5))
sns.scatterplot(x='age', y='thalach', hue='target', data=df)
plt.title("Age vs Maximum Heart Rate")
plt.xlabel("Age")
plt.ylabel("Maximum Heart Rate")
plt.show()

# CORRELATION HEATMAP

plt.figure(figsize=(14,10))
correlation = df.corr()
sns.heatmap(correlation,
            annot=True,
            cmap='coolwarm',
            linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# STEP 9 — GROUPBY ANALYSIS

print("\nAVERAGE CHOLESTEROL BY HEART DISEASE")
group1 = df.groupby('target')['chol'].mean()
print(group1)
print("\nAVERAGE AGE BY HEART DISEASE")
group2 = df.groupby('target')['age'].mean()
print(group2)

# STEP 10 — SAVE CLEANED DATASET

df.to_csv("cleaned_heart_data.csv", index=False)
print("\nCLEANED DATASET SAVED SUCCESSFULLY")

# STEP 11 — FINAL INSIGHTS

print("\nPROJECT COMPLETED SUCCESSFULLY")
