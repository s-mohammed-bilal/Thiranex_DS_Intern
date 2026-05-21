# EXPLORATORY DATA ANALYSIS (EDA)
# HEART DISEASE DATASET PROJECT

# STEP 1 — IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# STEP 2 — LOAD DATASET

df = pd.read_csv("heart.csv")

# STEP 3 — BASIC DATA EXPLORATION

print("FIRST 5 ROWS")
print(df.head())
print("LAST 5 ROWS")
print(df.tail())
print("DATASET SHAPE")
print(df.shape)
print("COLUMN NAMES")
print(df.columns)
print("DATA TYPES")
print(df.dtypes)
print("DATASET INFO")
print(df.info())

# STEP 4 — CHECK MISSING VALUES

print("MISSING VALUES")

print(df.isnull().sum())

# STEP 5 — REMOVE DUPLICATES

print("DUPLICATE VALUES")
print(df.duplicated().sum())
df.drop_duplicates(inplace=True)
print("\nDUPLICATES REMOVED")

# STEP 6 — STATISTICAL SUMMARY

print("STATISTICAL SUMMARY")
print(df.describe())

# STEP 7 — UNIQUE VALUES

print("UNIQUE VALUES")

columns = [
    'sex', 'cp', 'fbs',
    'restecg', 'exang',
    'slope', 'thal', 'target'
]

for col in columns:
    
    print(f"\n{col.upper()} VALUES:")
    print(df[col].value_counts())

# STEP 8 — UNIVARIATE ANALYSIS

# AGE DISTRIBUTION

plt.figure(figsize=(8,5))
sns.histplot(df['age'], kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# CHOLESTEROL DISTRIBUTION

plt.figure(figsize=(8,5))
sns.histplot(df['chol'], kde=True)
plt.title("Cholesterol Distribution")
plt.xlabel("Cholesterol")
plt.ylabel("Frequency")
plt.show()

# MAX HEART RATE DISTRIBUTION

plt.figure(figsize=(8,5))
sns.histplot(df['thalach'], kde=True)
plt.title("Maximum Heart Rate Distribution")
plt.xlabel("Maximum Heart Rate")
plt.ylabel("Frequency")
plt.show()

# STEP 9 — COUNT PLOTS

# HEART DISEASE COUNT

plt.figure(figsize=(6,5))
sns.countplot(x='target', data=df)
plt.title("Heart Disease Count")
plt.xlabel("Target")
plt.ylabel("Count")
plt.show()

# GENDER COUNT

plt.figure(figsize=(6,5))
sns.countplot(x='sex', data=df)
plt.title("Gender Distribution")
plt.xlabel("Sex")
plt.ylabel("Count")
plt.show()

# CHEST PAIN TYPE

plt.figure(figsize=(8,5))
sns.countplot(x='cp', data=df)
plt.title("Chest Pain Type Count")
plt.xlabel("Chest Pain Type")
plt.ylabel("Count")
plt.show()

# STEP 10 — BIVARIATE ANALYSIS

# AGE VS HEART DISEASE

plt.figure(figsize=(8,5))
sns.boxplot(x='target', y='age', data=df)
plt.title("Age vs Heart Disease")
plt.xlabel("Target")
plt.ylabel("Age")
plt.show()

# CHOLESTEROL VS HEART DISEASE

plt.figure(figsize=(8,5))
sns.boxplot(x='target', y='chol', data=df)
plt.title("Cholesterol vs Heart Disease")
plt.xlabel("Target")
plt.ylabel("Cholesterol")
plt.show()

# MAX HEART RATE VS HEART DISEASE

plt.figure(figsize=(8,5))
sns.boxplot(x='target', y='thalach', data=df)
plt.title("Maximum Heart Rate vs Heart Disease")
plt.xlabel("Target")
plt.ylabel("Maximum Heart Rate")
plt.show()

# STEP 11 — SCATTER PLOTS

# AGE VS CHOLESTEROL

plt.figure(figsize=(8,6))
sns.scatterplot(
    x='age',
    y='chol',
    hue='target',
    data=df
)
plt.title("Age vs Cholesterol")
plt.xlabel("Age")
plt.ylabel("Cholesterol")
plt.show()

# AGE VS HEART RATE

plt.figure(figsize=(8,6))
sns.scatterplot(
    x='age',
    y='thalach',
    hue='target',
    data=df
)
plt.title("Age vs Maximum Heart Rate")
plt.xlabel("Age")
plt.ylabel("Maximum Heart Rate")
plt.show()

# STEP 12 — CORRELATION ANALYSIS

correlation = df.corr()

print("CORRELATION MATRIX")
print(correlation)

# HEATMAP

plt.figure(figsize=(14,10))
sns.heatmap(
    correlation,
    annot=True,
    cmap='coolwarm',
    linewidths=0.5
)
plt.title("Correlation Heatmap")
plt.show()

# STEP 13 — OUTLIER DETECTION

outlier_columns = [
    'age',
    'trestbps',
    'chol',
    'thalach',
    'oldpeak'
]

for col in outlier_columns:
    plt.figure(figsize=(7,4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

# STEP 14 — GROUPBY ANALYSIS

print("AVERAGE AGE BASED ON HEART DISEASE")
print(df.groupby('target')['age'].mean())
print("AVERAGE CHOLESTEROL BASED ON HEART DISEASE")
print(df.groupby('target')['chol'].mean())
print("AVERAGE HEART RATE BASED ON HEART DISEASE")
print(df.groupby('target')['thalach'].mean())

# STEP 15 — FEATURE RELATIONSHIP

plt.figure(figsize=(8,5))
sns.countplot(
    x='sex',
    hue='target',
    data=df
)
plt.title("Gender vs Heart Disease")
plt.xlabel("Sex")
plt.ylabel("Count")
plt.show()

# CHEST PAIN VS HEART DISEASE

plt.figure(figsize=(8,5))
sns.countplot(
    x='cp',
    hue='target',
    data=df
)
plt.title("Chest Pain Type vs Heart Disease")
plt.xlabel("Chest Pain Type")
plt.ylabel("Count")
plt.show()

# STEP 16 — SAVE CLEANED DATASET

df.to_csv(
    "eda_cleaned_heart_data.csv",
    index=False
)
print("\nCLEANED DATASET SAVED SUCCESSFULLY")

# STEP 17 — FINAL INSIGHTS

print("EDA PROJECT COMPLETED SUCCESSFULLY")