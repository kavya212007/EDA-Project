import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("data/dataset.csv")

print("=" * 50)
print("FIRST 5 ROWS")
print("=" * 50)
print(df.head())

print("\n" + "=" * 50)
print("DATASET INFORMATION")
print("=" * 50)
print(df.info())

print("\n" + "=" * 50)
print("STATISTICAL SUMMARY")
print("=" * 50)
print(df.describe())

# -----------------------------
# Missing Values Analysis
# -----------------------------
print("\n" + "=" * 50)
print("MISSING VALUES")
print("=" * 50)
print(df.isnull().sum())

# -----------------------------
# Duplicate Records
# -----------------------------
print("\n" + "=" * 50)
print("DUPLICATE ROWS")
print("=" * 50)
print(df.duplicated().sum())

# Remove duplicates
df = df.drop_duplicates()

# -----------------------------
# Data Types
# -----------------------------
print("\n" + "=" * 50)
print("COLUMN DATA TYPES")
print("=" * 50)
print(df.dtypes)

# -----------------------------
# Numerical Columns
# -----------------------------
numeric_cols = df.select_dtypes(include=np.number).columns

# -----------------------------
# Histograms
# -----------------------------
if len(numeric_cols) > 0:
    df[numeric_cols].hist(figsize=(12, 10))
    plt.suptitle("Histograms of Numerical Features")
    plt.tight_layout()
    plt.show()

# -----------------------------
# Boxplots
# -----------------------------
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

# -----------------------------
# Correlation Heatmap
# -----------------------------
if len(numeric_cols) > 1:
    plt.figure(figsize=(10, 8))
    correlation_matrix = df[numeric_cols].corr()

    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Correlation Heatmap")
    plt.show()

    print("\n" + "=" * 50)
    print("CORRELATION MATRIX")
    print("=" * 50)
    print(correlation_matrix)

# -----------------------------
# Pair Plot
# -----------------------------
if len(numeric_cols) > 1:
    sns.pairplot(df[numeric_cols])
    plt.show()

# -----------------------------
# Key Insights
# -----------------------------
print("\n" + "=" * 50)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 50)

print("""
Insights to Look For:
1. Features with high correlation.
2. Presence of outliers in boxplots.
3. Distribution patterns in histograms.
4. Missing values that require cleaning.
5. Variables influencing each other.
""")