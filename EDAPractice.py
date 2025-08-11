import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('banknotes.csv')

# Show first few rows
print("First 5 rows of data:")
print(df.head())

# Summary statistics
print("\nSummary statistics:")
print(df.describe())

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())
# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Histogram of all numeric features
df.hist(bins=10, figsize=(10, 6), color='skyblue')
plt.suptitle("Histograms of All Features", fontsize=14)
plt.tight_layout()
plt.show()

# Distribution of 'variance'
sns.histplot(df['variance'], kde=True, color='orange')
plt.title("Distribution of Variance")
plt.xlabel("Variance")
plt.ylabel("Frequency")
plt.show()

# Boxplot to detect outliers
plt.figure(figsize=(8, 5))
sns.boxplot(data=df.drop(columns='class'))
plt.title("Boxplot of Features (excluding class)")
plt.xticks(rotation=45)
plt.show()
