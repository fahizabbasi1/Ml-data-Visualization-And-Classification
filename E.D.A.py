# Install required libraries if needed
# pip install matplotlib pandas seaborn scikit-learn

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris

# Load the Iris dataset
iris_dataset = load_iris()

# Create a DataFrame from the dataset
df = pd.DataFrame(iris_dataset.data, columns=iris_dataset.feature_names)
df['species'] = pd.Categorical.from_codes(iris_dataset.target, iris_dataset.target_names)

# Print first few rows (optional)
print(df.head())

# Scatter matrix using seaborn pairplot
sns.pairplot(df, hue='species', corner=True)
plt.suptitle("Scatter Matrix of Iris Dataset", y=1.02)
plt.show()
