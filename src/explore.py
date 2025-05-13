import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("data/creditcard.csv")
print("Dataset Shape:", data.shape)
print("\nFirst 5 Rows:\n", data.head())
print("\nClass Distribution:\n", data["Class"].value_counts(normalize=True))
print("\nSummary Statistics:\n", data.describe())

# Visualize class imbalance
sns.countplot(x="Class", data=data)
plt.title("Class Distribution (0: Non-Fraud, 1: Fraud)")
plt.savefig("figures/class_distribution.png")
plt.close()