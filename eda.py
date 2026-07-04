import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("dataset/combined_data.csv")

print("=" * 60)
print("Dataset Shape")
print(df.shape)

print("\n")
print(df.head())

print("\n")
print(df.describe())

print("\nSpam/Ham Count")
print(df["label"].value_counts())

# Create a readable label
df["Category"] = df["label"].map({0: "Ham", 1: "Spam"})

plt.figure(figsize=(6,5))
sns.countplot(data=df, x="Category")
plt.title("Spam vs Ham Emails")
plt.savefig("spam_ham_distribution.png")
plt.show()