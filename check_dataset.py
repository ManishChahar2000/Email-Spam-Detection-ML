import pandas as pd

# Load dataset
df = pd.read_csv("dataset/combined_data.csv")

print("=" * 60)
print("FIRST 5 ROWS")
print("=" * 60)
print(df.head())

print("\n" + "=" * 60)
print("COLUMNS")
print("=" * 60)
print(df.columns.tolist())

print("\n" + "=" * 60)
print("SHAPE")
print("=" * 60)
print(df.shape)

print("\n" + "=" * 60)
print("INFO")
print("=" * 60)
print(df.info())

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)
print(df.isnull().sum())

print("\n" + "=" * 60)
print("LABEL DISTRIBUTION")
print("=" * 60)

# Try to find the label column automatically
for col in df.columns:
    if col.lower() in ["label", "spam", "class", "category"]:
        print(df[col].value_counts())