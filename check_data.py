import pandas as pd

print("Loading dataset...")

data = pd.read_csv("creditcard.csv")

print("\nDataset successfully loaded!")

print("\nNumber of transactions:")
print(len(data))

print("\nDataset columns:")
print(data.columns.tolist())

print("\nTransaction classes:")
print(data["Class"].value_counts())

print("\nFraud percentage:")
print(data["Class"].value_counts(normalize=True)[1] * 100)