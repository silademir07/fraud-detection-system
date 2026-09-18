import pandas as pd

# Load the full dataset locally
data = pd.read_csv("creditcard.csv")

# Select normal and fraud examples
normal_samples = data[data["Class"] == 0].sample(
    n=50,
    random_state=42
)

fraud_samples = data[data["Class"] == 1].sample(
    n=50,
    random_state=42
)

# Combine and shuffle them
demo_data = pd.concat(
    [normal_samples, fraud_samples]
).sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# Save small demo dataset
demo_data.to_csv(
    "demo_transactions.csv",
    index=False
)

print("Demo dataset successfully created!")
print("Total demo transactions:", len(demo_data))
print()
print("Classes:")
print(demo_data["Class"].value_counts())