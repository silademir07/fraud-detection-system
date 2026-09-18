import pandas as pd
import joblib

print("=== Fraud Detection Prediction System ===")


model = joblib.load("fraud_model.pkl")

print("Model successfully loaded!")


data = pd.read_csv("creditcard.csv")

X = data.drop("Class", axis=1)
y = data["Class"]


normal_index = y[y == 0].index[0]


fraud_index = y[y == 1].index[0]

normal_transaction = X.loc[[normal_index]]
fraud_transaction = X.loc[[fraud_index]]

normal_prediction = model.predict(normal_transaction)[0]
normal_probability = model.predict_proba(normal_transaction)[0][1]

print("\n=== Normal Transaction Test ===")
print(f"Actual class: {y.loc[normal_index]}")
print(f"Predicted class: {normal_prediction}")
print(f"Fraud probability: {normal_probability * 100:.2f}%")


fraud_prediction = model.predict(fraud_transaction)[0]
fraud_probability = model.predict_proba(fraud_transaction)[0][1]

print("\n=== Fraud Transaction Test ===")
print(f"Actual class: {y.loc[fraud_index]}")
print(f"Predicted class: {fraud_prediction}")
print(f"Fraud probability: {fraud_probability * 100:.2f}%")