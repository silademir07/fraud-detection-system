import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

print("=== AI Fraud Detection System ===")


print("\nLoading dataset...")

data = pd.read_csv("creditcard.csv")

print("Dataset successfully loaded!")
print(f"Total transactions: {len(data)}")



X = data.drop("Class", axis=1)
y = data["Class"]

print(f"Normal transactions: {(y == 0).sum()}")
print(f"Fraud transactions: {(y == 1).sum()}")



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n=== Dataset Split ===")
print(f"Training transactions: {len(X_train)}")
print(f"Testing transactions: {len(X_test)}")



model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)



print("\nTraining model...")

model.fit(X_train, y_train)

print("Model successfully trained!")

joblib.dump(model, "fraud_model.pkl")

print("Model saved as fraud_model.pkl")



print("\nTesting model...")

y_pred = model.predict(X_test)



accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n=== Model Performance ===")

print(f"Accuracy:  {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall:    {recall * 100:.2f}%")
print(f"F1 Score:  {f1 * 100:.2f}%")



cm = confusion_matrix(y_test, y_pred)

print("\n=== Confusion Matrix ===")
print(cm)


display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Fraud"]
)

display.plot()

plt.title("Fraud Detection - Confusion Matrix")
plt.tight_layout()

plt.savefig("confusion_matrix.png")

plt.show()


feature_importance = pd.Series(
    model.feature_importances_,
    index=X.columns
)


top_features = feature_importance.sort_values(
    ascending=False
).head(10)

print("\n=== Top 10 Important Features ===")
print(top_features)


top_features.sort_values().plot(
    kind="barh"
)

plt.title("Top 10 Features for Fraud Detection")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()

plt.savefig("feature_importance.png")

plt.show()