#  AI Fraud Detection System V2

An interactive fraud detection portfolio project built with **Python, Scikit-learn, Random Forest, Pandas, and Streamlit**.

The project combines an interactive transaction risk simulator with a machine learning evaluation interface for credit card fraud detection.

---

##  Project Overview

AI Fraud Detection System V2 contains two main components:

###  1. New Transaction Analysis

Users can manually enter transaction information such as:

- Transaction amount
- Distance from home
- Online transaction status
- Foreign transaction status
- Number of failed attempts
- Unusual transaction behavior

The application evaluates these inputs using a transparent **rule-based demonstration risk simulator**.

The simulator returns:

- Demo Risk Score
- Risk Level
- Detected Risk Factors
- Transaction Summary

Risk levels are classified as:

- 🟢 LOW RISK
- 🟠 MEDIUM RISK
- 🔴 HIGH RISK

> This component is a demonstration simulator and is not the output of the trained Random Forest model.

---

###  2. ML Model Evaluation

The second component demonstrates a trained **Random Forest classifier** using anonymized credit card transaction data.

Users can select stored demo transactions and compare:

- Model prediction
- Model fraud score
- Actual dataset class
- Prediction correctness

The interface also includes:

- Confusion Matrix
- Feature Importance
- Precision
- Recall
- F1 Score

---

##  Machine Learning Model

The fraud detection model uses:

**Random Forest Classifier**

The original dataset contains anonymized numerical features:

```text
Time
V1 - V28
Amount
Class
```

Where:

```text
Class = 0 → Normal Transaction
Class = 1 → Fraudulent Transaction
```

The `V1–V28` variables are anonymized PCA-transformed features.

Because their original banking meanings are not available, users are not asked to manually enter these values in the application.

---

##  Model Performance

The trained Random Forest model achieved the following results on the held-out test set:

| Metric | Result |
|---|---:|
| Precision | 91.67% |
| Recall | 78.57% |
| F1 Score | 84.62% |

### Confusion Matrix

| Result | Count |
|---|---:|
| True Negative | 56,857 |
| False Positive | 7 |
| False Negative | 21 |
| True Positive | 77 |

These results show the model's performance on the held-out test data and should not be interpreted as guaranteed performance on real banking transactions.

---

##  New Transaction Risk Simulator

The New Transaction Analysis interface uses understandable transaction characteristics instead of anonymized PCA features.

Example inputs include:

```text
Transaction Amount
Distance From Home
Online Transaction
Foreign Transaction
Failed Attempts
Unusual Transaction
```

The simulator assigns demonstration risk points based on predefined rules and combinations of risk factors.

For example, a transaction may receive additional risk points when:

- The transaction amount is unusually high
- The transaction occurs far from the normal location
- The transaction is foreign
- Multiple attempts have failed
- The transaction is marked as unusual
- Multiple risk factors occur together

The final score is converted into a demonstration risk band:

```text
0 - 29   → LOW RISK
30 - 69  → MEDIUM RISK
70 - 100 → HIGH RISK
```

This score is **not a calibrated probability of fraud**.

---

##  Model Evaluation Features

The Streamlit application provides visual tools for understanding model performance.

### Confusion Matrix

The confusion matrix shows:

- True Positives
- True Negatives
- False Positives
- False Negatives

This is particularly important in fraud detection because false negatives represent fraudulent transactions that the model failed to detect.

### Feature Importance

The application also displays the most influential features used by the Random Forest model.

Since the dataset is anonymized, features such as `V1`, `V4`, or `V14` do not directly correspond to publicly available banking variables.

---

##  Demo Dataset

A lightweight demo dataset is used in the deployed Streamlit application.

This allows the application to demonstrate model predictions without requiring the full original dataset to be included in the deployment repository.

The demo sample is intended for **interface and model demonstration purposes** and should not be interpreted as representative of real-world fraud prevalence.

---

##  Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest
- Joblib
- Matplotlib
- Streamlit
- Git
- GitHub

---

##  Project Structure

```text
fraud-detection-system/
│
├── app.py
├── main.py
├── predict.py
├── check_data.py
├── create_demo_data.py
│
├── fraud_model.pkl
├── demo_transactions.csv
│
├── confusion_matrix.png
├── feature_importance.png
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

##  Running the Project Locally

Clone the repository:

```bash
git clone https://github.com/silademir07/fraud-detection-system.git
```

Enter the project directory:

```bash
cd fraud-detection-system
```

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

##  Streamlit Deployment

The application is designed for deployment with Streamlit Community Cloud.

The deployment uses:

```text
Branch: main
Main file: app.py
```

Updates pushed to the GitHub `main` branch can be reflected in the deployed application after Streamlit redeploys the project.

---

##  Important Disclaimer

This project was created for **educational and portfolio purposes**.

The **New Transaction Analysis** section is a rule-based demonstration simulator.

The **ML Model Evaluation** section uses a trained Random Forest classifier on anonymized credit card transaction data.

Neither component is intended for production banking systems or real financial decision-making.

The model fraud score should not be interpreted as a calibrated real-world probability of fraud.

---

##  Future Improvements

Possible future improvements include:

- SHAP-based model explainability
- Precision-Recall and ROC curves
- Probability calibration
- Adjustable classification thresholds
- Additional machine learning model comparisons
- REST API integration
- Docker deployment
- Database integration
- Transaction history dashboard
- Authentication and user roles

---

##  Project Purpose

This project demonstrates practical experience with:

- Machine learning
- Classification problems
- Imbalanced datasets
- Model evaluation
- Fraud detection concepts
- Data visualization
- Interactive application development
- Git/GitHub version control
- Streamlit deployment

---

**AI Fraud Detection System V2**

Built as a machine learning and software development portfolio project.