import streamlit as st
import pandas as pd
import joblib


st.set_page_config(
    page_title="AI Fraud Detection",
    page_icon="💳",
    layout="wide"
)


@st.cache_resource
def load_model():
    return joblib.load("fraud_model.pkl")


@st.cache_data
def load_data():
    return pd.read_csv("demo_transactions.csv")


model = load_model()
data = load_data()

X = data.drop("Class", axis=1)
y = data["Class"]


st.title("💳 AI Fraud Detection System")

st.caption(
    "Machine Learning based credit card fraud detection "
    "using a Random Forest classifier."
)

st.divider()


# MODEL OVERVIEW

st.subheader("📊 Model Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Demo Transactions",
        f"{len(data):,}"
    )

with col2:
    st.metric(
        "Demo Fraud Cases",
        f"{(y == 1).sum():,}"
    )

with col3:
    st.metric(
        "Precision",
        "91.67%"
    )

with col4:
    st.metric(
        "Recall",
        "78.57%"
    )


col5, col6 = st.columns(2)

with col5:
    st.metric(
        "F1 Score",
        "84.62%"
    )

with col6:
    st.metric(
        "Model",
        "Random Forest"
    )


st.caption(
    "The metrics above were calculated on the held-out test set "
    "of the original credit card dataset. "
    "The 100 transactions used in this interface are a lightweight "
    "demo sample for deployment."
)

st.divider()


# TRANSACTION ANALYSIS

st.subheader("🔍 Transaction Analysis")

transaction_index = st.number_input(
    "Transaction Index",
    min_value=0,
    max_value=len(data) - 1,
    value=0,
    step=1
)

transaction_index = int(transaction_index)

transaction = X.iloc[[transaction_index]]

amount = transaction["Amount"].iloc[0]


info1, info2 = st.columns(2)

with info1:
    st.metric(
        "Transaction ID",
        transaction_index
    )

with info2:
    st.metric(
        "Transaction Amount",
        f"${amount:,.2f}"
    )


st.write("")


if st.button(
    "Analyze Transaction",
    type="primary",
    width="stretch"
):

    prediction = model.predict(transaction)[0]

    fraud_score = model.predict_proba(
        transaction
    )[0][1]

    actual_class = y.iloc[transaction_index]


    # RISK LEVEL

    if fraud_score < 0.30:
        risk_level = "🟢 LOW RISK"

    elif fraud_score < 0.70:
        risk_level = "🟠 MEDIUM RISK"

    else:
        risk_level = "🔴 HIGH RISK"


    st.divider()

    st.subheader("🧠 Analysis Result")


    result1, result2, result3, result4 = st.columns(4)


    # PREDICTION

    with result1:

        if prediction == 1:
            st.error("⚠️ FRAUD DETECTED")

        else:
            st.success("✅ NORMAL TRANSACTION")


    # FRAUD SCORE

    with result2:

        st.metric(
            "Model Fraud Score",
            f"{fraud_score * 100:.2f}%"
        )


    # ACTUAL CLASS

    with result3:

        if actual_class == 1:

            st.metric(
                "Actual Dataset Class",
                "Fraud"
            )

        else:

            st.metric(
                "Actual Dataset Class",
                "Normal"
            )


    # RISK LEVEL

    with result4:

        st.metric(
            "Risk Level",
            risk_level
        )


    # RISK SCORE PROGRESS BAR

    st.write("### 📊 Risk Score")

    st.progress(fraud_score)

    st.caption(
        f"Model Risk Score: {fraud_score * 100:.2f}%"
    )


    # PREDICTION CHECK

    if prediction == actual_class:

        st.success(
            "Model prediction matches the actual dataset class."
        )

    else:

        st.warning(
            "Model prediction does not match the actual dataset class."
        )


    st.caption(
        "The Model Fraud Score is the classifier's output score "
        "and should not be interpreted as a calibrated real-world "
        "probability of fraud."
    )


st.divider()


# MODEL EVALUATION

st.subheader("📈 Model Evaluation")

tab1, tab2 = st.tabs(
    [
        "Confusion Matrix",
        "Feature Importance"
    ]
)


# CONFUSION MATRIX

with tab1:

    st.write(
        "The confusion matrix shows how the Random Forest model "
        "performed on the held-out test dataset."
    )

    st.image(
        "confusion_matrix.png",
        caption="Random Forest Confusion Matrix",
        width="stretch"
    )


    metric1, metric2, metric3, metric4 = st.columns(4)


    with metric1:

        st.metric(
            "True Negative",
            "56,857"
        )


    with metric2:

        st.metric(
            "False Positive",
            "7"
        )


    with metric3:

        st.metric(
            "False Negative",
            "21"
        )


    with metric4:

        st.metric(
            "True Positive",
            "77"
        )


# FEATURE IMPORTANCE

with tab2:

    st.write(
        "Feature importance shows which transformed variables "
        "contributed most to the Random Forest model's decisions."
    )

    st.image(
        "feature_importance.png",
        caption="Top 10 Feature Importances",
        width="stretch"
    )

    st.info(
        "V1–V28 are anonymized PCA-transformed features. "
        "Their original banking variables are not available "
        "in this dataset."
    )


st.divider()


# ABOUT PROJECT

st.subheader("ℹ️ About This Project")

st.write(
    "This portfolio project demonstrates an end-to-end machine "
    "learning workflow for credit card fraud detection."
)


about1, about2, about3 = st.columns(3)


with about1:

    st.write("**Machine Learning**")

    st.write(
        "Random Forest Classifier"
    )


with about2:

    st.write("**Backend / Analysis**")

    st.write(
        "Python, Pandas, Scikit-learn"
    )


with about3:

    st.write("**Web Application**")

    st.write(
        "Streamlit"
    )


st.info(
    "This application is an educational portfolio project. "
    "It is not intended for real banking or financial decision-making."
)


st.divider()


st.caption(
    "AI Fraud Detection System V2 • "
    "Python • Scikit-learn • Random Forest • Streamlit"
)