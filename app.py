import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# --------------------------------------------------
# LOAD MODEL AND DATA
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("fraud_model.pkl")


@st.cache_data
def load_data():
    return pd.read_csv("creditcard.csv")


model = load_model()
data = load_data()

X = data.drop("Class", axis=1)
y = data["Class"]

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("💳 AI Fraud Detection System")

st.caption(
    "Machine Learning based credit card fraud detection "
    "using a Random Forest classifier."
)

st.divider()

# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------

st.subheader("📊 Model Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Transactions",
        f"{len(data):,}"
    )

with col2:
    st.metric(
        "Fraud Cases",
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

st.divider()

# --------------------------------------------------
# TRANSACTION ANALYSIS
# --------------------------------------------------

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

# --------------------------------------------------
# ANALYSIS
# --------------------------------------------------

if st.button(
    "Analyze Transaction",
    type="primary",
    use_container_width=True
):

    prediction = model.predict(transaction)[0]

    fraud_score = model.predict_proba(
        transaction
    )[0][1]

    actual_class = y.iloc[transaction_index]

    st.divider()

    st.subheader("🧠 Analysis Result")

    result1, result2, result3 = st.columns(3)

    # Prediction
    with result1:

        if prediction == 1:
            st.error("⚠️ FRAUD DETECTED")
        else:
            st.success("✅ NORMAL TRANSACTION")

    # Fraud score
    with result2:

        st.metric(
            "Model Fraud Score",
            f"{fraud_score * 100:.2f}%"
        )

    # Actual class
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

    st.caption(
        "The Model Fraud Score is the classifier's output score "
        "and should not be interpreted as a calibrated real-world "
        "probability of fraud."
    )

st.divider()

# --------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------

st.subheader("📈 Model Evaluation")

tab1, tab2 = st.tabs(
    [
        "Confusion Matrix",
        "Feature Importance"
    ]
)

with tab1:

    st.write(
        "The confusion matrix shows how the model performed "
        "on the held-out test dataset."
    )

    st.image(
        "confusion_matrix.png",
        caption="Random Forest Confusion Matrix",
        use_container_width=True
    )

    st.write(
        "**True Negative:** 56,857  |  "
        "**False Positive:** 7  |  "
        "**False Negative:** 21  |  "
        "**True Positive:** 77"
    )

with tab2:

    st.write(
        "Feature importance shows which transformed variables "
        "contributed most to the Random Forest decisions."
    )

    st.image(
        "feature_importance.png",
        caption="Top 10 Feature Importances",
        use_container_width=True
    )

    st.info(
        "V1–V28 are anonymized PCA-transformed features. "
        "Their original banking variables are not available "
        "in this dataset."
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "AI Fraud Detection Portfolio Project • "
    "Python • Scikit-learn • Random Forest • Streamlit"
)