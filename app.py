import streamlit as st
import pandas as pd
import joblib


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="AI Fraud Detection System V2",
    page_icon="💳",
    layout="wide"
)


# ==================================================
# LOAD REAL DATASET MODEL
# ==================================================

@st.cache_resource
def load_model():
    return joblib.load("fraud_model.pkl")


@st.cache_data
def load_demo_data():
    return pd.read_csv("demo_transactions.csv")


model = load_model()
data = load_demo_data()

X = data.drop("Class", axis=1)
y = data["Class"]


# ==================================================
# HEADER
# ==================================================

st.title("💳 AI Fraud Detection System V2")

st.write(
    "Interactive transaction risk analysis and "
    "machine learning fraud detection demonstration."
)

st.caption(
    "Portfolio Project • Python • Scikit-learn • "
    "Random Forest • Streamlit"
)

st.divider()


# ==================================================
# MAIN TABS
# ==================================================

tab_new, tab_model = st.tabs(
    [
        "📝 New Transaction Analysis",
        "🧪 ML Model Evaluation"
    ]
)


# ==================================================
# TAB 1
# NEW TRANSACTION ANALYSIS
# ==================================================

with tab_new:

    st.header("📝 New Transaction Analysis")

    st.write(
        "Enter transaction information below to perform "
        "an interactive demonstration risk assessment."
    )

    st.info(
        "This section is a demonstration risk simulator. "
        "The risk score is based on transparent demonstration "
        "rules and is not the output of the trained credit-card "
        "Random Forest model."
    )

    st.subheader("Transaction Information")


    # --------------------------------------------------
    # INPUT ROW 1
    # --------------------------------------------------

    input1, input2 = st.columns(2)

    with input1:

        amount = st.number_input(
            "Transaction Amount ($)",
            min_value=0.0,
            max_value=100000.0,
            value=100.0,
            step=10.0
        )

    with input2:

        distance = st.number_input(
            "Distance From Home (km)",
            min_value=0.0,
            max_value=10000.0,
            value=5.0,
            step=1.0
        )


    # --------------------------------------------------
    # INPUT ROW 2
    # --------------------------------------------------

    input3, input4 = st.columns(2)

    with input3:

        online_option = st.selectbox(
            "Online Transaction",
            [
                "No",
                "Yes"
            ]
        )

    with input4:

        foreign_option = st.selectbox(
            "Foreign Transaction",
            [
                "No",
                "Yes"
            ]
        )


    # --------------------------------------------------
    # INPUT ROW 3
    # --------------------------------------------------

    input5, input6 = st.columns(2)

    with input5:

        failed_attempts = st.number_input(
            "Failed Attempts",
            min_value=0,
            max_value=20,
            value=0,
            step=1
        )

    with input6:

        unusual_option = st.selectbox(
            "Unusual Transaction",
            [
                "No",
                "Yes"
            ]
        )


    online_transaction = (
        1 if online_option == "Yes" else 0
    )

    foreign_transaction = (
        1 if foreign_option == "Yes" else 0
    )

    unusual_transaction = (
        1 if unusual_option == "Yes" else 0
    )


    st.write("")


    analyze_new = st.button(
        "🔍 Analyze New Transaction",
        type="primary",
        width="stretch",
        key="analyze_new_transaction"
    )


    # ==================================================
    # DEMO RISK ANALYSIS
    # ==================================================

    if analyze_new:

        risk_points = 0

        risk_factors = []


        # ----------------------------------------------
        # AMOUNT
        # ----------------------------------------------

        if amount > 500:
            risk_points += 5

        if amount > 1500:
            risk_points += 10
            risk_factors.append(
                "High transaction amount"
            )

        if amount > 4000:
            risk_points += 10


        # ----------------------------------------------
        # DISTANCE
        # ----------------------------------------------

        if distance > 50:
            risk_points += 5

        if distance > 150:
            risk_points += 10
            risk_factors.append(
                "Transaction far from normal location"
            )

        if distance > 300:
            risk_points += 10


        # ----------------------------------------------
        # ONLINE
        # ----------------------------------------------

        if online_transaction == 1:
            risk_points += 5


        # ----------------------------------------------
        # FOREIGN
        # ----------------------------------------------

        if foreign_transaction == 1:
            risk_points += 20

            risk_factors.append(
                "Foreign transaction"
            )


        # ----------------------------------------------
        # FAILED ATTEMPTS
        # ----------------------------------------------

        if failed_attempts >= 1:
            risk_points += 5

        if failed_attempts >= 2:
            risk_points += 10

            risk_factors.append(
                "Multiple failed attempts"
            )

        if failed_attempts >= 4:
            risk_points += 10


        # ----------------------------------------------
        # UNUSUAL TRANSACTION
        # ----------------------------------------------

        if unusual_transaction == 1:
            risk_points += 25

            risk_factors.append(
                "Unusual transaction behavior"
            )


        # ----------------------------------------------
        # INTERACTION RISKS
        # ----------------------------------------------

        if (
            foreign_transaction == 1
            and unusual_transaction == 1
        ):
            risk_points += 15

            risk_factors.append(
                "Foreign and unusual activity combination"
            )


        if (
            amount > 1500
            and unusual_transaction == 1
        ):
            risk_points += 10


        if (
            failed_attempts >= 2
            and unusual_transaction == 1
        ):
            risk_points += 10


        if (
            distance > 150
            and foreign_transaction == 1
        ):
            risk_points += 10


        # ----------------------------------------------
        # LIMIT SCORE TO 100
        # ----------------------------------------------

        risk_score = min(
            risk_points,
            100
        )


        # ----------------------------------------------
        # RISK LEVEL
        # ----------------------------------------------

        if risk_score < 30:
            risk_level = "LOW"

        elif risk_score < 70:
            risk_level = "MEDIUM"

        else:
            risk_level = "HIGH"


        # ==================================================
        # DISPLAY RESULT
        # ==================================================

        st.divider()

        st.header("🧠 Risk Analysis Result")


        result1, result2 = st.columns(2)


        with result1:

            st.metric(
                "Demo Risk Score",
                f"{risk_score}%"
            )


        with result2:

            st.metric(
                "Risk Level",
                risk_level
            )


        st.subheader("📊 Risk Score")

        st.progress(
            risk_score / 100
        )


        # ----------------------------------------------
        # RISK LEVEL MESSAGE
        # ----------------------------------------------

        if risk_level == "LOW":

            st.success(
                "🟢 LOW RISK"
            )

        elif risk_level == "MEDIUM":

            st.warning(
                "🟠 MEDIUM RISK"
            )

        else:

            st.error(
                "🔴 HIGH RISK"
            )


        # ----------------------------------------------
        # RISK FACTORS
        # ----------------------------------------------

        st.subheader("🔎 Detected Risk Factors")


        if len(risk_factors) == 0:

            st.success(
                "No major demonstration risk factors "
                "were detected."
            )

        else:

            for factor in risk_factors:

                st.write(
                    f"• {factor}"
                )


        # ----------------------------------------------
        # TRANSACTION SUMMARY
        # ----------------------------------------------

        with st.expander(
            "View Transaction Summary"
        ):

            summary = pd.DataFrame({
                "Field": [
                    "Amount",
                    "Distance From Home",
                    "Online Transaction",
                    "Foreign Transaction",
                    "Failed Attempts",
                    "Unusual Transaction"
                ],

                "Value": [
                    f"${amount:,.2f}",
                    f"{distance:,.1f} km",
                    online_option,
                    foreign_option,
                    failed_attempts,
                    unusual_option
                ]
            })

            st.dataframe(
                summary,
                width="stretch",
                hide_index=True
            )


        st.caption(
            "This score is a demonstration risk indicator "
            "generated from predefined rules. It is not a "
            "calibrated fraud probability and must not be used "
            "for real financial decisions."
        )


# ==================================================
# TAB 2
# MACHINE LEARNING MODEL
# ==================================================

with tab_model:

    st.header("🧪 Machine Learning Model Evaluation")

    st.write(
        "This section demonstrates the trained Random Forest "
        "classifier using anonymized credit-card transaction data."
    )

    st.info(
        "The original dataset uses anonymized PCA-transformed "
        "features V1–V28. The interface therefore uses stored "
        "demo transactions rather than asking users to manually "
        "enter values such as V1 or V14."
    )


    # ==================================================
    # MODEL PERFORMANCE
    # ==================================================

    st.subheader("📊 Model Performance")

    performance1, performance2, performance3, performance4 = (
        st.columns(4)
    )


    with performance1:

        st.metric(
            "Precision",
            "91.67%"
        )


    with performance2:

        st.metric(
            "Recall",
            "78.57%"
        )


    with performance3:

        st.metric(
            "F1 Score",
            "84.62%"
        )


    with performance4:

        st.metric(
            "Model",
            "Random Forest"
        )


    st.caption(
        "These metrics were calculated on the held-out "
        "test portion of the original credit-card dataset."
    )


    st.divider()


    # ==================================================
    # DATASET DEMO
    # ==================================================

    st.subheader("🔬 Dataset Transaction Test")

    st.write(
        "Select a stored transaction to compare the model's "
        "prediction with the transaction's actual dataset label."
    )


    transaction_index = st.number_input(
        "Transaction Index",
        min_value=0,
        max_value=len(data) - 1,
        value=0,
        step=1,
        key="dataset_transaction_index"
    )

    transaction_index = int(
        transaction_index
    )


    transaction = X.iloc[
        [transaction_index]
    ]


    actual_class = y.iloc[
        transaction_index
    ]


    amount_dataset = transaction[
        "Amount"
    ].iloc[0]


    transaction1, transaction2 = st.columns(2)


    with transaction1:

        st.metric(
            "Transaction ID",
            transaction_index
        )


    with transaction2:

        st.metric(
            "Transaction Amount",
            f"${amount_dataset:,.2f}"
        )


    st.write("")


    analyze_dataset = st.button(
        "Analyze Dataset Transaction",
        type="primary",
        width="stretch",
        key="analyze_dataset_transaction"
    )


    # ==================================================
    # DATASET MODEL PREDICTION
    # ==================================================

    if analyze_dataset:

        prediction = model.predict(
            transaction
        )[0]


        fraud_score = model.predict_proba(
            transaction
        )[0][1]


        st.divider()

        st.subheader(
            "🧠 Machine Learning Result"
        )


        ml1, ml2, ml3 = st.columns(3)


        # ----------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------

        with ml1:

            if prediction == 1:

                st.error(
                    "⚠️ FRAUD DETECTED"
                )

            else:

                st.success(
                    "✅ NORMAL TRANSACTION"
                )


        # ----------------------------------------------
        # MODEL SCORE
        # ----------------------------------------------

        with ml2:

            st.metric(
                "Model Fraud Score",
                f"{fraud_score * 100:.2f}%"
            )


        # ----------------------------------------------
        # ACTUAL CLASS
        # ----------------------------------------------

        with ml3:

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


        st.subheader(
            "📊 Model Score"
        )


        st.progress(
            float(fraud_score)
        )


        # ----------------------------------------------
        # MODEL VS ACTUAL
        # ----------------------------------------------

        if prediction == actual_class:

            st.success(
                "The model prediction matches "
                "the actual dataset class."
            )

        else:

            st.warning(
                "The model prediction does not match "
                "the actual dataset class."
            )


        st.caption(
            "The Model Fraud Score is the classifier's "
            "output score and should not be interpreted "
            "as a calibrated real-world probability of fraud."
        )


    st.divider()


    # ==================================================
    # MODEL VISUALIZATIONS
    # ==================================================

    st.subheader(
        "📈 Model Evaluation"
    )


    evaluation1, evaluation2 = st.tabs(
        [
            "Confusion Matrix",
            "Feature Importance"
        ]
    )


    # ----------------------------------------------
    # CONFUSION MATRIX
    # ----------------------------------------------

    with evaluation1:

        st.image(
            "confusion_matrix.png",
            caption="Random Forest Confusion Matrix",
            width="stretch"
        )


        cm1, cm2, cm3, cm4 = st.columns(4)


        with cm1:

            st.metric(
                "True Negative",
                "56,857"
            )


        with cm2:

            st.metric(
                "False Positive",
                "7"
            )


        with cm3:

            st.metric(
                "False Negative",
                "21"
            )


        with cm4:

            st.metric(
                "True Positive",
                "77"
            )


    # ----------------------------------------------
    # FEATURE IMPORTANCE
    # ----------------------------------------------

    with evaluation2:

        st.image(
            "feature_importance.png",
            caption="Top 10 Feature Importances",
            width="stretch"
        )


        st.info(
            "V1–V28 are anonymized PCA-transformed "
            "features. Their original banking meanings "
            "are not available in this dataset."
        )


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.subheader(
    "ℹ️ Project Information"
)

project1, project2, project3 = st.columns(3)


with project1:

    st.write(
        "**Machine Learning**"
    )

    st.write(
        "Random Forest"
    )


with project2:

    st.write(
        "**Technology**"
    )

    st.write(
        "Python • Pandas • Scikit-learn"
    )


with project3:

    st.write(
        "**Web Application**"
    )

    st.write(
        "Streamlit"
    )


st.warning(
    "Educational portfolio project only. "
    "The New Transaction Analysis is a rule-based demonstration "
    "simulator, while the ML Model Evaluation section uses the "
    "trained Random Forest classifier. Neither should be used "
    "for real financial decisions."
)


st.caption(
    "AI Fraud Detection System V2 • "
    "Portfolio Project"
)