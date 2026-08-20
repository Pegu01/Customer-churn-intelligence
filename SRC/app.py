# Streamlit application for customer churn intelligence

import os
import joblib
import pandas as pd
import streamlit as st

from llm_recommendation import generate_llm_recommendation


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide"
)


# ==================================================
# PROJECT PATHS
# ==================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "xgb_churn_model.pkl"
)

SHAP_PATH = os.path.join(
    BASE_DIR,
    "shap_explainer.pkl"
)

FEATURE_PATH = os.path.join(
    BASE_DIR,
    "feature_columns.pkl"
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "Notebooks",
    "data",
    "processed",
    "modeling_table.parquet"
)


# ==================================================
# LOAD MODEL COMPONENTS
# ==================================================

@st.cache_resource
def load_model_components():

    xgb_model = joblib.load(
        MODEL_PATH
    )

    explainer = joblib.load(
        SHAP_PATH
    )

    feature_columns = joblib.load(
        FEATURE_PATH
    )

    return (
        xgb_model,
        explainer,
        feature_columns
    )


@st.cache_data
def load_customer_data():

    return pd.read_parquet(
        DATA_PATH
    )


xgb_model, explainer, feature_columns = (
    load_model_components()
)

df = load_customer_data()


# ==================================================
# CREATE MODEL INPUT
# ==================================================

@st.cache_data
def prepare_model_data(df, feature_columns):

    model_df = df.copy()

    # The modeling table already contains the
    # one-hot encoded columns.

    for column in feature_columns:

        if column not in model_df.columns:

            model_df[column] = 0

    X_all = model_df[
        feature_columns
    ].copy()

    return model_df, X_all


df, X_all = prepare_model_data(
    df,
    feature_columns
)


# ==================================================
# TITLE
# ==================================================

st.title("📊 Customer Churn Intelligence")

st.markdown(
    """
    **AI-powered customer retention analytics**

    Predict customer churn, understand the key drivers,
    estimate revenue at risk, and generate evidence-based
    retention recommendations using a local LLM.
    """
)

st.caption(
    "XGBoost • SHAP • Ollama / Llama 3.2 • Streamlit"
)

st.divider()


# ==================================================
# CUSTOMER INPUT
# ==================================================

st.subheader("Customer Analysis")

customer_id = st.text_input(
    "Customer ID",
    placeholder="Paste a customer ID here..."
)


analyze = st.button(
    "🔍 Analyze Customer",
    type="primary"
)

st.caption(
    "The model predicts churn first. SHAP explains the prediction. "
    "The local LLM converts those findings into a retention action."
)

# ==================================================
# ANALYSIS
# ==================================================

if analyze:

    if not customer_id:

        st.warning(
            "Please enter a Customer ID."
        )

    else:

        customer_rows = df[
            df["customer_id"] == customer_id
        ]

        if customer_rows.empty:

            st.error(
                "Customer ID was not found."
            )

        else:

            customer_index = customer_rows.index[0]

            # ------------------------------------------
            # Model prediction
            # ------------------------------------------

            customer_features = X_all.loc[
                [customer_index]
            ]

            churn_probability = (
                xgb_model.predict_proba(
                    customer_features
                )[0, 1]
            )

            # ------------------------------------------
            # Risk level
            # ------------------------------------------

            if churn_probability >= 0.70:

                risk_level = "High"

            elif churn_probability >= 0.40:

                risk_level = "Medium"

            else:

                risk_level = "Low"

            # ------------------------------------------
            # SHAP
            # ------------------------------------------

            customer_shap = (
                explainer.shap_values(
                    customer_features
                )[0]
            )

            shap_explanation = pd.DataFrame({

                "Feature": feature_columns,

                "Value": (
                    customer_features
                    .iloc[0]
                    .values
                ),

                "SHAP": customer_shap

            })

            shap_explanation[
                "Absolute_SHAP"
            ] = shap_explanation[
                "SHAP"
            ].abs()

            shap_explanation = (
                shap_explanation
                .sort_values(
                    "Absolute_SHAP",
                    ascending=False
                )
            )

            top_5_drivers = (
                shap_explanation
                .head(5)
            )

            # ------------------------------------------
            # Prepare SHAP data for Llama
            # ------------------------------------------

            top_shap_drivers = []

            for _, row in top_5_drivers.iterrows():

                feature = row["Feature"]

                value = row["Value"]

                shap_value = float(
                    row["SHAP"]
                )

                if feature == "preferred_channel":

                    preferred_channel_map = {

                        0: "Email",

                        1: "App",

                        2: "SMS"

                    }

                    value = (
                        preferred_channel_map
                        .get(
                            int(value),
                            "Unknown"
                        )
                    )

                if shap_value > 0:

                    direction = (
                        "pushes toward churn"
                    )

                else:

                    direction = (
                        "pushes away from churn"
                    )

                top_shap_drivers.append({

                    "feature": feature,

                    "value": value,

                    "shap": shap_value,

                    "direction": direction

                })

            # ------------------------------------------
            # Actual customer spend
            # ------------------------------------------

            total_spend = float(
                df.loc[
                    df["customer_id"] == customer_id,
                    "total_spend"
                ].iloc[0]
            )

            # Modeling table stores normalized price.
            # Convert to project monetary units.

            total_spend = (
                total_spend * 100
            )

            # ------------------------------------------
            # Revenue at risk
            # ------------------------------------------

            revenue_at_risk = (
                total_spend
                * churn_probability
            )

            # ------------------------------------------
            # LLM recommendation
            # ------------------------------------------

            recommendation = (
                generate_llm_recommendation(

                    customer_id=customer_id,

                    churn_probability=(
                        churn_probability
                    ),

                    risk_level=risk_level,

                    total_spend=total_spend,

                    revenue_at_risk=(
                        revenue_at_risk
                    ),

                    top_shap_drivers=(
                        top_shap_drivers
                    )

                )
            )

            # ==================================================
            # DISPLAY RESULTS
            # ==================================================

            st.success(
                "Customer analysis completed."
            )

            st.subheader(
                "Customer Overview"
            )

                        # ==================================================
            # DISPLAY RESULTS
            # ==================================================

            st.success(
                "Customer analysis completed."
            )

            st.subheader(
                "Customer Overview"
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.metric(
                    "Churn Probability",
                    f"{churn_probability:.2%}"
                )

            with col2:

                if risk_level == "High":

                    st.error(
                        f"Risk Level: {risk_level}"
                    )

                elif risk_level == "Medium":

                    st.warning(
                        f"Risk Level: {risk_level}"
                    )

                else:

                    st.success(
                        f"Risk Level: {risk_level}"
                    )

            with col3:

                st.metric(
                    "Total Spend",
                    f"{total_spend:,.2f}"
                )

            with col4:

                st.metric(
                    "Revenue at Risk",
                    f"{revenue_at_risk:,.2f}"
                )

            # ------------------------------------------
            # Risk interpretation
            # ------------------------------------------

            if risk_level == "High":

                st.warning(
                    "This customer has a high predicted "
                    "churn risk and should be considered "
                    "for retention action."
                )

            elif risk_level == "Medium":

                st.info(
                    "This customer has a moderate predicted "
                    "churn risk and may benefit from targeted "
                    "engagement."
                )

            else:

                st.success(
                    "This customer currently has a lower "
                    "predicted churn risk."
                )

            st.divider()


                        # ==================================================
            # SHAP DRIVERS
            # ==================================================

            st.subheader(
                "Top Churn Drivers"
            )

            st.caption(
                "Positive SHAP values push the prediction "
                "toward churn. Negative SHAP values push "
                "it away from churn."
            )

            # Create chart data
            chart_data = (
                top_5_drivers[
                    ["Feature", "SHAP"]
                ]
                .set_index("Feature")
            )

            st.bar_chart(
                chart_data,
                horizontal=True
            )

            # Detailed SHAP table
            st.dataframe(
                top_5_drivers[
                    [
                        "Feature",
                        "Value",
                        "SHAP"
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )

            st.divider()


            # ==================================================
            # LLM RECOMMENDATION
            # ==================================================

            st.subheader(
                "🤖 Retention Recommendation"
            )

            st.markdown(
                recommendation
            )

            # ==================================================
# HOW IT WORKS
# ==================================================

st.divider()

with st.expander("🔎 How this system works"):

    st.markdown(
        """
        ### Customer Churn Intelligence Pipeline

        **1. XGBoost**
        
        Predicts the customer's probability of churn.

        **2. SHAP**
        
        Explains which features contributed most to
        the individual churn prediction.

        **3. Revenue at Risk**
        
        Estimates potential revenue exposure using
        customer spend and predicted churn probability.

        **4. Local Llama 3.2**
        
        Converts the model prediction and SHAP evidence
        into a practical retention recommendation.

        ### Important principle

        The LLM does **not** predict churn and does not
        invent customer facts.

        **XGBoost predicts → SHAP explains → Llama translates.**
        """
    )