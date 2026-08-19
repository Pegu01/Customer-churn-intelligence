import ollama


def generate_recommendation(
    churn_probability,
    top_shap_drivers,
    risk_level
):
    """
    Generate a customer retention recommendation
    using the local Llama 3.2 3B model.
    """

    # Convert SHAP information into readable text
    drivers_text = ""

    for driver in top_shap_drivers:

        drivers_text += (
            f"- Feature: {driver['feature']}\n"
            f"  Observed value: {driver['value']}\n"
            f"  SHAP contribution: {driver['shap']:.4f}\n"
            f"  Model direction: {driver['direction']}\n"
            f"  Note: This is a model contribution for this customer, "
            f"not a proven cause.\n\n"
        )

    # Create the prompt
    prompt = f"""
You are a customer retention recommendation assistant.

Your job is to translate machine-learning findings
into a practical business recommendation.

IMPORTANT RULES:

1. Use ONLY the information provided below.
2. Do NOT invent customer information.
3. Do NOT diagnose the customer.
4. Do NOT claim that a feature causes churn.
5. SHAP values are model contributions, not causal explanations.
6. Do NOT create thresholds from individual SHAP values.
7. Do NOT say that a feature is high, low, good, bad, recent,
   old, loyal, inactive, or valuable unless explicitly stated.
8. Do NOT infer the reason behind a SHAP contribution.
9. Do NOT interpret the business meaning of a feature beyond
   its name and supplied value.
10. If the supplied information is insufficient to justify a
    specific intervention, recommend a general retention review.
11. The strongest positive SHAP value identifies the strongest
    supplied contribution toward the model's churn prediction.
12. The strongest negative SHAP value identifies a contribution
    away from the model's churn prediction.
13. Keep the recommendation concise and evidence-based.

CUSTOMER CHURN PROBABILITY:
{churn_probability:.2%}

RISK LEVEL:
{risk_level}

TOP SHAP DRIVERS:
{drivers_text}

Return the answer using exactly these sections:

Risk Level:
{risk_level}

Primary Churn Signal:
State the feature with the strongest positive SHAP contribution.
Mention its observed value and SHAP contribution.
Do not explain why it causes churn.

Recommended Action:
Recommend one cautious business action related to reviewing
or addressing the supplied churn signals.
Do not invent a specific cause or customer motivation.

Reason:
Briefly explain that the customer has the supplied churn
probability and that the listed SHAP feature contributes
toward the model's prediction.
Do not claim causality.
"""

    # Send the prompt to the local Llama model
    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


# ============================================================
# REAL CUSTOMER TEST
# ============================================================

if __name__ == "__main__":

    # Actual XGBoost probability
    churn_probability = 0.7782976

    # Determine risk level using Python
    if churn_probability >= 0.70:
        risk_level = "High"
    elif churn_probability >= 0.40:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    # Actual SHAP results from our test customer
    top_shap_drivers = [
        {
            "feature": "tenure_days",
            "value": 208,
            "shap": 0.249485,
            "direction": "pushes toward churn"
        },
        {
            "feature": "days_since_last_purchase",
            "value": 208,
            "shap": 0.210726,
            "direction": "pushes toward churn"
        },
        {
            "feature": "total_orders",
            "value": 7,
            "shap": 0.198351,
            "direction": "pushes toward churn"
        },
        {
            "feature": "preferred_channel",
            "value": "App",
            "shap": -0.180787,
            "direction": "pushes away from churn"
        },
        {
            "feature": "prior_90d_orders",
            "value": 0,
            "shap": 0.128365,
            "direction": "pushes toward churn"
        }
    ]

    # Generate recommendation
    recommendation = generate_recommendation(
        churn_probability=churn_probability,
        top_shap_drivers=top_shap_drivers,
        risk_level=risk_level
    )

    # Display result
    print()
    print("=" * 60)
    print("CUSTOMER CHURN RECOMMENDATION")
    print("=" * 60)
    print(recommendation)
    print("=" * 60)