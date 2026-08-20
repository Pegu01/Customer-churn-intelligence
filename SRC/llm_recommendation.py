# LLM Customer Recommendation Engine

import ollama


def generate_llm_recommendation(
    customer_id,
    churn_probability,
    risk_level,
    total_spend,
    revenue_at_risk,
    top_shap_drivers
):
    """
    Generate an evidence-based customer retention recommendation
    using the local Llama 3.2 3B model.

    The LLM receives:
    - Churn probability
    - Risk level
    - Total spend
    - Expected revenue at risk
    - Top SHAP drivers

    The LLM does NOT receive the raw customer dataset.
    """

    # --------------------------------------------------
    # Prepare SHAP information
    # --------------------------------------------------

    drivers_text = ""

    for driver in top_shap_drivers:

        drivers_text += (
            f"- Feature: {driver['feature']}\n"
            f"  Observed value: {driver['value']}\n"
            f"  SHAP contribution: {driver['shap']:.4f}\n"
            f"  Direction: {driver['direction']}\n\n"
        )

    # --------------------------------------------------
    # Create controlled prompt
    # --------------------------------------------------

    prompt = f"""
You are a customer retention recommendation assistant.

Your job is to translate machine-learning findings
into a concise business recommendation.

IMPORTANT RULES:

1. Use ONLY the information provided below.
2. Do NOT invent customer information.
3. Do NOT claim that any feature causes churn.
4. SHAP values are model contributions, not causal explanations.
5. Do NOT create thresholds from SHAP values.
6. Do NOT assume a feature is good, bad, high, low,
   recent, old, loyal, inactive, or valuable unless
   explicitly stated.
7. Do NOT infer customer motivations.
8. Do NOT diagnose the customer.
9. Do NOT change the supplied risk level.
10. Expected revenue at risk is an estimate, not guaranteed
    future revenue loss.
11. Keep the recommendation concise and evidence-based.

CUSTOMER ID:
{customer_id}

CHURN PROBABILITY:
{churn_probability:.2%}

RISK LEVEL:
{risk_level}

TOTAL SPEND:
{total_spend:,.2f}

EXPECTED REVENUE AT RISK:
{revenue_at_risk:,.2f}

TOP SHAP DRIVERS:
{drivers_text}

Return exactly these sections:

Risk Level:
{risk_level}

Primary Churn Signal:
Identify the strongest positive SHAP driver.
Mention its observed value and SHAP contribution.
Do not claim causality.

Recommended Action:
Give one cautious and practical business retention action
based only on the supplied information.

Reason:
Briefly explain the recommendation using only the churn
probability, SHAP information, and expected revenue at risk.
"""

    # --------------------------------------------------
    # Call local Llama
    # --------------------------------------------------

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


# ------------------------------------------------------
# SIMPLE TEST
# ------------------------------------------------------

if __name__ == "__main__":

    test_customer_id = (
        "de6066796e7c487ac8f560a0054a2d33f46670665f70220e98729fbbdf7ea7ad"
    )

    test_churn_probability = 0.7782976

    if test_churn_probability >= 0.70:
        test_risk_level = "High"
    elif test_churn_probability >= 0.40:
        test_risk_level = "Medium"
    else:
        test_risk_level = "Low"

    test_total_spend = 18500.00

    test_revenue_at_risk = (
        test_total_spend * test_churn_probability
    )

    test_shap_drivers = [
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

    recommendation = generate_llm_recommendation(
        customer_id=test_customer_id,
        churn_probability=test_churn_probability,
        risk_level=test_risk_level,
        total_spend=test_total_spend,
        revenue_at_risk=test_revenue_at_risk,
        top_shap_drivers=test_shap_drivers
    )

    print()
    print("=" * 70)
    print("LLM CUSTOMER RECOMMENDATION ENGINE")
    print("=" * 70)

    print("Customer ID:", test_customer_id)
    print("Churn Probability:", f"{test_churn_probability:.2%}")
    print("Risk Level:", test_risk_level)
    print("Total Spend:", f"{test_total_spend:,.2f}")
    print(
        "Expected Revenue at Risk:",
        f"{test_revenue_at_risk:,.2f}"
    )

    print()
    print("RECOMMENDATION")
    print("-" * 70)
    print(recommendation)

    print("=" * 70)