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
7. The primary churn signal MUST be the feature with
   the largest POSITIVE SHAP contribution.
8. A feature with a NEGATIVE SHAP value must NOT be
   described as the primary churn signal.
9. Do NOT convert a SHAP contribution into a
   population-level statement.
10. A SHAP value only describes this customer's
    model prediction.
11. Do NOT describe the customer as loyal, inactive,
    valuable, engaged, or disengaged unless the supplied
    evidence explicitly supports that description.
12. Do NOT infer customer motivations.
13. Do NOT diagnose the customer.
14. Do NOT change the supplied risk level.
15. Expected revenue at risk is an estimate, not guaranteed
    future revenue loss.
16. Keep the recommendation concise and evidence-based.
17. Do NOT invent dates, time windows, deadlines, renewal dates,
    tenure end dates, or future events that are not provided.

18. Do NOT describe customer loyalty, motivation, concerns,
    engagement, or intent unless explicitly supported by the
    supplied data.

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
Identify ONLY the feature with the largest positive SHAP value.
Mention its observed value and SHAP contribution.
Do not use a negative SHAP value.
Do not claim causality.

Recommended Action:
Give ONE practical retention action based only on the supplied
churn probability, SHAP drivers, observed values, and revenue
at risk.

Do not invent dates, deadlines, customer motivations,
loyalty status, or behavioral thresholds.

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