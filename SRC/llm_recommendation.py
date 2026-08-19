# LLM Customer Retention Recommendation

import ollama


def generate_recommendation(
    churn_probability,
    top_shap_drivers,
    customer_facts,
    revenue_at_risk
):
    """
    Generate a customer retention recommendation using
    the local Llama 3.2 3B model.

    The LLM receives only model/SHAP-derived facts.
    """

    drivers_text = "\n".join(
        [
            f"- {driver}: {value}"
            for driver, value in top_shap_drivers.items()
        ]
    )

    facts_text = "\n".join(
        [
            f"- {feature}: {value}"
            for feature, value in customer_facts.items()
        ]
    )

    prompt = f"""
You are a customer retention recommendation assistant.

Your job is to translate machine-learning findings into
a practical business recommendation.

IMPORTANT RULES:
1. Use ONLY the information provided below.
2. Do NOT invent customer information.
3. Do NOT diagnose the customer.
4. Do NOT claim something is a cause unless the supplied
   SHAP information supports that interpretation.
5. Do NOT recommend actions unrelated to the supplied facts.
6. Keep the recommendation concise and business-focused.

CUSTOMER CHURN PROBABILITY:
{churn_probability:.2%}

REVENUE AT RISK:
₹{revenue_at_risk:,.2f}

TOP SHAP DRIVERS:
{drivers_text}

CUSTOMER FACTS:
{facts_text}

Return the answer in exactly this structure:

Risk Level:
[Low / Medium / High]

Primary Churn Signal:
[One short sentence based on the strongest SHAP driver]

Recommended Action:
[One practical retention action]

Reason:
[One or two sentences explaining the recommendation using
only the supplied churn probability and SHAP/customer facts]
"""

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


if __name__ == "__main__":

    # --------------------------------------------------
    # TEST DATA
    # --------------------------------------------------
    # This is temporary test data.
    # Later we will replace this with your actual
    # XGBoost + SHAP output.
    # --------------------------------------------------

    churn_probability = 0.82

    top_shap_drivers = {
        "usage_change_pct": "-42%",
        "days_since_last_purchase": "87 days",
        "customer_satisfaction_score": "2.4"
    }

    customer_facts = {
        "usage_change_pct": "-42%",
        "days_since_last_purchase": "87 days",
        "customer_satisfaction_score": "2.4",
        "recent_90d_orders": "1"
    }

    revenue_at_risk = 18500

    recommendation = generate_recommendation(
        churn_probability=churn_probability,
        top_shap_drivers=top_shap_drivers,
        customer_facts=customer_facts,
        revenue_at_risk=revenue_at_risk
    )

    print("\n" + "=" * 60)
    print("LLM CUSTOMER RETENTION RECOMMENDATION")
    print("=" * 60)
    print(recommendation)
    print("=" * 60)