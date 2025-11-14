import requests

def get_inr_value(rul_hours: float) -> float:
    """
    Estimate maintenance cost in INR from RUL (Remaining Useful Life).

    Args:
        rul_hours (float): Predicted Remaining Useful Life in hours (0–100 scale)

    Returns:
        float: Estimated maintenance cost in INR
    """
    # Try to fetch live USD→INR exchange rate
    try:
        response = requests.get("https://open.er-api.com/v6/latest/USD", timeout=3)
        usd_to_inr = response.json()["rates"]["INR"]
    except Exception:
        usd_to_inr = 84.0  # fallback exchange rate

    # Base cost logic — lower RUL → higher cost
    base_usd_cost = max(100, (100 - rul_hours) * 5)
    cost_in_inr = base_usd_cost * usd_to_inr

    return round(cost_in_inr, 2)
