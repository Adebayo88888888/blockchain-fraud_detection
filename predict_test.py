# -------------------- IMPORTS --------------------
import requests

# -------------------- CONFIG --------------------
tx_hash = "0xa536035bcf5c36976b989e025339f6cc0b3943bc60171de75a224d19ac80000d"
host = "Eth-fraud-detection-env.eba-ts7tk7mw.eu-west-1.elasticbeanstalk.com"  # change if deployed elsewhere
url = f"http://{host}/predict"

# -------------------- TRANSACTION FEATURES --------------------
transaction_features = {
    "Hour": 14,
    "total_received": 300,
    "mean_value_received": 0.0,
   "time_diff_first_last_received": 200.0,
    "total_tx_sent": 40,
    "total_tx_sent_unique":16,
    "has_activity": 1
}

# Merge tx_hash outside the model features
payload = {"tx_hash": tx_hash, **transaction_features}

# -------------------- SEND REQUEST --------------------
response = requests.post(url, json=payload).json()
print("API Response:", response)

# -------------------- HUMAN-READABLE INTERPRETATION --------------------
prob = response["fraud_probability"]
classification = response.get("classification", "Unknown")
message = response.get("message", "")

print("\nINTERPRETATION:")
print(f"Transaction Hash: {tx_hash}")
print(f"Fraud Probability: {prob:.2f}")
print(f"Classification: {classification}")
print(f"Message: {message}\n")

# Optional additional action based on risk
if prob >= 0.85:
    print("> Action: Immediately block or review transaction.\n")
elif prob >= 0.55:
    print("> Action: Increase monitoring or request additional verification.\n")
else:
    print("> Action: Normal processing.\n")
