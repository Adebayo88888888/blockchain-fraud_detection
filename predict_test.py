import requests

# Transaction identifier (metadata only)
tx_hash = "0xa536035bcf5c36976b989e025339f6cc0b3943bc60171de75a224d19ac80000d"

# Host for the deployed API (or local)
host = "127.0.0.1:8000"  # Replace if deployed
url = f"http://{host}/predict"

# Transaction features (only the model features)
transaction_features = {
    "Hour": 14,
    "total_received": 0.0,
    "mean_value_received": 0.0,
    "time_diff_first_last_received": 0.0,
    "total_tx_sent": 3,
    "total_tx_sent_unique": 2,
    "has_activity": 1
}

# Merge tx_hash outside the features
payload = {"tx_hash": tx_hash, **transaction_features}

# Send request
response = requests.post(url, json=payload).json()
print("API Response:", response)

# Extract probability
prob = response["fraud_probability"]

# Human-readable interpretation
if prob >= 0.85:
    print(f"⚠️ Transaction {tx_hash} flagged as HIGH RISK.")
    print("> Action: Immediately block or review transaction.\n")

elif prob >= 0.55:
    print(f"🔶 Transaction {tx_hash} flagged as MEDIUM RISK.")
    print("> Action: Increase monitoring or request additional verification.\n")

else:
    print(f"✅ Transaction {tx_hash} classified as LOW RISK.")
    print("> Action: Normal processing.\n")