# -------------------- IMPORTS --------------------
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier


# -------------------- PREPROCESSING --------------------
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare the dataset for training.
    """

    # Drop irrelevant columns safely
    cols_to_drop = [
        'blockNumber', 'confirmations', 'total_tx_sent_malicious',
        'total_tx_sent_malicious_unique', 'total_tx_received_malicious_unique'
    ]
    df = df.drop(columns=cols_to_drop, errors='ignore')

    # Shuffle dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Create has_activity feature
    df['has_activity'] = np.where(
        (df.get('total_tx_sent', 0) > 0) | (df.get('total_received', 0) > 0),
        1,0
    )

    return df


# -------------------- FINAL FEATURES --------------------
final_features = [
    'Hour',
    'total_received',
    'mean_value_received',
    'time_diff_first_last_received',
    'total_tx_sent',
    'total_tx_sent_unique',
    'has_activity',
    'Fraud'
]


# -------------------- MODEL TRAINING --------------------
def train_model(df: pd.DataFrame):
    """
    Train the XGBoost model and persist it using joblib.
    """

    # Keep only required features
    df = df[final_features].copy()

    X = df.drop(columns=['Fraud'])
    y = df['Fraud']

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # XGBoost model with tuned parameters
    model = XGBClassifier(
        colsample_bytree=0.8,
        gamma=0,
        learning_rate=0.2,
        max_depth=7,
        n_estimators=200,
        subsample=1.0,
        random_state=42,
        eval_metric="logloss"  # avoids warnings
    )

    # Train model
    model.fit(X_train, y_train)

    # -------- SAFE SERIALIZATION --------
    try:
        joblib.dump(model, "trained_xgb_model.joblib")
    finally:
        # Ensures file handles are properly closed even if error occurs
        joblib.dump(model, "trained_xgb_model.joblib")
    print ("                                       ")
    print("Model trained and saved as trained_xgb_model.joblib")
    print ("                                       ")


    # ---------------- EVALUATE ----------------
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score
    )

    print("----- MODEL PERFORMANCE (XGBoost) -----")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")
    print("----------------------------------------")


# -------------------- MAIN EXECUTION --------------------
if __name__ == "__main__":
    df = pd.read_csv('data/etfd_dataset.csv')
    df = preprocess_data(df)
    train_model(df)
