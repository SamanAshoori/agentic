import pandas as pd
import numpy as np
import json
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    recall_score
)

def train_fraud_model():
    # 1. Load data
    df = pd.read_csv('data/cleaned_fraud.csv')

    # 2. Load selected features
    with open('data/selected_features.json', 'r') as f:
        feat_data = json.load(f)
        selected_features = feat_data["selected_features"] if isinstance(feat_data, dict) else feat_data

    # 3. Load class weights and ensure keys are integers
    with open('data/class_weights.json', 'r') as f:
        raw_weights = json.load(f)
        class_weights = {int(k): v for k, v in raw_weights.items()}

    # 4. Define target
    y = df['is_fraud']
    X = df[selected_features]

    # 5. Chronological 80/20 split (no shuffle)
    split_idx = int(len(df) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    # 6. Initialize and train RandomForest
    # Tree-based models are invariant to feature scaling; class_weight handles imbalance.
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight=class_weights,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)

    # 7. Evaluate
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    cm = confusion_matrix(y_val, y_pred)
    report = classification_report(y_val, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_val, y_prob)
    fraud_recall = recall_score(y_val, y_pred)

    print("--- Model Evaluation ---")
    print(f"Confusion Matrix:\n{cm}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print(f"Fraud Recall (Primary Metric): {fraud_recall:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))

    # 8. Feature Importances
    importances = pd.DataFrame({
        'feature': selected_features,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False)

    print("\n--- Feature Importances ---")
    print(importances.to_string(index=False))

    # 9. Save the trained model
    joblib.dump(model, 'data/model.pkl')

    # 10. Save validation metrics
    metrics = {
        "roc_auc": float(roc_auc),
        "fraud_recall": float(fraud_recall),
        "confusion_matrix": cm.tolist(),
        "classification_report": report
    }
    
    with open('data/model_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    print("\nModel and metrics successfully saved to data/ directory.")

if __name__ == "__main__":
    train_fraud_model()