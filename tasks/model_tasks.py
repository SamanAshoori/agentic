from crewai import Task
from agents import model_agent

model_task = Task(
    description='''
    Write a complete Python script to train a Random Forest model for fraud detection.

    The script must:
    1. Load data/cleaned_fraud.csv
    2. Load data/selected_features.json:
       data = json.load(f)
       selected_features = data["selected_features"] if isinstance(data, dict) else data
    3. Load data/class_weights.json:
       {int(k): v for k, v in raw_weights.items()}
    4. Set y = is_fraud
    5. Chronological 80/20 split — do NOT shuffle, preserve time order:
       split_idx = int(len(df) * 0.8)
    6. Train RandomForestClassifier with:
       - class_weight from class_weights.json
       - n_estimators=100
       - max_depth=10
       - random_state=42
       - n_jobs=-1
       - NO scaling needed for Random Forest
    7. Evaluate on validation set and print:
       - Confusion matrix
       - Classification report (precision, recall, F1)
       - ROC-AUC score
       - Fraud recall specifically
    8. Print feature importances ranked highest to lowest
    9. Save the trained model to data/model.pkl using joblib
    10. Save validation metrics to data/model_metrics.json

    Use only: pandas, sklearn, joblib, json, numpy
    Output only the complete runnable Python script with inline comments. No explanations.
    ''',
    expected_output='A complete runnable Python script that produces data/model.pkl and data/model_metrics.json',
    agent=model_agent
)