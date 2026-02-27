from crewai import Task
from agents import model_agent

model_task = Task(
    description='''
    Write a complete Python script to train a logistic regression model for fraud detection.

    The script must:
    1. Load data/cleaned_fraud.csv
    2. Load data/selected_features.json and use ONLY those features as X
    3. Load data/class_weights.json for class weights
    4. Set y = is_fraud
    5. Split into train/validation using a chronological 80/20 split (do NOT shuffle — preserve time order)
    6. Train LogisticRegression with:
       - class_weight parameter set from class_weights.json
       - max_iter=1000
       - solver=lbfgs
    7. Evaluate on validation set and print:
       - Confusion matrix
       - Classification report (precision, recall, F1)
       - ROC-AUC score
       - Highlight recall for fraud class specifically
    8. Save the trained model to data/model.pkl using joblib
    9. Save validation metrics to data/model_metrics.json

    Use only: pandas, sklearn, joblib, json, numpy
    Output only the complete runnable Python script with inline comments. No explanations.
    ''',
    expected_output='A complete runnable Python script that produces data/model.pkl and data/model_metrics.json',
    agent=model_agent
)