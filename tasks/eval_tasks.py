from crewai import Task
from agents import eval_agent

eval_task = Task(
    description='''
    Write a complete Python script to evaluate the trained Random Forest fraud detection model on unseen test data.

    The script must:
    1. Load data/model.pkl using joblib
    2. Load data/fraudTest.csv (raw test data)
    3. Apply identical preprocessing to training:
       - Drop: Unnamed: 0, trans_num, cc_num, unix_time, first, last,
         street, city, zip, lat, long, merch_lat, merch_long, job, dob, merchant
       - Extract hour, day_of_week, month from trans_date_trans_time then drop it
       - Compute age from dob and trans_date_trans_time then drop dob
       - Compute haversine distance between (lat, long) and (merch_lat, merch_long) then drop coordinates
       - LabelEncoder for category, gender, state
    4. Load selected features:
       data = json.load(f)
       selected_features = data["selected_features"] if isinstance(data, dict) else data
    5. NO scaling — Random Forest does not need it
    6. Generate predictions using model.predict and model.predict_proba
    7. Evaluate at default threshold (0.5):
       - Confusion matrix
       - Classification report
       - ROC-AUC score
       - Fraud recall specifically
    8. Tune decision threshold from 0.1 to 0.9 in steps of 0.05
       find threshold that maximises F1 for fraud class (class 1)
    9. Re-evaluate at optimal threshold
    10. Save to data/eval_report.json:
        - default threshold metrics
        - optimal threshold and metrics
        - confusion matrices for both
        - roc_auc

    Use only: pandas, numpy, sklearn, joblib, json
    Output only the complete runnable Python script with inline comments. No explanations.
    ''',
    expected_output='A complete runnable Python script that saves data/eval_report.json',
    agent=eval_agent
)