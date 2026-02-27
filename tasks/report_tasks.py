import json
from crewai import Task
from agents import report_agent

# Load all pipeline outputs dynamically
with open('data/eval_report.json') as f:
    eval_report = json.load(f)

with open('data/model_metrics.json') as f:
    model_metrics = json.load(f)

with open('data/selected_features.json') as f:
    features_data = json.load(f)
    selected_features = features_data["selected_features"] if isinstance(features_data, dict) else features_data

with open('data/class_weights.json') as f:
    class_weights = json.load(f)

# Extract all metrics dynamically
val_roc_auc = model_metrics["roc_auc"]
val_recall = model_metrics["fraud_recall"]
val_cm = model_metrics["confusion_matrix"]
val_precision = model_metrics["classification_report"]["1"]["precision"]

test_roc_auc = eval_report["roc_auc"]
test_recall_default = eval_report["default_threshold_0.5"]["fraud_recall"]
test_precision_default = eval_report["default_threshold_0.5"]["classification_report"]["1"]["precision"]
test_cm_default = eval_report["default_threshold_0.5"]["confusion_matrix"]
test_fp_default = test_cm_default[0][1]
test_fn_default = test_cm_default[1][0]

optimal_threshold = eval_report["optimal_threshold_tuning"]["optimal_threshold"]
test_recall_optimal = eval_report["optimal_threshold_tuning"]["fraud_recall"]
test_precision_optimal = eval_report["optimal_threshold_tuning"]["classification_report"]["1"]["precision"]
test_cm_optimal = eval_report["optimal_threshold_tuning"]["confusion_matrix"]
test_fp_optimal = test_cm_optimal[0][1]
test_fn_optimal = test_cm_optimal[1][0]
best_f1 = eval_report["optimal_threshold_tuning"]["best_f1_score"]

n_features = len(selected_features)
features_str = ", ".join(selected_features)

report_task = Task(
    description=f'''
    Write a complete pptxgenjs JavaScript script to create a professional end-to-end fraud detection ML pipeline deck.

    Use this dynamically loaded data:

    DATASET:
    - Training size: 1,296,675 transactions
    - Fraud rate: 0.58% (severe class imbalance)
    - Class weight ratio: {class_weights}

    FEATURES:
    - Number of selected features: {n_features}
    - Selected features: {features_str}
    - All VIF scores near 1.0 (no multicollinearity)
    - All p-values significant (p < 0.05)

    VALIDATION METRICS (training holdout):
    - ROC-AUC: {val_roc_auc:.4f}
    - Fraud Recall: {val_recall:.4f}
    - Fraud Precision: {val_precision:.4f}
    - Confusion matrix: {val_cm}

    TEST METRICS (unseen fraudTest.csv):
    - ROC-AUC: {test_roc_auc:.4f}
    - Default threshold (0.5):
        Recall: {test_recall_default:.4f}
        Precision: {test_precision_default:.4f}
        False Positives: {test_fp_default}
        False Negatives: {test_fn_default}
    - Optimal threshold ({optimal_threshold:.2f}):
        Recall: {test_recall_optimal:.4f}
        Precision: {test_precision_optimal:.4f}
        False Positives: {test_fp_optimal}
        False Negatives: {test_fn_optimal}
        Best F1: {best_f1:.4f}

    Create these 10 slides using the Midnight Executive palette (navy 1E2761, ice blue CADCFC, white FFFFFF):

    Slide 1 — TITLE (dark navy bg)
    "Credit Card Fraud Detection" large title, "End-to-End ML Pipeline Report" subtitle

    Slide 2 — THE PROBLEM (light bg)
    Large stat callouts: 1.29M transactions, 0.58% fraud rate, 99.42% legitimate
    Explain why class imbalance makes this hard

    Slide 3 — PIPELINE OVERVIEW (light bg)
    5-step horizontal flow diagram: ETL → Statistical Testing → Model Training → Evaluation → Reporting
    One line description under each step

    Slide 4 — ETL & FEATURE ENGINEERING (light bg)
    Two columns: left = columns dropped and why, right = new features engineered
    (distance, age, hour, day_of_week, month from raw coordinates and datetime)

    Slide 5 — STATISTICAL TESTING & FEATURE SELECTION (light bg)
    Stat callouts: {n_features} features selected, VIF near 1.0, all p-values < 0.05
    List the {n_features} selected features: {features_str}

    Slide 6 — MODEL TRAINING (light bg)
    Random Forest, chronological 80/20 split, class weighting used
    Validation results as large stat callouts: ROC-AUC {val_roc_auc:.4f}, Recall {val_recall:.4f}

    Slide 7 — TEST RESULTS (light bg)
    Two column layout: validation vs test comparison
    Show generalisation held — minimal performance drop

    Slide 8 — THRESHOLD TUNING (light bg)
    Two column comparison table:
    Default (0.5): Recall {test_recall_default:.4f}, Precision {test_precision_default:.4f}, FP {test_fp_default}
    Optimal ({optimal_threshold:.2f}): Recall {test_recall_optimal:.4f}, Precision {test_precision_optimal:.4f}, FP {test_fp_optimal}
    Explain the business tradeoff

    Slide 9 — KEY FINDINGS (light bg)
    Large ROC-AUC callout {test_roc_auc:.4f}
    Bullet points summarising pipeline achievements

    Slide 10 — NEXT STEPS & RECOMMENDATIONS (dark navy bg)
    Recommend XGBoost as next model iteration
    Suggest threshold selection based on business priority (recall vs precision)
    Suggest retraining pipeline on combined train+test data

    Critical pptxgenjs rules — follow exactly:
    - NEVER use # prefix on hex colours (e.g. use "1E2761" not "#1E2761")
    - NEVER reuse option objects across multiple calls — always create fresh objects
    - Use bullet: true for lists, NEVER unicode bullet characters
    - Use breakLine: true between array text items
    - Large stat numbers at 60-72pt with small labels at 14pt below
    - Dark navy backgrounds for slides 1 and 10, light FFFFFF or F4F6FB for the rest
    - Minimum 0.5" margins on all slides
    - Save final file to data/fraud_detection_report.pptx

    Output only the complete runnable JavaScript script. No explanations.
    ''',
    expected_output='A complete runnable pptxgenjs JavaScript script that produces data/fraud_detection_report.pptx',
    agent=report_agent
)