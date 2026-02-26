from crewai import Task
from agents import stats_agent

stats_task = Task(
    description='''
    Write a complete Python script to statistically validate data/cleaned_fraud.csv
    and select the best features for logistic regression.

    The script must:
    1. Load data/cleaned_fraud.csv
    2. Compute a correlation matrix and flag feature pairs with correlation > 0.85
    3. Run chi-squared tests for all categorical/binary features against is_fraud, report p-values
    4. Compute VIF (Variance Inflation Factor) for all numerical features, flag any with VIF > 5
    5. Rank features by predictive value using mutual information scores against is_fraud
    6. Output a final recommended feature list excluding:
       - One from each highly correlated pair
       - Features with non-significant p-values (p > 0.05)
       - Features with VIF > 5
    7. Save the recommended feature list to data/selected_features.json
    8. Print a clear summary report of all findings

    Output only the complete runnable Python script with inline comments. No explanations.
    ''',
    expected_output='A complete runnable Python script that produces data/selected_features.json',
    agent=stats_agent
)