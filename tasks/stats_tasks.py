from crewai import Task
from agents import stats_agent

stats_task = Task(
    description='''
    Write a complete Python script to statistically validate data/cleaned_fraud.csv
    and select the best features for logistic regression.

    The script must:
    1. Load data/cleaned_fraud.csv
    2. Compute a correlation matrix and flag feature pairs with correlation > 0.85
    3. Run chi-squared tests for categorical/binary features against is_fraud using scipy.stats.chi2_contingency
    4. Compute VIF using sklearn only — use this exact approach:
       from sklearn.linear_model import LinearRegression
       vif for each feature = 1 / (1 - LinearRegression().fit(X_others, X_feature).score(X_others, X_feature))
    5. Rank features using mutual_info_classif from sklearn.feature_selection
    6. Exclude from final list:
       - One from each highly correlated pair (correlation > 0.85)
       - Features with p-value > 0.05
       - Features with VIF > 5
    7. Save recommended feature list to data/selected_features.json
    8. Print a clear summary report

    Use only: pandas, numpy, scipy, sklearn. Do NOT use statsmodels.
    - Encode ALL string/object columns using LabelEncoder before saving — 
    no string columns should remain in cleaned_fraud.csv
    Output only the complete runnable Python script with inline comments. No explanations.
    ''',
    expected_output='A complete runnable Python script that produces data/selected_features.json',
    agent=stats_agent
)