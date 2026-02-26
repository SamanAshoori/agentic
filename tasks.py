import pandas as pd
from crewai import Task
from agents import eda_agent, cleaning_agent

# Load data and build context for agents
data = pd.read_csv('fraudTrain.csv')
summary = data.describe().to_string()
data_head = data.head(3).to_markdown()

eda_task = Task(
    description=f'''
    Review the following sample and statistical summary of the credit card dataset and provide insights.

    Data Sample:
{data_head}

    Data Summary:
{summary}

    Please provide:
    - Identification of target variable for fraud detection
    - A brief analysis of class imbalances
    - List of columns that may cause data leakage, or are identifying/redundant
    - A concise plan for the data cleaning agent
    ''',
    expected_output='A clear markdown report with: target variable, class imbalance analysis, leakage columns, and cleaning plan.',
    agent=eda_agent
)

cleaning_task = Task(
    description='''
    Based on the EDA report from the previous task, write a complete Python script to clean fraudTrain.csv.

    The script must:
    - Drop all columns identified as redundant, identifying, or leakage risks
    - Engineer features from datetime and coordinate columns as recommended
    - Handle class imbalance using class weighting (NOT SMOTE) — compute and save weights for the model training agent
    - Keep the cleaned dataset human readable: no scaling, preserve original value formats where possible
    - Print dataset shape and class balance before and after
    - Save cleaned data to cleaned_fraud.csv
    - Save class weights to class_weights.json
    - Use sklearn's compute_class_weight with keyword arguments: 
        class_weight='balanced', 
        classes=np.unique(y), 
        y=y
    )
    - When saving class weights to JSON, convert numpy types first: 
    {int(k): float(v) for k, v in class_weights_dict.items()}


    Use the EDA report as your primary source of truth.
    Output only the complete runnable Python script with inline comments. No explanations.
    ''',
    expected_output='A complete runnable Python script that produces a human-readable cleaned_fraud.csv and class_weights.json',
    agent=cleaning_agent,
    context=[eda_task]
)