import pandas as pd
from crewai import Task
from agents import eda_agent, cleaning_agent

data = pd.read_csv('data/fraudTrain.csv')
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
    expected_output='''A markdown report with these exact sections:
    ## Target Variable
    ## Class Imbalance
    ## Columns To Drop (list every column to remove with one line reason each)
    ## Feature Engineering Steps (numbered, specific)
    ## Cleaning Plan (numbered steps for the cleaning agent to follow exactly)
    ''',
    agent=eda_agent
)

cleaning_task = Task(
    description='''
    You will receive an EDA report. Follow it exactly.

    - Drop every column listed under "Columns To Drop" — no exceptions
    - Follow every step listed under "Feature Engineering Steps" in order
    - Follow every step listed under "Cleaning Plan" in order
    - Handle class imbalance with class weighting (NOT SMOTE)
    - Save cleaned data to data/cleaned_fraud.csv
    - Save class weights to data/class_weights.json

    Output only the complete runnable Python script. No explanations.
    ''',
    expected_output='A complete runnable Python script that produces data/cleaned_fraud.csv and data/class_weights.json',
    agent=cleaning_agent,
    context=[eda_task]
)