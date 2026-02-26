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
    You have been provided with an EDA report from the Data Analyst agent.
    
    Based on their findings, write a complete, runnable Python script that:
    - Drops all columns identified as redundant, identifying, or leakage risks
    - Engineers any features the analyst recommended
    - Handles class imbalance using the suggested approach
    - Encodes and scales features appropriately for logistic regression
    - Prints dataset shape and class balance before and after cleaning
    - Saves the result to data/cleaned_fraud.csv

    Use the EDA report as your primary source of truth. 
    Output only the Python script with inline comments. No explanations.
    ''',
    expected_output='A complete runnable Python script based on the EDA findings.',
    agent=cleaning_agent,
    context=[eda_task]
)