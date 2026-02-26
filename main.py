import pandas as pd
from crewai import Agent,Task,Crew
from langchain_community.llms import ollama

#Load training data
data = pd.read_csv('training_data.csv')

#create quick summary of data for the agent
summary = data.describe().to_string()
data_head = data.head().to_markdown()

#local llm
llm = ollama.Ollama(model = "llama3.2")

#define agent params
eda_agent = Agent(
    role = "Data Analyst",
    goal="Analyse the raw credit card dataset and identify key features,target variables and potential data quality issues",
    backstory = "You are a data analyst tasked with analyzing a raw credit card dataset in prepartation for logistic regression. " \
    "You communicate clearly and in a concise manner. " \
    "You have access to a summary of the dataset and the first few rows of the data to assist you in your analysis." 
    verbose = True,
    allow_delegation = False,
    llm = llm
)
#create task
eda_task = Task(
    description=f'''
    Review the following sample and statistical summary of the credit card dataset and provide insights on key features, 
    target variables and potential data quality issues.
    Data Sample:
    {data_head}
    Data Summary:
    {summary}

    Please provide:
    Identification of target variable for fraud detection.
    A brief analysis of class imbalances
    List of columns that may cause data leakage, strictly identifying or redundant
    And a concise plan for the data cleaning agent
    ''',
    expected_output='A clear markdown report summarizing the insights and analysis of the dataset, including identification of target variable, class imbalance analysis, potential data leakage columns, and a plan for data cleaning.',
    agent=eda_agent
)

if __name__ == "__main__":
    #create crew and add task
    crew = Crew()
    crew.add_task(eda_task)

    print("Starting the crew to analyze the dataset...")
    #run the crew
    result = crew.kickoff()

    print("\n === EDA AGENT OUTPUT === \n")
    print(result)