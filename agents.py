from crewai import Agent
from config import llm

eda_agent = Agent(
    role="Data Analyst",
    goal="Analyse the raw credit card dataset and produce an explicit, unambiguous cleaning plan",
    backstory="""You are a data analyst preparing a credit card dataset for logistic regression fraud detection.
    You know that the following types of columns must always be dropped:
    - PII (names, addresses, date of birth)
    - Unique identifiers (transaction IDs, card numbers)
    - High cardinality text columns that cannot be meaningfully encoded (merchant names, job titles, city names)
    - Columns redundant with engineered features (raw timestamps, unix time)
    - Raw coordinate columns that will be replaced by distance features
    You are explicit and decisive in your recommendations. You always produce a clearly structured report.""",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

cleaning_agent = Agent(
    name = "Data Cleaning Agent",
    role = "Data Engineer",
    goal="Clean and preprocess the credit card dataset for logistic regression analysis",
    backstory = "You are a data engineer responsible for cleaning and preprocessing raw credit card datasets. You ensure data quality, handle missing values, remove duplicates, and prepare the dataset for modeling.",
    verbose = True,
    allow_delegation = False,
    llm = llm
)

stats_agent = Agent(
    role="Statistical Analyst",
    goal="Validate the cleaned dataset and identify the most statistically significant features for logistic regression",
    backstory="You are a statistician specialising in feature selection and validation for binary classification problems. You use statistical tests to identify which features are truly predictive and flag any issues like multicollinearity that could harm logistic regression performance.",
    verbose=True,
    allow_delegation=False,
    llm=llm

)

model_agent = Agent(
    role="Machine Learning Engineer",
    goal="Train a logistic regression model for fraud detection using the selected features and class weights",
    backstory="""You are a machine learning engineer specialising in binary classification problems.
    You write clean, production quality sklearn code. You know that for imbalanced fraud detection:
    - Class weights must be applied to penalise missed fraud cases
    - The model must be saved for use by the evaluation agent
    - Precision, recall and F1 must be reported on a validation split
    - Recall is the most important metric — missing fraud is worse than a false alarm""",
    verbose=True,
    allow_delegation=False,
    llm=llm
)