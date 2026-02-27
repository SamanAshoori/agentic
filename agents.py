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
    goal="Train a Random Forest model for fraud detection using the selected features and class weights",
    backstory="""You are a machine learning engineer specialising in binary classification problems.
    You write clean, production quality sklearn code. You know that for fraud detection:
    - Random Forest handles class imbalance natively via class_weight parameter
    - No feature scaling needed for tree based models
    - The model and feature list must be saved for the evaluation agent
    - Recall is the most important metric — missing fraud is worse than a false alarm
    - n_estimators=100 and max_depth=10 are good starting points for large datasets""",
    verbose=True,
    allow_delegation=False,
    llm=llm
)
eval_agent = Agent(
    role="Model Evaluator",
    goal="Evaluate the trained fraud detection model on unseen test data and produce a comprehensive performance report",
    backstory="""You are a model evaluation specialist who rigorously tests ML models on held-out data.
    You know that for fraud detection:
    - Recall is the primary metric — missing fraud is catastrophic
    - Precision matters too — too many false alarms erode trust
    - Threshold tuning beyond 0.5 can significantly improve the precision/recall tradeoff
    - Results must be saved clearly for the reporting agent to use""",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

report_agent = Agent(
    role="Data Storyteller",
    goal="Create a professional PowerPoint presentation summarising the fraud detection ML pipeline and results",
    backstory="""You are a data storyteller who creates compelling, visually professional presentations for business audiences.
    You write clean pptxgenjs JavaScript code to produce boardroom-ready decks.
    You know that great slides use large stat callouts, minimal text, strong colour contrast, and varied layouts.
    You never use accent lines under titles. You never use #prefix on hex colours. You never reuse option objects across calls.""",
    verbose=True,
    allow_delegation=False,
    llm=llm
)