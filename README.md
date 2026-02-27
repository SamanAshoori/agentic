# Agentic Credit Card Fraud Detection Pipeline

An end-to-end ML pipeline for credit card fraud detection, built using autonomous AI agents. Each stage of the pipeline is handled by a dedicated CrewAI agent powered by the Gemini API — from raw data through to a generated PowerPoint report.

Built as a personal project to explore MLOps and agentic AI workflows, inspired by an internal talk on AI at work and conveniently timed with a Principles of Data Science module.

---

## Pipeline Overview

```
etl.py → stats.py → model.py → evaluate.py → report.py
```

| Stage | Agent | Output |
|-------|-------|--------|
| ETL | Data Cleaning Specialist | `cleaned_fraud.csv`, `class_weights.json` |
| Statistical Testing | Statistical Analyst | `selected_features.json` |
| Model Training | ML Engineer | `model.pkl`, `model_metrics.json` |
| Evaluation | Model Evaluator | `eval_report.json` |
| Reporting | Data Storyteller | `fraud_detection_report.pptx` |

Each agent generates a Python or JavaScript script which is automatically saved and run. All outputs feed into the next stage.

---

## Results

| Metric | Value |
|--------|-------|
| Model | Random Forest |
| ROC-AUC (test) | 0.993 |
| Fraud Recall @ 0.5 threshold | 92.7% |
| Fraud Precision @ 0.8 threshold | 79.2% |
| False Positives @ 0.8 threshold | 429 |

The optimal threshold of 0.8 significantly reduces false positives (5,706 → 429) at the cost of some recall — a tradeoff configurable based on business priority.

---

## Tech Stack

- **CrewAI** — agent orchestration
- **Gemini API** — LLM backbone (started with Ollama locally)
- **scikit-learn** — Random Forest, feature selection, evaluation
- **pandas / numpy** — data processing
- **pptxgenjs** — PowerPoint report generation
- **Python 3.13**

---

## Project Structure

```
agentic/
├── config.py               # LLM configuration
├── agents.py               # All agent definitions
├── etl.py                  # ETL pipeline runner
├── stats.py                # Statistical testing runner
├── model.py                # Model training runner
├── evaluate.py             # Evaluation runner
├── report.py               # PowerPoint report runner
├── pipeline.py             # Full end-to-end runner
├── tasks/
│   ├── etl_tasks.py
│   ├── stats_tasks.py
│   ├── model_tasks.py
│   ├── eval_tasks.py
│   └── report_tasks.py
└── data/                   # Generated outputs (gitignored)
```

---

## Setup

**1. Clone and install dependencies:**
```bash
git clone https://github.com/SamanAshoori/Machine_Learning_Agentic_Workflow.git
cd agentic
python3 -m venv venv
source venv/bin/activate
pip install crewai "crewai[google-genai]" pandas numpy scikit-learn imbalanced-learn joblib python-dotenv
npm install pptxgenjs
```

**2. Add your Gemini API key:**
```bash
cp .env.example .env
# Add your key from https://aistudio.google.com
```

**3. Add your dataset:**

Download the [Kaggle Credit Card Fraud dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection) and place in `data/`:
```
data/fraudTrain.csv
data/fraudTest.csv
```

**4. Run the full pipeline:**
```bash
python3 pipeline.py
```

Or run individual stages:
```bash
python3 etl.py        # Clean data
python3 stats.py      # Feature selection
python3 model.py      # Train model
python3 evaluate.py   # Evaluate on test set
python3 report.py     # Generate PowerPoint
```

---

## Key Learnings

**This project was about MLOps, not the model.** The goal was building a repeatable, automated, auditable pipeline where each stage feeds the next cleanly.

**AI agents accelerate but do not replace human oversight.** Every generated script required review — wrong API signatures, incorrect operation ordering, hallucinated filenames were all common. The speed gain is real; so is the need to understand what the agent produced.

**Local vs API LLMs matter.** Ollama (llama3.2, gemma3:12b) works well for exploration but hit quality limits on code generation tasks. Gemini API produced significantly more consistent output.

---

## Next Steps

- Swap Random Forest for XGBoost for likely performance gains
- Add a hyperparameter tuning agent
- Retrain on combined train + test data for production use
- Add threshold selection logic based on configurable business rules

---

## Dataset

[Credit Card Fraud Detection — Kaggle](https://www.kaggle.com/datasets/kartik2112/fraud-detection)

Simulated credit card transactions 2019–2020, 1.29M training records, 0.58% fraud rate.
