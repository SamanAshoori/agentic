import subprocess

steps = [
    ("etl.py",      "ETL — Cleaning & Preprocessing"),
    ("stats.py",    "Statistical Analysis & Feature Selection"),
    ("model.py",    "Model Training"),
    ("evaluate.py", "Model Evaluation"),
]

for script, label in steps:
    print(f"\n{'='*50}\n{label}\n{'='*50}")
    subprocess.run(["python3", script], check=True)

print("\n Pipeline complete")