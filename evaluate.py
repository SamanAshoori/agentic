import re
from crewai import Crew
from agents import eval_agent
from tasks.eval_tasks import eval_task

def extract_and_save_code(text, filename="eval_generated_code.py"):
    match = re.search(r'```python(.*?)```', text, re.DOTALL)
    if match:
        code = match.group(1).strip()
        with open(filename, 'w') as f:
            f.write(code)
        print(f"\n✅ Code saved to {filename}")
    else:
        print("\n⚠️ No python code block found in agent output")

if __name__ == "__main__":
    crew = Crew(
        agents=[eval_agent],
        tasks=[eval_task],
        verbose=True
    )

    print("Starting evaluation agent...")
    result = crew.kickoff()
    extract_and_save_code(str(result))