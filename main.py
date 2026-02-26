from crewai import Crew
from tasks import eda_task, cleaning_task
from agents import eda_agent, cleaning_agent
import re   

def extract_and_save_code(text, filename="agent_generated_code.py"):
    match = re.search(r'```python(.*?)```', text, re.DOTALL)
    if match:
        code = match.group(1).strip()
        with open(filename, 'w') as f:
            f.write(code)
        print(f"\nCode saved to {filename}")
    else:
        print("\n No python code block found in agent output")

if __name__ == "__main__":
    crew = Crew(
        agents=[eda_agent, cleaning_agent],
        tasks=[eda_task, cleaning_task],
        verbose=True
    )
    result = crew.kickoff()
    print(result)
    extract_and_save_code(str(result))