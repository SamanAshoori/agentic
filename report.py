import re
import subprocess
from crewai import Crew
from agents import report_agent
from tasks.report_tasks import report_task

def extract_and_save_code(text, filename="report_generated_code.js"):
    match = re.search(r'```javascript(.*?)```', text, re.DOTALL)
    if not match:
        match = re.search(r'```js(.*?)```', text, re.DOTALL)
    if match:
        code = match.group(1).strip()
        with open(filename, 'w') as f:
            f.write(code)
        print(f"\n Code saved to {filename}")
        return filename
    else:
        print("\n No javascript code block found in agent output")
        return None

if __name__ == "__main__":
    crew = Crew(
        agents=[report_agent],
        tasks=[report_task],
        verbose=True
    )

    print("Starting report agent...")
    result = crew.kickoff()
    filename = extract_and_save_code(str(result))

    if filename:
        print("\nRunning generated script...")
        subprocess.run(["node", filename], check=True)
        print("\n Presentation saved to data/fraud_detection_report.pptx")