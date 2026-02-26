from crewai import Crew
from tasks import eda_task, cleaning_task
from agents import eda_agent, cleaning_agent

if __name__ == "__main__":
    crew = Crew(
        agents=[eda_agent, cleaning_agent],
        tasks=[eda_task, cleaning_task],
        verbose=True
    )
    result = crew.kickoff()
    print(result)