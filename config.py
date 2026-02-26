from crewai import LLM

#local llm
llm = LLM(
    model="ollama/gemma3:12b",
    base_url="http://localhost:11434"
)
