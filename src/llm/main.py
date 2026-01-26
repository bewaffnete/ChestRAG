from langchain_ollama import ChatOllama
from env_config import LLM

def llm_instance():
    """
    Initializes an instance of the ChatOllama model with the specified parameters.
    """
    llm = ChatOllama(
        model=LLM,
        temperature=0.2,
        num_ctx=8192 * 2,
        base_url="http://ollama:11434",
    )
    return llm


if __name__ == "__main__":
    llm = llm_instance()