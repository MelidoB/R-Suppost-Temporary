# src/config.py
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

def get_production_llm():
    """Returns a configured production LLM."""
    print("--- Creating Production LLM ---")
    return ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

def get_production_embeddings():
    """Returns a configured production Embeddings model."""
    print("--- Creating Production Embeddings ---")
    return OpenAIEmbeddings(model="text-embedding-3-large")