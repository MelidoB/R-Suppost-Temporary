# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any
from dotenv import load_dotenv

# --- 1. Load Environment and Blueprints ---
load_dotenv()
from src.config import get_production_llm, get_production_embeddings
from src.data_loader import get_retrievers
from src.chains import create_master_chain

# --- 2. Assemble the Application ---
print("="*50)
print("ASSEMBLING PRODUCTION APPLICATION")
print("="*50)
# Create the real AI components
llm = get_production_llm()
embeddings = get_production_embeddings()
# Inject them to build the retrievers and chains
retrievers = get_retrievers(embeddings, is_mock=False)
master_chain, suggestion_chain = create_master_chain(llm, retrievers)

# --- 3. FastAPI Application (remains the same) ---
app = FastAPI(title="Hybrid R Chatbot - Production Version")
class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
def chat(request: ChatRequest) -> Dict[str, Any]:
    result = master_chain.invoke({"input": request.question})
    final_answer, sources = "", []
    if isinstance(result, dict) and "answer" in result:
        final_answer = result.get("answer", "Could not find a specific answer.")
        if "context" in result: sources = [doc.metadata for doc in result["context"]]
    else:
        final_answer = result.content if hasattr(result, "content") else str(result)
    try:
        suggestions_result = suggestion_chain.invoke({"input": request.question, "answer": final_answer})
        suggestions_text = suggestions_result.content.strip()
        suggested_prompts = [line.strip() for line in suggestions_text.splitlines() if line.strip()]
    except Exception as e:
        print(f"Error generating suggestions: {e}")
        suggested_prompts = []
    return {"answer": final_answer, "sources": sources, "suggested_prompts": suggested_prompts}