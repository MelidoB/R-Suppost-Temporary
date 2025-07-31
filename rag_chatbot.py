import os
import json
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any, Literal

# --- Core LangChain Imports ---
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- 1. INITIALIZE AI MODELS ---
# This requires the OPENAI_API_KEY environment variable to be set.
# These classes are the main interface to OpenAI's services.
# Source for LangChain + OpenAI integration: https://python.langchain.com/v0.2/docs/integrations/llms/openai/
# and https://python.langchain.com/v0.2/docs/integrations/text_embedding/openai/
print("Initializing AI models...")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# --- 2. SETUP THE RAG "PRIVATE EXPERT" FROM JSON DATA ---
persist_directory = "course_module_db"
if not os.path.exists(persist_directory):
    print(f"Database not found. Building new database from 'course_modules.json'...")
    
    # This custom logic loads our structured JSON data.
    with open('course_modules.json', 'r') as f:
        course_data = json.load(f)

    # We convert our custom data into LangChain's standard 'Document' object.
    # Each Document holds page_content (what is searched) and metadata (the extra info).
    # Source for Document objects: https://python.langchain.com/v0.2/docs/concepts/#documents
    all_documents = []
    for module in course_data:
        for segment in module['transcript_segments']:
            metadata = {
                "source_module": module['module_title'],
                "timestamp": segment['timestamp'],
                "video_url": module.get('video_url', '')
            }
            doc = Document(page_content=segment['content'], metadata=metadata)
            all_documents.append(doc)
    
    print(f"Creating embeddings for {len(all_documents)} document segments...")
    # This creates the vector database using Chroma. The .from_documents method handles
    # embedding all documents and storing them efficiently.
    # Source for Chroma integration: https://python.langchain.com/v0.2/docs/integrations/vectorstores/chroma/
    db = Chroma.from_documents(documents=all_documents, embedding=embeddings, persist_directory=persist_directory)
    print("Database built and saved successfully.")
else:
    print(f"Loading existing database from '{persist_directory}'...")
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    print("Database loaded successfully.")

# A retriever is an interface that fetches documents based on a query.
# Source for retrievers: https://python.langchain.com/v0.2/docs/concepts/#retrievers
retriever = db.as_retriever()

# --- 3. DEFINE ALL THE CHAINS (THE EXPERTS) USING LCEL ---
# The following chains are built using LangChain Expression Language (LCEL),
# which uses the pipe (|) symbol to connect components.
# Source for LCEL: https://python.langchain.com/v0.2/docs/concepts/#lcel

# A. The Router Chain: Decides where to send the question.
router_prompt = PromptTemplate.from_template(
    # ... (prompt text as before) ...
)
router = router_prompt | llm

# B. The RAG Chain (Private Expert)
# This chain is constructed using two helper functions for clarity and power.
# Source for create_retrieval_chain: https://python.langchain.com/v0.2/docs/how_to/qa_sources/
rag_prompt = ChatPromptTemplate.from_template(
    """You are an expert teaching assistant. Use the following context from the course material to answer the question.
If the context is not relevant, politely say you cannot answer from the provided material.

Context:
{context}

Question: {question}
Answer:"""
)
# create_stuff_documents_chain "stuffs" the retrieved documents into the prompt's context.
# Source: https://python.langchain.com/v0.2/docs/how_to/stuff_documents/
question_answer_chain = create_stuff_documents_chain(llm, rag_prompt)
# create_retrieval_chain combines the retriever and the document-stuffing chain.
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# C. The General Knowledge Chain (Public Generalist)
general_prompt = PromptTemplate.from_template(
    """You are a helpful general knowledge assistant. Provide a helpful, concise answer to the following question.
Question: {question}
Answer:"""
)
general_chain = general_prompt | llm

# D. The Suggestion Generation Chain
suggestion_prompt = PromptTemplate.from_template(
    """Based on the following question and its answer, suggest three logical follow-up questions a student might ask next.
Present them as a simple numbered list. Do not add any extra text or introductory phrases.
QUESTION: {question}
ANSWER: {answer}
SUGGESTED NEXT QUESTIONS:"""
)
suggestion_chain = suggestion_prompt | llm

# --- 4. CREATE THE MASTER HYBRID CHAIN WITH A RUNNABLE BRANCH ---

# This custom Python function provides the logic for the router's decision.
def route(info: Dict[str, Any]) -> Literal["course_specific", "general_knowledge"]:
    if "course_specific" in info["topic"].content.lower():
        return "course_specific"
    else:
        return "general_knowledge"

# RunnableBranch is the modern way to create conditional chains (routing).
# It takes pairs of (condition, runnable) and executes the first one that evaluates to true.
# Source for RunnableBranch: https://python.langchain.com/v0.2/docs/how_to/routing/
master_chain = {"topic": router, "question": lambda x: x["question"]} | RunnableBranch(
    (lambda x: route(x) == "course_specific", rag_chain),
    (lambda x: route(x) == "general_knowledge", general_chain),
    general_chain # Default route
)

# --- 5. SETUP THE FASTAPI APPLICATION ---
# FastAPI is a modern, high-performance web framework for building APIs.
# Source: https://fastapi.tiangolo.com/
app = FastAPI(
    title="Hybrid R Chatbot - Production Version",
    description="An AI chatbot that uses RAG for course content and a general model for other questions."
)

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
def chat(request: ChatRequest) -> Dict[str, Any]:
    # Invoke the master chain to get a result.
    result = master_chain.invoke({"question": request.question})

    # This custom logic standardizes the output, since the RAG chain and general chain
    # return slightly different data structures.
    final_answer = ""
    sources = []
    
    if isinstance(result, dict) and "answer" in result:
        route_taken = "course_specific"
        final_answer = result.get("answer", "Could not find a specific answer in the course material.")
        if "context" in result:
            sources = [doc.metadata for doc in result["context"]]
    else:
        route_taken = "general_knowledge"
        final_answer = result.content if hasattr(result, "content") else str(result)

    # Generate follow-up suggestions
    suggestions_text = ""
    try:
        suggestions_result = suggestion_chain.invoke({"question": request.question, "answer": final_answer})
        suggestions_text = suggestions_result.content.strip()
    except Exception as e:
        print(f"Error generating suggestions: {e}")
        
    suggested_prompts = [line.strip() for line in suggestions_text.splitlines() if line.strip()]

    return {
        "answer": final_answer,
        "sources": sources,
        "route_taken": route_taken,
        "suggested_prompts": suggested_prompts
    }