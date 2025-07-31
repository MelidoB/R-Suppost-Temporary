# ======================================================================================
# RAG Chatbot with Hybrid Routing & Suggested Prompts - FULL CODE
# ======================================================================================
# This version implements a full-featured hybrid system. It includes:
# 1. A Router to decide between a RAG expert and a General expert.
# 2. A persistent vector store to avoid re-indexing on every startup.
# 3. A new chain that generates suggested follow-up prompts after each answer.
# ======================================================================================

import os
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chains.router import MultiRouteChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any

# --- 1. SETUP THE RAG "PRIVATE EXPERT" ---
# This section prepares the expert that knows about the R package documentation.

# Define the path for the persistent vector database
persist_directory = "chroma_db_persistent"
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Check if the database already exists to avoid rebuilding it on every startup
if os.path.exists(persist_directory):
    print(f"Loading existing vector database from '{persist_directory}'...")
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    print("Database loaded successfully.")
else:
    print(f"Database not found. Building a new one at '{persist_directory}'...")
    urls = [
        "https://cran.r-project.org/web/packages/dplyr/dplyr.pdf",
        "https://cran.r-project.org/web/packages/ggplot2/ggplot2.pdf"
    ]
    print("Downloading documents...")
    loader = UnstructuredURLLoader(urls=urls)
    documents = loader.load()
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    print("Creating embeddings and persisting the database...")
    db = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=persist_directory)
    print("Database built and saved successfully.")

# Set up the LLM
# NOTE: Using gpt-3.5-turbo is cost-effective. GPT-4 will yield higher quality routing and suggestions.
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Create a retriever from the database
retriever = db.as_retriever()

# Create the RAG chain (our "Private Expert")
rag_prompt = ChatPromptTemplate.from_template(
    """You are an assistant for R programming questions. Use the given context to answer the question. If you don't know the answer from the context, say that you don't know. Keep the answer concise and maximum three sentences.
Context: {context}
Question: {input}
Answer:"""
)
question_answer_chain = create_stuff_documents_chain(llm, rag_prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


# --- 2. SETUP THE "PUBLIC GENERALIST" EXPERT ---
# This is a simple chain that answers general knowledge questions.

general_prompt_text = """You are a helpful general knowledge assistant. The user has asked a question that is not specific to the provided R documentation. Provide a helpful, concise answer.

Question: {input}
Answer:"""
general_prompt = PromptTemplate.from_template(general_prompt_text)
general_chain = LLMChain(llm=llm, prompt=general_prompt)


# --- 3. SETUP THE HYBRID ROUTER AND SUGGESTION CHAIN ---
# This section contains the "Smart Receptionist" and the new "Suggestion Generator".

# A. The Router Logic
prompt_infos = [
    {
        "name": "specific_r_questions",
        "description": "Good for answering specific questions about the R packages dplyr or ggplot2",
        "chain": rag_chain,
    },
    {
        "name": "general_knowledge",
        "description": "Good for answering general knowledge questions, programming concepts, or anything not covered in the R documents",
        "chain": general_chain,
    },
]

router_template_str = MULTI_PROMPT_ROUTER_TEMPLATE.format(
    destinations="\n".join([f"{p['name']}: {p['description']}" for p in prompt_infos])
)
router_prompt = PromptTemplate(
    template=router_template_str,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
)
router_chain = LLMRouterChain.from_llm(llm, router_prompt)

# The master chain combines the router and the experts
master_chain = MultiRouteChain(
    router_chain=router_chain,
    destination_chains={p["name"]: p["chain"] for p in prompt_infos},
    default_chain=general_chain,
    verbose=True
)

# B. The Suggestion Generation Chain (NEW FEATURE)
suggestion_prompt_text = """Based on the following question and its answer, please suggest exactly three logical and concise follow-up questions that a beginner might ask next. Present them as a simple numbered list, with each question on a new line. Do not add any extra text or introductory phrases.

QUESTION:
{question}

ANSWER:
{answer}

SUGGESTED NEXT QUESTIONS:"""
suggestion_prompt = PromptTemplate.from_template(suggestion_prompt_text)
suggestion_chain = LLMChain(llm=llm, prompt=suggestion_prompt)


# --- 4. SETUP FASTAPI APPLICATION ---
# The API now uses both the master_chain and the new suggestion_chain.

app = FastAPI(
    title="Hybrid R Chatbot with Suggested Prompts",
    description="An AI chatbot that routes questions and suggests follow-ups."
)

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
def chat(request: ChatRequest) -> Dict[str, Any]:
    """
    Handles a question with the HYBRID chatbot.
    It routes the question, gets an answer, AND generates suggested follow-ups.
    """
    # Step 1: Get the main answer using the master router chain
    result = master_chain.invoke({"input": request.question})
    
    # Standardize the output from different chains
    final_answer = ""
    sources = []
    route_taken = result.get("destination", "default")

    if route_taken == "specific_r_questions":
        final_answer = result.get("answer", "Could not find an answer in the documents.")
        if "context" in result:
            sources = [doc.metadata for doc in result.get("context", [])]
    else:
        final_answer = result.get("text", "Sorry, I could not process your request.")
        
    # Step 2: Generate the follow-up suggestions (New Feature)
    try:
        suggestions_result = suggestion_chain.invoke({
            "question": request.question,
            "answer": final_answer
        })
        suggestions_text = suggestions_result.get("text", "").strip()
        suggested_prompts = [line.strip() for line in suggestions_text.splitlines() if line.strip()]
    except Exception as e:
        print(f"Error generating suggestions: {e}")
        suggested_prompts = []

    # Step 3: Combine everything into the final response
    return {
        "answer": final_answer,
        "sources": sources,
        "route_taken": route_taken,
        "suggested_prompts": suggested_prompts
    }