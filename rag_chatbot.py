# ======================================================================================
# RAG Chatbot with Hybrid Routing - FULL CODE
# ======================================================================================
# This version implements a hybrid system. It uses a "Router" to first decide
# if a question is specific to the R documentation or if it's a general question.
# It then routes the query to the appropriate "expert" chain.
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
# NOTE: Using gpt-3.5-turbo is cost-effective. GPT-4 will yield higher quality routing.
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
# This is a new, simple chain that answers general knowledge questions.

general_prompt_text = """You are a helpful general knowledge assistant. The user has asked a question that is not specific to the provided R documentation. Provide a helpful, concise answer.

Question: {input}
Answer:"""
general_prompt = PromptTemplate.from_template(general_prompt_text)
general_chain = LLMChain(llm=llm, prompt=general_prompt)


# --- 3. SETUP THE HYBRID ROUTER LOGIC ---
# This is the "Smart Receptionist" that decides which expert to use.

# Define the information for each destination (expert)
prompt_infos = [
    {
        "name": "specific_r_questions",
        "description": "Good for answering specific questions about the R packages dplyr or ggplot2",
        "chain": rag_chain, # The RAG expert
    },
    {
        "name": "general_knowledge",
        "description": "Good for answering general knowledge questions, programming concepts, or anything not covered in the R documents",
        "chain": general_chain, # The General expert
    },
]

# Create the template for the router prompt
router_template_str = MULTI_PROMPT_ROUTER_TEMPLATE.format(
    destinations="\n".join([f"{p['name']}: {p['description']}" for p in prompt_infos])
)
router_prompt = PromptTemplate(
    template=router_template_str,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
)

# Create the router chain, which uses an LLM to decide the route
router_chain = LLMRouterChain.from_llm(llm, router_prompt)

# Create the final master chain that combines the router and the experts
master_chain = MultiRouteChain(
    router_chain=router_chain,
    destination_chains={p["name"]: p["chain"] for p in prompt_infos},
    default_chain=general_chain, # If the router is confused, it will default here
    verbose=True # Set to True to see the router's decision-making in the console
)


# --- 4. SETUP FASTAPI APPLICATION ---
# The API now uses the 'master_chain' to handle all incoming queries.

app = FastAPI(
    title="Hybrid R Chatbot",
    description="An AI chatbot that uses a router to answer questions about R packages or general knowledge."
)

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
def chat(request: ChatRequest) -> Dict[str, Any]:
    """
    Handles a question with the HYBRID chatbot.
    It first routes the question to the appropriate expert (RAG or General)
    and then returns the answer.
    """
    # The master_chain handles everything: routing and answering.
    result = master_chain.invoke({"input": request.question})
    
    # The output format is different depending on the chain called.
    # We need to standardize the output.
    final_answer = ""
    sources = []
    route_taken = result.get("destination", "default")

    if route_taken == "specific_r_questions":
        final_answer = result.get("answer", "Could not find an answer in the documents.")
        if "context" in result:
            sources = [doc.metadata for doc in result.get("context", [])]
    else: # general_knowledge or default_chain
        final_answer = result.get("text", "Sorry, I could not process your request.")
        
    return {"answer": final_answer, "sources": sources, "route_taken": route_taken}