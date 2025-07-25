# ======================================================================================
# RAG Chatbot for R Package Documentation - FULL OPTIMIZED CODE
# ======================================================================================
# Source: Adapted from user-provided code, LangChain documentation
# (https://python.langchain.com/docs/how_to/qa_chat_history_how_to/) and OpenAI API guides
# (https://platform.openai.com/docs/guides/text-generation).
#
# This version includes persistence logic to avoid re-indexing on every startup.
# ======================================================================================

import os
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Tuple

# Ensure OPENAI_API_KEY is set in your environment variables
# For example: export OPENAI_API_KEY="your_api_key"
# os.environ["OPENAI_API_KEY"] = "your_api_key" 

# --- 1. SETUP DATABASE WITH PERSISTENCE ---

# Define the path for the persistent vector database
persist_directory = "chroma_db_persistent"
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Check if the database already exists.
# If it does, load it. If not, build it from scratch.
if os.path.exists(persist_directory):
    # Load the existing database from disk
    print(f"Loading existing vector database from '{persist_directory}'...")
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    print("Database loaded successfully.")
else:
    # If database doesn't exist, create and populate it
    print(f"Database not found. Building a new one at '{persist_directory}'...")

    # Define the URLs for the R package documentation
    urls = [
        "https://cran.r-project.org/web/packages/dplyr/dplyr.pdf",
        "https://cran.r-project.org/web/packages/ggplot2/ggplot2.pdf"
    ]
    
    # Load the documents from the URLs
    print("Downloading documents...")
    loader = UnstructuredURLLoader(urls=urls)
    documents = loader.load()

    # Split the documents into smaller text chunks
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # Create the vector store with the text chunks and embeddings, and save it to disk
    print("Creating embeddings and persisting the database. This may take a moment...")
    db = Chroma.from_documents(
        documents=texts, 
        embedding=embeddings, 
        persist_directory=persist_directory
    )
    print("Database built and saved successfully.")


# --- 2. SETUP RETRIEVER AND CONVERSATIONAL CHAIN ---

# Create a retriever from the loaded/created database
retriever = db.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 relevant chunks

# Set up the Large Language Model (LLM)
# NOTE: Switched to gpt-3.5-turbo for better cost-effectiveness.
# Change back to "gpt-4" for higher quality but higher cost.
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Prompt for reformulating a question based on chat history (for the retriever)
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Prompt for answering the question based on retrieved context (for the final answer)
qa_system_prompt = (
    "You are an assistant for R programming questions. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise."
    "\n\n"
    "Context: {context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create the chains that will put everything together
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


# --- 3. SETUP FASTAPI APPLICATION AND ENDPOINTS ---

app = FastAPI(
    title="RAG Chatbot for R Package Documentation",
    description="An AI chatbot to answer questions about R packages and provide learning resources."
)

# Pydantic models for robust API request validation
class ChatRequest(BaseModel):
    question: str
    history: List[Tuple[str, str]] = []

class Feedback(BaseModel):
    question: str
    answer: str
    is_correct: bool

@app.get("/modules", summary="Get R Learning Modules")
def get_modules():
    """
    Returns a static list of R programming video lectures from Johns Hopkins University.
    """
    # Source for video lectures: Johns Hopkins University via Coursera
    # https://www.coursera.org/learn/r-programming
    return {
        "academy": "Johns Hopkins University",
        "course_name": "R Programming",
        "modules": [
            {"week": 1, "title": "Background, Getting Started, and Nuts & Bolts"},
            {"week": 2, "title": "Programming with R"},
            {"week": 3, "title": "Loop Functions and Debugging"},
            {"week": 4, "title": "Simulation & Profiling"}
        ]
    }

@app.post("/chat", summary="Chat with the R Documentation Bot")
def chat(request: ChatRequest):
    """
    Handles a conversational turn with the chatbot.
    It takes the user's question and chat history, and returns an answer based on the R package documents.
    """
    # Convert the list-of-tuples history into the format LangChain expects
    chat_history = [HumanMessage(content=q) if role == 'human' else AIMessage(content=a) for role, (q, a) in enumerate(request.history)]
    
    # Invoke the RAG chain to get the answer
    result = rag_chain.invoke({"input": request.question, "chat_history": chat_history})
    
    # Extract and return the answer and the sources used
    answer = result.get("answer", "Sorry, I could not find an answer.")
    sources = [doc.metadata for doc in result.get("context", [])]
    
    return {"answer": answer, "sources": sources}

@app.post("/feedback", summary="Submit Feedback on an Answer")
def receive_feedback(feedback: Feedback):
    """
    An endpoint to receive student feedback on whether an answer was helpful.
    In a real application, this would be saved to a database for analysis.
    """
    print(f"Feedback received: {feedback.dict()}")
    # This is a placeholder. A real implementation would log this to a file or database.
    return {"status": "success", "message": "Thank you for your feedback!"}