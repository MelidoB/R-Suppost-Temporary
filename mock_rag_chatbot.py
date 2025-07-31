# ======================================================================================
# RAG Chatbot with Hybrid Routing & REALISTIC MOCKING - FULL CODE
# ======================================================================================
# This version includes a more sophisticated FakeChatModel that simulates the
# exact output of the RAG chain, including sources and timestamps, allowing for

# complete end-to-end testing without an API key.
# ======================================================================================

import os
import json # Import the json library
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any, Literal

# --- Check for the MOCK AI environment variable ---
USE_MOCK_AI = os.getenv("USE_MOCK_AI", "false").lower() == "true"

# --- Conditionally import and set up models ---
if USE_MOCK_AI:
    print("="*50)
    print("RUNNING IN REALISTIC MOCK MODE - NO API KEY NEEDED")
    print("="*50)
    
    # --- Use FAKE models for testing without an API key ---
    from langchain_core.embeddings import Embeddings
    from langchain.chat_models.base import BaseChatModel
    from langchain_core.messages import AIMessage, BaseMessage
    from langchain_core.outputs import ChatResult, ChatGeneration
    
    class FakeEmbeddings(Embeddings):
        """A fake embedding class that returns consistent dimensions."""
        _dimension = 10 
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            return [[0.1] * self._dimension for _ in texts]
        def embed_query(self, text: str) -> List[float]:
            return [0.1] * self._dimension

    class FakeChatModel(BaseChatModel):
        """A smarter fake chat model that simulates RAG output."""
        def _generate(self, messages: List[BaseMessage], stop: List[str] = None, **kwargs) -> ChatResult:
            last_message_content = messages[-1].content.lower()
            response_text = "This is a generic mock response from the general chain."

            if "classify it as either" in last_message_content:
                # Simulate the router's decision
                response_text = "specific_r_questions" if "ggplot" in last_message_content or "variable" in last_message_content else "general_knowledge"
            
            # THIS IS THE KEY CHANGE: Simulate the RAG chain's structured output
            elif "context:" in last_message_content:
                # This simulates the RAG chain's output by creating a JSON string.
                # The real RAG chain would get this data from the vector store's metadata.
                response_data = {
                    "answer": "According to the course material, you should use the arrow operator (<-) to assign a value to a variable.",
                    "sources": [
                        {
                            "source_module": "Module 1: Getting Started with R",
                            "timestamp": "00:03:45"
                        }
                    ]
                }
                response_text = json.dumps(response_data) # Serialize the dictionary to a string
            
            elif "suggested next questions" in last_message_content:
                response_text = "1. Mock Suggestion 1?\n2. Mock Suggestion 2?\n3. Mock Suggestion 3?"
            
            message = AIMessage(content=response_text)
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])

        def _llm_type(self) -> str:
            return "fake-chat-model"

    # Initialize the fake models
    llm = FakeChatModel()
    embeddings = FakeEmbeddings()

else:
    # This block remains the same, for running with a real API key
    print("="*50)
    print("RUNNING IN PRODUCTION MODE - API KEY REQUIRED")
    print("="*50)
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


# --- The rest of the code is structurally the same ---
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_chroma import Chroma

# --- 1. SETUP THE RAG "PRIVATE EXPERT" ---
persist_directory = "chroma_db_persistent_mock" if USE_MOCK_AI else "chroma_db_persistent"
if not os.path.exists(persist_directory):
    print(f"Building new database at '{persist_directory}'...")
    if USE_MOCK_AI:
        texts = [Document(page_content="To assign a value to a variable, we use the arrow operator", metadata={"source_module": "Module 1", "timestamp": "00:03:45"}), Document(page_content="ggplot2 is a plotting library.", metadata={"source_module": "Module 2", "timestamp": "00:02:10"})]
    else:
        # In a real scenario, this would be your JSON loading logic
        urls = ["https://cran.r-project.org/web/packages/dplyr/dplyr.pdf", "https://cran.r-project.org/web/packages/ggplot2/ggplot2.pdf"]
        loader = UnstructuredURLLoader(urls=urls)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
    
    db = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=persist_directory)
else:
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

retriever = db.as_retriever()

# --- 2. DEFINE THE CHAINS USING MODERN LCEL SYNTAX ---
router_prompt = PromptTemplate.from_template("""Given the user question below, classify it as either `specific_r_questions` or `general_knowledge`. Do not respond with more than one word.\n<question>{question}</question>\nClassification:""")
router = router_prompt | llm

# IMPORTANT: The RAG chain's final step is just the LLM. 
# In mock mode, this will be our FakeChatModel, which returns a JSON string.
# In production mode, this will be the real OpenAI LLM.
rag_prompt = ChatPromptTemplate.from_template("""Based on the following context, answer the user's question.
Context: {context}
Question: {question}
Answer:""")
def format_docs(docs: List[Document]) -> str:
    # Here is where we would format the real documents to include metadata
    # For now, we just combine the content. The metadata is handled in the real retriever.
    return "\n\n".join(doc.page_content for doc in docs)
rag_chain = {"context": retriever | RunnableLambda(format_docs), "question": lambda x: x["question"]} | rag_prompt | llm

general_prompt = PromptTemplate.from_template("""Question: {question}\nAnswer:""")
general_chain = general_prompt | llm

suggestion_prompt = PromptTemplate.from_template("""QUESTION: {question}\nANSWER: {answer}\nSUGGESTED NEXT QUESTIONS:""")
suggestion_chain = suggestion_prompt | llm

# --- 3. COMBINE EVERYTHING WITH A RUNNABLE BRANCH ---
def route(info: Dict[str, Any]) -> Literal["specific_r_questions", "general_knowledge"]:
    topic_str = info["topic"].content if hasattr(info["topic"], "content") else str(info["topic"])
    if "specific_r_questions" in topic_str.lower():
        return "specific_r_questions"
    else:
        return "general_knowledge"

master_chain = {"topic": router, "question": lambda x: x["question"]} | RunnableBranch(
    (lambda x: route(x) == "specific_r_questions", rag_chain),
    (lambda x: route(x) == "general_knowledge", general_chain),
    general_chain
)

# --- 4. SETUP FASTAPI APPLICATION ---
app = FastAPI(title="Hybrid R Chatbot - Realistic Mock Version")
class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
def chat(request: ChatRequest) -> Dict[str, Any]:
    # Determine the route first to know how to process the result
    routing_result = router.invoke({"question": request.question})
    route_taken = "specific_r_questions" if "specific" in routing_result.content.lower() else "general_knowledge"

    # Invoke the appropriate chain
    if route_taken == "specific_r_questions":
        main_answer_result = rag_chain.invoke({"question": request.question})
    else:
        main_answer_result = general_chain.invoke({"question": request.question})
        
    raw_answer = main_answer_result.content if hasattr(main_answer_result, "content") else str(main_answer_result)
    
    # THIS IS THE KEY CHANGE: Parse the response based on the route
    final_answer = ""
    sources = []
    
    if route_taken == "specific_r_questions":
        try:
            # Try to parse the special JSON string from our smart mock
            parsed_output = json.loads(raw_answer)
            final_answer = parsed_output['answer']
            sources = parsed_output['sources']
        except (json.JSONDecodeError, TypeError):
            # Fallback for the real API or if the mock fails
            final_answer = raw_answer
            sources = [] # In a real app, you'd get this from the retriever step
    else:
        # The general chain always returns a simple string
        final_answer = raw_answer

    # Generate suggestions based on the final, clean answer
    suggestions_result = suggestion_chain.invoke({"question": request.question, "answer": final_answer})
    suggestions_text = (suggestions_result.content if hasattr(suggestions_result, "content") else str(suggestions_result)).strip()
    suggested_prompts = [line.strip() for line in suggestions_text.splitlines() if line.strip()]

    return {
        "answer": final_answer,
        "sources": sources,
        "route_taken": route_taken,
        "suggested_prompts": suggested_prompts
    }