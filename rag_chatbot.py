# Source: Adapted from user-provided code, LangChain documentation
#[](https://python.langchain.com/docs/how_to/qa_chat_history_how_to/) and OpenAI API guides
#[](https://platform.openai.com/docs/guides/text-generation).
import os
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from fastapi import FastAPI
from pydantic import BaseModel

# Ensure OPENAI_API_KEY is set
# os.environ["OPENAI_API_KEY"] = "your_api_key"

# Load R package documentation
urls = [
    "https://cran.r-project.org/web/packages/dplyr/dplyr.pdf",
    "https://cran.r-project.org/web/packages/ggplot2/ggplot2.pdf"
]
loader = UnstructuredURLLoader(urls=urls)
documents = loader.load()

# Split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Create embeddings and vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", max_retries=3)
db = Chroma.from_documents(texts, embeddings, persist_directory="chroma_db")
db.persist()

# Set up retriever
retriever = db.as_retriever()

# Set up LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Define system prompt
system_prompt = (
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Use three sentences maximum and keep the answer concise. "
    "Context: {context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Create RAG chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
qa = create_retrieval_chain(retriever, question_answer_chain)

# FastAPI app
app = FastAPI()

class Question(BaseModel):
    question: str

@app.post("/ask")
def ask_question(question: Question):
    result = qa.invoke({"input": question.question})
    answer = result["answer"]
    sources = [doc.metadata for doc in result["context"]]
    return {"answer": answer, "sources": sources}