# src/data_loader.py
import os
import json
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, JSONLoader

# --- THIS IS THE FIX ---
# REMOVE the old, incorrect import:
# from src.config import embeddings, USE_MOCK_AI  <-- DELETE THIS LINE

VECTOR_STORE_ROOT = "./vector_stores"
KNOWLEDGE_BASE_ROOT = "./knowledge_base"

# The function now RECEIVES embeddings and is_mock as arguments
def get_retrievers(embeddings, is_mock=False):
    """
    Builds or loads vector databases for all knowledge sources
    and returns a dictionary of configured retrievers.
    """
    db_suffix = "_mock" if is_mock else ""
    
    knowledge_bases = [
        {"name": "r_packages", "loader_class": PyPDFLoader},
        {"name": "course_modules", "loader_class": JSONLoader, "loader_kwargs": {"jq_schema": '.transcript_segments[]', "text_content": False}}
    ]

    retrievers = {}
    print("Initializing all knowledge base retrievers...")

    for kb in knowledge_bases:
        kb_name = kb["name"]
        source_path = os.path.join(KNOWLEDGE_BASE_ROOT, kb_name)
        db_path = os.path.join(VECTOR_STORE_ROOT, f"{kb_name}_db{db_suffix}")

        if not os.path.exists(db_path):
            print(f"Database for '{kb_name}' not found. Building...")
            all_documents = []

            if not os.path.exists(source_path) or not os.listdir(source_path):
                 if is_mock:
                    print(f"Mock mode: Creating dummy data for '{kb_name}'.")
                    all_documents = [Document(page_content=f"This is a mock document for {kb_name}.")]
                 else:
                    raise FileNotFoundError(f"Source directory '{source_path}' is empty or missing.")
            else:
                for filename in os.listdir(source_path):
                    file_path = os.path.join(source_path, filename)
                    if os.path.isfile(file_path):
                        print(f"  - Loading file: {filename}")
                        loader_kwargs = kb.get("loader_kwargs", {})
                        
                        if kb["loader_class"] == JSONLoader:
                             with open(file_path, 'r') as f:
                                 data = json.load(f)
                                 module_title = data.get("module_title", "Unknown Module")
                                 for segment in data.get("transcript_segments", []):
                                     metadata = {"source_module": module_title, "timestamp": segment.get("timestamp", "N/A")}
                                     all_documents.append(Document(page_content=segment["content"], metadata=metadata))
                        else:
                            loader = kb["loader_class"](file_path)
                            all_documents.extend(loader.load())

            print(f"Creating embeddings for '{kb_name}' with {len(all_documents)} document(s)...")
            db = Chroma.from_documents(documents=all_documents, embedding=embeddings, persist_directory=db_path)
        else:
            print(f"Loading existing database for '{kb_name}'...")
            db = Chroma(persist_directory=db_path, embedding_function=embeddings)
        
        retrievers[kb_name] = db.as_retriever()
        print(f"Retriever for '{kb_name}' is ready.")

    return retrievers