import re
from typing import List, Dict, Any
from langchain_core.embeddings import Embeddings
from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult, ChatGeneration

class FakeEmbeddings(Embeddings):
    _dimension = 10 
    def embed_documents(self, texts: List[str]) -> List[List[float]]: return [[0.1] * self._dimension for _ in texts]
    def embed_query(self, text: str) -> List[float]: return [0.1] * self._dimension

class FakeChatModel(BaseChatModel):
    def _generate(self, messages: List[BaseMessage], **kwargs) -> ChatResult:
        full_prompt = messages[-1].content
        lower_prompt = full_prompt.lower()
        text = "This is a generic mock response." # Default fallback

        # --- THIS IS THE FIX ---
        # Only check for keywords inside the <question> tag for routing.
        if "classify it as" in lower_prompt:
            # Extract the actual question from the prompt
            match = re.search(r"<question>(.*?)<\/question>", lower_prompt)
            question_text = match.group(1) if match else ""

            # Now, check for keywords ONLY in the extracted question
            if "assign" in question_text or "module" in question_text:
                text = "course_modules"
            elif "ggplot" in question_text or "dplyr" in question_text:
                text = "r_packages"
            else:
                text = "general_knowledge" # This will now work correctly

        elif "course material context" in lower_prompt:
            text = "Mock answer from course module."
        elif "r package manual context" in lower_prompt:
            text = "Mock answer from R package manual."
        elif "suggested next questions" in lower_prompt:
            text = "1. Mock Suggestion 1\n2. Mock Suggestion 2"
            
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def _llm_type(self) -> str: return "fake-chat-model"

def get_mock_llm():
    """Returns a configured mock LLM."""
    print("--- Creating Mock LLM ---")
    return FakeChatModel()

def get_mock_embeddings():
    """Returns a configured mock Embeddings model."""
    print("--- Creating Mock Embeddings ---")
    return FakeEmbeddings()