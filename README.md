# Hybrid AI Chatbot for R Package Documentation & General Questions

This project is an advanced AI-powered chatbot designed to answer questions about R packages while also handling general knowledge queries. It uses a sophisticated **hybrid architecture** to provide the best possible answer.

### How It Works

The chatbot operates like an office with two expert assistants and a smart receptionist:

1.  **The "Private Expert" (RAG System):** This AI has read and indexed the official R package manuals for `dplyr` and `ggplot2`. It can answer highly specific questions about these packages and cite the exact source document.

2.  **The "Public Generalist" (Standard LLM):** This AI has broad, general knowledge about programming and many other topics. It answers questions that are not covered in the R manuals.

3.  **The "Smart Receptionist" (Router):** When a user asks a question, this router first analyzes it. It intelligently decides which expert is better suited to answer and directs the question accordingly. This ensures you get a precise, fact-based answer if possible, and a helpful general answer otherwise.

This hybrid approach provides a robust, accurate, and versatile user experience.

---

### ✅ Steps to Use the Hybrid Chatbot

1.  **Download `requirements.txt`**
    *   Make sure this file is saved in your project directory.

2.  **Install Python dependencies**
    Run this command in your terminal:
    > pip install -r requirements.txt

3.  **Install required system dependencies**
    These are needed to process PDFs:
    > sudo apt-get install poppler-utils tesseract-ocr

4.  **Set your OpenAI API key**
    Replace "your_api_key" with your actual key:
    > export OPENAI_API_KEY="your_api_key"

5.  **Run the application**
    Start the FastAPI server using Uvicorn:
    > uvicorn rag_chatbot:app --reload

    The **first time** you run this, it will take **60-90 seconds** to download and index the R manuals. Every subsequent startup will be much faster (~2-5 seconds).

---

### 📌 API Endpoints

*   `POST /chat`: The main endpoint to ask questions.
    *   **Body:** `{"question": "Your question here"}`
    *   **Returns:** The answer, the sources (if applicable), and the route taken (which expert was used).