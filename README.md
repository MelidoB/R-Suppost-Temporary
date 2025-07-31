# Hybrid AI Chatbot for Course Content

This project is an advanced AI-powered chatbot designed to be a comprehensive teaching assistant for a specific course. It can answer highly specific questions about the course material while also handling general knowledge queries.

It uses a sophisticated **hybrid architecture** to provide the best and most relevant answer for any type of question.

### How It Works

The chatbot operates like an office with two expert assistants and a smart receptionist:

1.  **The "Private Expert" (RAG System):** This AI has read and indexed the course's `course_modules.json` file, which contains all the timestamped video transcripts. It can answer highly specific questions about the course content and cite the exact module and timestamp for its answer.

2.  **The "Public Generalist" (Standard LLM):** This AI has broad, general knowledge about the subject (e.g., R programming), related fields, and many other topics. It answers questions that are not covered in the local course material, such as debugging user-specific code or defining general concepts.

3.  **The "Smart Receptionist" (Router):** When a user asks a question, this router first analyzes it. It intelligently decides which expert is better suited to answer and directs the question accordingly. This ensures you get a precise, fact-based answer from the course material if possible, and a helpful general answer otherwise.

---

### Project Structure


.
├── course_module_db/ # Will be created automatically to store the indexed knowledge base.
├── course_modules.json # IMPORTANT: You must create this file to hold your course data.
├── rag_chatbot.py # The main FastAPI application file containing all the AI logic.
├── requirements.txt # A list of all the Python packages required to run the project.
├── Dockerfile # (Optional) For containerizing the application.
└── README.md # This file.

Generated code
---

### ✅ Steps to Use the Chatbot

#### 1. Create Your Course Data File
*   You **must** create a file named `course_modules.json` in the project root.
*   This file will contain the timestamped transcripts of your course lectures. Use the provided mockup as a template for the structure.

#### 2. Install Dependencies
*   **System Dependencies:** These are needed for some Python libraries to process data.
    > `sudo apt-get update && sudo apt-get install poppler-utils tesseract-ocr`
*   **Python Dependencies:** This command will install all the required Python packages.
    > `pip install -r requirements.txt`

#### 3. Set Your OpenAI API Key
*   The application needs your API key to communicate with OpenAI. Replace `"your_api_key"` with your actual key.
    > `export OPENAI_API_KEY="your_api_key"`

#### 4. Run the Application
*   Start the FastAPI server using Uvicorn:
    > `uvicorn rag_chatbot:app --reload`
*   The **first time** you run this, it will take **60-90 seconds** to read your `course_modules.json`, create embeddings, and build the knowledge base. Every subsequent startup will be much faster (~2-5 seconds).

---

### API Endpoint

The main endpoint to interact with the chatbot is `POST /chat`.

*   **URL:** `http://127.0.0.1:8000/chat`
*   **Method:** `POST`
*   **Body (Example):**
    ```json
    {
      "question": "What operator should I use to assign a value?"
    }
    ```
*   **Success Response (200 OK):** The endpoint returns a rich JSON object.

    *   **Example for a Course-Specific Question:**
        ```json
        {
          "answer": "According to the course material, you should use the arrow operator (<-)...",
          "sources": [
            {
              "source_module": "Module 1: Getting Started with R",
              "timestamp": "00:03:45",
              "video_url": "https://example.com/course/module-1"
            }
          ],
          "route_taken": "course_specific",
          "suggested_prompts": [
            "1. What is the difference between `<-` and `=` for assignment?",
            "2. How do I view the value of a variable?",
            "3. What are the rules for naming variables in R?"
          ]
        }
        ```

    *   **Example for a General Question:**
        ```json
        {
          "answer": "The main difference is that a vector in R can only contain elements of the same data type...",
          "sources": [],
          "route_taken": "general_knowledge",
          "suggested_prompts": [
            "1. How do I create a list in R?",
            "2. When should I use a list instead of a data frame?",
            "3. Can you give me an example of a nested list?"
          ]
        }
        ```

---

### 📚 Conceptual Backing & References

The architectural decisions in this project are based on industry-standard best practices for building reliable AI systems. You can validate these concepts with the following resources:

*   **LangChain Documentation on Question-Answering:** A practical guide from a leading AI development framework, explaining how to implement the RAG (Retrieval-Augmented Generation) method.
    *   **Link:** [https://python.langchain.com/docs/use_cases/question_answering/](https://python.langchain.com/docs/use_cases/question_answering/)

*   **"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"**: The original 2020 research paper from Meta AI researchers that introduced the RAG model. It provides the academic foundation for this architecture.
    *   **Link:** [https://arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401)

*   **OpenAI Cookbook: Question Answering using Embeddings:** A hands-on guide from OpenAI on the core technique of using vector embeddings for search and question-answering, which is the heart of the RAG system.
    *   **Link:** [https://cookbook.openai.com/examples/question_answering_using_embeddings](https://cookbook.openai.com/examples/question_answering_using_embeddings)
