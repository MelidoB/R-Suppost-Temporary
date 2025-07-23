=========================
✅ Steps to Use the RAG Chatbot
=========================

1. Download the `requirements.txt`
   - Make sure this file is saved in your project directory.

2. Install Python dependencies
   Run this command in your terminal:
   > pip install -r requirements.txt

3. Install required system dependencies
   These are needed to process PDFs and perform OCR (Optical Character Recognition):
   > sudo apt-get install poppler-utils tesseract-ocr

4. Set your OpenAI API key
   Replace "your_api_key" with your actual key:
   > export OPENAI_API_KEY="your_api_key"

5. Fix rate limit errors (if any)
   - If you see a "RateLimitError", you might need to check your OpenAI plan and upgrade it if needed.
   - Visit: https://platform.openai.com/docs/guides/error-codes/api-errors

6. Run the application
   Start the FastAPI server using Uvicorn:
   > uvicorn rag_chatbot:app --reload

-------------------------
📌 Notes
-------------------------
- The `requirements.txt` file uses pinned versions to avoid issues with deprecated imports or incompatible packages.
- If you're using Windows or macOS, you may need different steps for installing system packages. Let me know if you need help with that.
