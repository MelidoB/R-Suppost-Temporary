# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for PDF processing
RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY rag_chatbot.py .

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable for the OpenAI API key
# The actual key should be passed in at runtime, e.g., using `docker run -e OPENAI_API_KEY="your_key"`
ENV OPENAI_API_KEY=""

# Run the application when the container launches
CMD ["uvicorn", "rag_chatbot:app", "--host", "0.0.0.0", "--port", "8000"]