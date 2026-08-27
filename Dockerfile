# Use python 3.12 as the base image
FROM python:3.12-slim

# Set working directory inside the container
WORKDIR /app

# Install build dependencies
RUN pip install --no-cache-dir setuptools wheel

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Download NLTK data (needed for preprocessing)
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Expose port 8000
EXPOSE 8080

# Run the FastAPI server
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8080"]
