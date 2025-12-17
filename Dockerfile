FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY backend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy application files
COPY backend/ ./backend/

# Create models directory
RUN mkdir -p models

# Expose port (use PORT env variable in production)
EXPOSE 8000

# Set environment variable
ENV PYTHONUNBUFFERED=1

# Run application
# Models will download on first request
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

