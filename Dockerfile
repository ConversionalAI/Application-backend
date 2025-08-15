# Use lightweight Python base image
FROM python:3.11-slim

# Set working directory in container
WORKDIR /app

# Install system dependencies (for psycopg2, etc.)
RUN apt-get update && apt-get install -y gcc libpq-dev && rm -rf /var/lib/apt/lists/*

# Copy dependencies file first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Run the FastAPI app (Cloud Run requires port 8080)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
