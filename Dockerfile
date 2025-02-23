# Use an official lightweight Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install any system dependencies (FAISS may require build tools)
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Ensure the documents directory exists for persistent storage
RUN mkdir -p /app/documents

# Expose the port (Coolify passes the port via an env variable, default 3000)
EXPOSE 3000

# Start the app: process existing PDFs, then run Gunicorn
CMD ["sh", "-c", "python process_existing.py && gunicorn --bind 0.0.0.0:${PORT:-3000} app:app"]
