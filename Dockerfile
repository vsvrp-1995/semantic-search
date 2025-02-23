# Use an official lightweight Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install any system dependencies (faiss may require build tools)
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port (Coolify will pass the port via an env variable, default to 5000)
EXPOSE 3000

# Start the app using gunicorn
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-3000} app:app"]
