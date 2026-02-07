# syntax=docker/dockerfile:1
FROM python:3.12-slim

# Clean + real-time output (important for CLI)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY main.py .

# Run the CLI app
CMD ["python", "main.py"]
