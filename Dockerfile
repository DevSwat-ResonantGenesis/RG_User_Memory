FROM python:3.11-slim

WORKDIR /app

# Install curl for health checks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY frontend/ ./frontend/

# Create data directory
RUN mkdir -p /data/user_memories

ENV PYTHONPATH=/app
ENV USER_MEMORY_DATA_DIR=/data/user_memories

EXPOSE 8094

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8094"]
