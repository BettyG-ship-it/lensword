# LensWord Dockerfile
# Runs the FastAPI sentiment analysis API with RAG + Groq LLM + SQLite

FROM python:3.11-slim

# Install system dependencies needed for LIME and other packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for faster rebuilds
COPY requirements.txt .

# Install all Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Create the src directory with correct permissions
# SQLite database will be created here at runtime
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app

USER appuser

WORKDIR /app/src

EXPOSE 8000

# Pass GROQ_API_KEY at runtime using:
# docker run -e GROQ_API_KEY=your-key-here -p 8000:8000 lensword
# Never hardcode secrets in the Dockerfile

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]