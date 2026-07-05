# LensWord - Dockerfile
# Base image with Python 3.11
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements file first (for faster rebuilds)
COPY requirements.txt .

# Install all Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Move into the src folder where api.py lives
WORKDIR /app/src

# Expose port 8000 for the API
EXPOSE 8000

# F11 fix: run as non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# F11 fix: HEALTHCHECK against the health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

# Command to run when the container starts
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"