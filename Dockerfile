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

# Command to run when the container starts
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]