# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose MLflow + FastAPI ports (future use)
EXPOSE 5000
EXPOSE 8000

# Default command (we can override later)
CMD ["bash"]