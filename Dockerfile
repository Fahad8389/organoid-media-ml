FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 10000

# Run the application - use PORT env variable from Render
CMD uvicorn web.app:app --host 0.0.0.0 --port ${PORT:-10000}
