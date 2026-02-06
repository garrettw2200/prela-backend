FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY services/ingest-gateway/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared modules (from repo root)
COPY services/shared /app/shared

# Copy application code
COPY services/ingest-gateway/app /app/app

# Expose port
EXPOSE 8001

# Run with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
