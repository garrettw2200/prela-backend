FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY services/api-gateway/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared modules (from repo root)
COPY services/shared /app/shared

# Copy application code
COPY services/api-gateway/app /app/app

# Expose port
EXPOSE 8000

# Run with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
