FROM python:3.10-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code and model files
COPY . .

# HF Spaces uses port 7860
EXPOSE 7860

# gunicorn: 1 worker (model is large), 120s timeout for inference
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "120", "app:app"]
