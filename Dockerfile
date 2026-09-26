FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/model.onnx

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY models ./models

EXPOSE 8000

# Single worker is enough for a demo; raise workers behind a reverse proxy in prod
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
