FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/app/models \
    LOG_FILE=/app/logs/service.log

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY models ./models
COPY logs ./logs

RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install --no-deps .

EXPOSE 8000

CMD ["uvicorn", "minds14_intent_service.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
