FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
# llama-cpp-python builds llama.cpp from source
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY ragsys ./ragsys
COPY scripts ./scripts
RUN pip install ".[llm]"
RUN useradd -m -u 1000 rag && mkdir -p /app/data /app/models /app/results && chown -R rag /app
USER rag
ENV RAG_DATA_DIR=/app/data RAG_MODEL_DIR=/app/models
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s CMD curl -fs localhost:8000/healthz || exit 1
CMD ["uvicorn", "ragsys.api:app", "--host", "0.0.0.0", "--port", "8000"]
