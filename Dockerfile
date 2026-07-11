# Detective AI — production image
# Core pipeline + RAG embeddings. Add the "cv" extra for real video ingestion.
FROM python:3.12-slim AS builder

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir --prefix=/install ".[embeddings]"

FROM python:3.12-slim

RUN useradd --create-home --shell /usr/sbin/nologin detective
WORKDIR /app

COPY --from=builder /install /usr/local
COPY src ./src
COPY benchmarks ./benchmarks
COPY frontend ./frontend

RUN mkdir -p /app/data && chown detective:detective /app/data

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000

USER detective
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/api/health').raise_for_status()"

CMD ["uvicorn", "detective_ai.api.app:app", "--host", "0.0.0.0", "--port", "8000"]