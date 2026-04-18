FROM python:3.11-slim

LABEL org.opencontainers.image.title="waggle-mcp" \
      org.opencontainers.image.description="MCP server that gives LLMs persistent graph-structured memory" \
      org.opencontainers.image.version="0.1.3" \
      org.opencontainers.image.authors="Abhigyan Shekhar" \
      org.opencontainers.image.url="https://github.com/Abhigyan-Shekhar/Waggle-mcp" \
      org.opencontainers.image.source="https://github.com/Abhigyan-Shekhar/Waggle-mcp" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence-transformers \
    WAGGLE_TRANSPORT=stdio \
    WAGGLE_BACKEND=sqlite \
    WAGGLE_DB_PATH=memory.db \
    WAGGLE_HTTP_HOST=0.0.0.0 \
    WAGGLE_HTTP_PORT=8080 \
    WAGGLE_DEFAULT_TENANT_ID=local-default \
    WAGGLE_MODEL=all-MiniLM-L6-v2 \
    WAGGLE_EXTRACT_BACKEND=auto \
    WAGGLE_LOG_LEVEL=INFO

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

# 1) Pre-install CPU-only PyTorch so sentence-transformers never pulls CUDA wheels (~2 GB)
# 2) Then install the project + neo4j extras
RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install ".[neo4j]" && \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Non-root user required by Glama's security policy
RUN useradd --no-create-home --shell /bin/false waggle && \
    mkdir -p /app/data /app/.cache && \
    chown -R waggle:waggle /app

USER waggle

# Only bound when WAGGLE_TRANSPORT=http
EXPOSE 8080

ENTRYPOINT ["python", "-m", "waggle.server"]
CMD ["serve"]
