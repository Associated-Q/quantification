# 构建上下文是项目根目录：docker build -f web/Dockerfile -t ta-kb .
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    HF_ENDPOINT=https://hf-mirror.com

WORKDIR /app
COPY web/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 预下载嵌入模型，避免首次请求时等待
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

COPY web /app/web
COPY _distill/rag/children.jsonl _distill/rag/parents.jsonl _distill/rag/embeddings.npy /app/_distill/rag/
COPY _distill/out /app/_distill/out

WORKDIR /app/web
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
