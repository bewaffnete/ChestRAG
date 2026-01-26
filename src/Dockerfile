FROM ollama/ollama:latest AS ollama

ENTRYPOINT ["/bin/sh", "-c"]
CMD ["ollama serve & sleep 8 && ollama pull $LLM && wait"]

FROM python:3.11-slim AS api

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]