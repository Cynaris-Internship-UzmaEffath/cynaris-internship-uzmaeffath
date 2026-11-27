FROM python:3.11-slim

WORKDIR /app

COPY W8D3-Haystack-API/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY W7D2-Haystack-Retrieval/ ./W7D2-Haystack-Retrieval/
COPY W8D3-Haystack-API/src/ ./W8D3-Haystack-API/src/

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app:app --app-dir W8D3-Haystack-API/src --host 0.0.0.0 --port ${PORT:-8000}"]
