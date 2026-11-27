W8D3 - Deployment: Haystack API on Railway

## 1. Overview

This project implements a document retrieval API using **Haystack, FastAPI, and Docker**.

It provides two retrieval approaches:

- **BM25 Retrieval:** Keyword-based document retrieval using term frequency and inverse document frequency.
- **Dense Retrieval:** Semantic document retrieval using Sentence Transformers embeddings.

The API retrieves relevant content from five PDF documents created during W7D2.

## 2. Project Structure

```text
W8D3-Haystack-API/
├── src/
│   └── app.py
├── requirements.txt
├── Dockerfile
└── README.md

W7D2-Haystack-Retrieval/
├── load_documents.py
├── bm25_pipeline.py
├── dense_pipeline.py
└── data/
    └── pdfs/
        ├── 01_artificial_intelligence.pdf
        ├── 02_natural_language_processing.pdf
        ├── 03_information_retrieval.pdf
        ├── 04_cybersecurity.pdf
        └── 05_cloud_computing.pdf
```

## 3. Technologies Used

- Python 3.11
- FastAPI
- Haystack 3
- Sentence Transformers
- PyPDF
- Uvicorn
- Docker
- PowerShell

## 4. API Endpoints

The application exposes the following endpoints:

| HTTP Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Checks API health and available retrievers |
| GET | `/search` | Retrieves the top three relevant documents |
| GET | `/answer` | Returns retrieved context and source filenames |

### Query Parameters

- `q`: User's search question.
- `method`: Retrieval method (`bm25` or `dense`).

Example:

```text
/search?q=What%20is%20machine%20learning?&method=bm25
```

**Note:** The `/answer` endpoint returns retrieved document context. It does not generate an LLM-based answer.

## 5. Local Setup

Install the required dependencies:

```powershell
pip install -r .\W8D3-Haystack-API\requirements.txt
```

Start the FastAPI application from the repository root:

```powershell
uvicorn app:app --app-dir W8D3-Haystack-API/src --reload
```

Open the Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

## 6. Docker Deployment

### Step 1: Build the Docker Image

Run from the repository root:

```powershell
docker build -f .\W8D3-Haystack-API\Dockerfile -t haystack-retrieval-api:latest .
```

### Step 2: Run the Container

```powershell
docker run -d --name haystack-api-test -p 8001:8000 haystack-retrieval-api:latest
```

### Step 3: Verify Container Status

```powershell
docker ps
```

### Step 4: Check Container Logs

```powershell
docker logs haystack-api-test
```

### Step 5: Access the API

```text
http://127.0.0.1:8001/
```

Swagger documentation:

```text
http://127.0.0.1:8001/docs
```

## 7. API Testing

### Health Check

```powershell
Invoke-RestMethod "http://127.0.0.1:8001/"
```

Expected response:

```json
{
    "status": "healthy",
    "service": "Haystack Retrieval API",
    "retrievers": [
        "bm25",
        "dense"
    ]
}
```

### BM25 Retrieval

```powershell
$result = Invoke-RestMethod "http://127.0.0.1:8001/search?q=What%20is%20machine%20learning%3F&method=bm25"

$result | ConvertTo-Json -Depth 5
```

### Dense Retrieval

```powershell
$result = Invoke-RestMethod "http://127.0.0.1:8001/search?q=What%20is%20machine%20learning%3F&method=dense"

$result | ConvertTo-Json -Depth 5
```

### Retrieved Context

```powershell
$result = Invoke-RestMethod "http://127.0.0.1:8001/answer?q=What%20is%20machine%20learning%3F&method=dense"

$result | ConvertTo-Json -Depth 5
```

## 8. Docker Validation Results

The following tests were successfully completed:

- Docker image built successfully.
- Docker container started and remained running.
- All five PDF documents loaded successfully.
- BM25 retrieval pipeline initialized successfully.
- Dense retrieval pipeline initialized successfully.
- Sentence Transformers embedding model loaded successfully.
- Health check returned `healthy`.
- BM25 search returned the top three relevant documents.
- Dense search returned the top three relevant documents.
- Answer endpoint returned combined retrieved context and source filenames.
- Swagger UI displayed all three API endpoints.

### Sample Retrieval Comparison

For the query:

`What is machine learning?`

Both BM25 and Dense retrieval returned these documents in the same order:

1. `01_artificial_intelligence.pdf`
2. `02_natural_language_processing.pdf`
3. `03_information_retrieval.pdf`

This confirms that both retrieval methods operate successfully in the Docker environment for the tested query.

## 9. Deployment Configuration

The Docker application:

- Uses Python 3.11 Slim as its base image.
- Runs the FastAPI application using Uvicorn.
- Listens on `0.0.0.0`.
- Uses the `PORT` environment variable for cloud deployment.
- Defaults to port `8000` when `PORT` is not provided.

The Docker configuration supports deployment on cloud platforms such as Railway.

## 10. Railway Cloud Deployment

The application was successfully deployed to Railway using Docker.

**Live API URL:**

https://cynaris-internship-uzmaeffath-production-34001c40.up.railway.app

**Deployment Configuration:**

- Platform: Railway
- Builder: Dockerfile
- Dockerfile Path: `W8D3-Haystack-API/Dockerfile`
- Git Branch: `feat/aiml-W8-UzmaEffath`
- Application Port: 8000
- Deployment Status: Successful

### Production API Testing

The following endpoints were tested successfully on the live Railway deployment:

| Endpoint | Test Result |
|---|---|
| `/` | Healthy |
| `/search?method=bm25` | Successful |
| `/search?method=dense` | Successful |
| `/answer?method=dense` | Successful |

Both BM25 and Dense retrieval returned three ranked results for the query:

`What is machine learning?`

The Answer endpoint returned combined retrieval context from the relevant PDF documents along with their source filenames.

**Production Swagger Documentation:**

https://cynaris-internship-uzmaeffath-production-34001c40.up.railway.app/docs

## 11. Conclusion

Successfully developed and containerized a Haystack-based PDF retrieval API using FastAPI and Docker.

The application supports both keyword-based BM25 retrieval and semantic Dense retrieval. Local Docker testing confirmed that the API endpoints, document loading, embedding model initialization, and retrieval operations work successfully.

---

**Project:** W8D3 – Deployment: Haystack API on Railway

**Track:** AI/ML Internship

**Repository:** Cynaris Internship

## 12. Automated API Tests

Automated API tests are available in:

`W8D3-Haystack-API/tests/test_api.py`

The test suite uses **pytest**, **FastAPI TestClient**, and mocked retrieval pipelines.

### Test Coverage

| Test | Description |
|---|---|
| Health check | Verifies API health response |
| BM25 search | Verifies keyword retrieval |
| Dense search | Verifies semantic retrieval |
| Invalid method | Checks unsupported retrieval methods |
| Short query | Checks minimum query length validation |
| Answer endpoint | Verifies combined context and sources |
| Empty results | Checks behaviour when no documents are retrieved |

### Run Tests

From the repository root, execute:

```powershell
python -m pytest .\W8D3-Haystack-API\tests\test_api.py -v
```
