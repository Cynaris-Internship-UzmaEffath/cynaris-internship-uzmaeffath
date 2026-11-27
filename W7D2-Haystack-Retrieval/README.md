# W7D2 - Haystack Retrieval: BM25 and Dense Retrieval

## 1. Objective

The objective of this task is to build and compare two document retrieval approaches using Haystack:

- BM25: keyword-based document retrieval.
- Dense Retrieval: semantic document retrieval using sentence embeddings.

Both approaches are evaluated using the same five PDF documents and ten questions.

## 2. Technologies Used

- Python 3.11
- Haystack 3.4.1
- PyPDFToDocument
- InMemoryDocumentStore
- InMemoryBM25Retriever
- InMemoryEmbeddingRetriever
- Sentence Transformers
- all-MiniLM-L6-v2
- Cosine similarity
- VS Code
- Git and GitHub

## 3. Project Structure

```text
W7D2-Haystack-Retrieval/
|
|-- data/
|   |-- pdfs/
|       |-- 01_artificial_intelligence.pdf
|       |-- 02_natural_language_processing.pdf
|       |-- 03_information_retrieval.pdf
|       |-- 04_cybersecurity.pdf
|       |-- 05_cloud_computing.pdf
|
|-- results/
|   |-- bm25_results.json
|   |-- dense_results.json
|   |-- comparison_results.json
|
|-- evidence/
|
|-- load_documents.py
|-- bm25_pipeline.py
|-- dense_pipeline.py
|-- evaluation_questions.py
|-- evaluate_bm25.py
|-- evaluate_dense.py
|-- compare_retrieval.py
|-- README.md
```

## 4. Implementation

### Document Loading

The `load_documents.py` script uses Haystack's `PyPDFToDocument` converter to extract content from five PDF files and convert them into Haystack documents. Source filenames and file types are stored as metadata.

### BM25 Retrieval

The `bm25_pipeline.py` script uses:

- `InMemoryDocumentStore` to store documents.
- `InMemoryBM25Retriever` to retrieve documents based on keyword relevance.
- Top 3 documents for each query.

### Dense Retrieval

The `dense_pipeline.py` script uses:

- `SentenceTransformersDocumentEmbedder` to generate document embeddings.
- `SentenceTransformersTextEmbedder` to generate query embeddings.
- `InMemoryEmbeddingRetriever` to retrieve semantically similar documents.
- `all-MiniLM-L6-v2` as the embedding model.
- Cosine similarity to compare embeddings.

## 5. Evaluation

Ten questions were prepared across five topics:

1. Artificial Intelligence
2. Natural Language Processing
3. Information Retrieval
4. Cybersecurity
5. Cloud Computing

Both retrieval methods use the same questions and return the top 3 documents for each query.

Evaluation metrics:

- Precision@3: Proportion of retrieved documents matching the expected source.
- Hit Rate@3: Proportion of questions for which the expected source appears in the top 3.
- Top-1 Accuracy: Proportion of questions for which the expected source is ranked first.

## 6. Results

| Evaluation Metric |   BM25 | Dense Retrieval |
| ----------------- | -----: | --------------: |
| Precision@3       | 0.3333 |          0.3333 |
| Hit Rate@3        | 1.0000 |          1.0000 |
| Top-1 Accuracy    | 1.0000 |          1.0000 |

### Observations

- Both methods retrieved the expected PDF in the top 3 results for all ten questions.
- Both methods ranked the expected PDF first for all ten questions.
- Both methods achieved the same evaluation scores.
- The remaining retrieved documents differed between BM25 and Dense Retrieval.

## 7. Execution Instructions

Run these commands from the repository root after activating the Python virtual environment.

Load and verify PDF documents:

```powershell
python .\W7D2-Haystack-Retrieval\load_documents.py
```

Run BM25 retrieval:

```powershell
python .\W7D2-Haystack-Retrieval\bm25_pipeline.py
```

Run Dense Retrieval:

```powershell
python .\W7D2-Haystack-Retrieval\dense_pipeline.py
```

Evaluate BM25:

```powershell
python .\W7D2-Haystack-Retrieval\evaluate_bm25.py
```

Evaluate Dense Retrieval:

```powershell
python .\W7D2-Haystack-Retrieval\evaluate_dense.py
```

Compare both methods:

```powershell
python .\W7D2-Haystack-Retrieval\compare_retrieval.py
```

## 8. Limitations

- The evaluation uses only five educational PDF documents.
- Each question has one expected relevant source document.
- The evaluation measures document retrieval, not the correctness or quality of generated answers.
- Documents are stored in an in-memory document store, so they must be indexed again when a script runs.
- Results may differ with larger datasets, more complex questions, and different document content.

## 9. Conclusion

This task implemented and evaluated keyword-based BM25 retrieval and embedding-based Dense Retrieval using Haystack.

Both approaches successfully retrieved the expected source document for all ten evaluation questions. They achieved 100% Hit Rate@3 and 100% Top-1 Accuracy on this dataset.

The task provided practical experience with PDF document conversion, document indexing, keyword retrieval, semantic retrieval, embeddings, and retrieval evaluation.
