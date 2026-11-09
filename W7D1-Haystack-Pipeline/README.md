# W7D1 – Haystack Pipeline Architecture

## 1. Objective

The objective of this task is to build a document retrieval and extractive question-answering pipeline using Haystack.

The system loads PDF documents, indexes their content, retrieves relevant documents, and extracts a relevant sentence for a user question.

Two retrieval approaches are implemented and compared:
- BM25 lexical retrieval
- Dense retrieval using Sentence Transformers

Both pipelines use the same custom Extractive Reader.

## 2. Technologies Used

- Python 3.11
- Haystack 3.4.1
- PyPDF 6.21.3
- Sentence Transformers 6.2.0
- sentence-transformers-haystack 2.1.0
- all-MiniLM-L6-v2
- ReportLab

## 3. Dataset

Five sample PDF documents were created for this experiment:

1. Artificial Intelligence
2. Natural Language Processing
3. Information Retrieval
4. Cybersecurity
5. Cloud Computing

The PDF files are stored in `data/pdfs/`.

## 4. System Architecture

### PDF Loading

The PDF documents are read using PyPDF. Their extracted text is converted into Haystack Document objects with source metadata.

### BM25 Retrieval Pipeline

1. Load the PDF documents.
2. Create an InMemoryDocumentStore.
3. Index the documents.
4. Use InMemoryBM25Retriever to retrieve the top three relevant documents.
5. Pass the retrieved documents and question to the Extractive Reader.
6. Return the selected answer and its source document.

### Dense Retrieval Pipeline

1. Load the PDF documents.
2. Generate document embeddings using Sentence Transformers.
3. Store the embedded documents in InMemoryDocumentStore.
4. Convert the user question into an embedding.
5. Use InMemoryEmbeddingRetriever with cosine similarity to retrieve the top three documents.
6. Pass the retrieved documents and question to the same Extractive Reader.
7. Return the selected answer and its source document.

The embedding model used is `sentence-transformers/all-MiniLM-L6-v2`.

### Extractive Reader

A custom Haystack component is implemented in `reader.py`.

- Removes common stop words from the question and sentences.
- Compares the meaningful words in the question with each retrieved sentence.
- Selects the sentence with the highest word-overlap score.
- Returns the selected sentence and its source PDF.

This is a lightweight extractive approach. It does not use a generative language model.

## 5. Evaluation

Ten questions were prepared across the five subject areas.

The same questions and expected source documents were used to evaluate both retrieval methods.

### Evaluation Metrics

- Precision@3: Number of relevant retrieved documents divided by the total number of retrieved documents.
- Hit Rate@3: Percentage of questions for which the expected document appears in the top three results.
- Top-1 Accuracy: Percentage of questions for which the expected document is ranked first.

Each question has one expected relevant document in this experiment.

## 6. Results

| Metric | BM25 | Dense Retrieval |
|---|---:|---:|
| Precision@3 | 33.33% | 33.33% |
| Hit Rate@3 | 100% | 100% |
| Top-1 Accuracy | 100% | 100% |

Both methods retrieved the expected document at rank 1 for all ten questions.

Detailed results are available in:
- `results/bm25_results.json`
- `results/dense_results.json`
- `results/comparison_results.json`

## 7. API Compatibility

The practical reference material uses older Haystack APIs. This implementation uses Haystack 3.4.1.

The following modern components and alternatives are used:

- PDFToTextConverter: PDF text extraction using PyPDF.
- FAISSDocumentStore: InMemoryDocumentStore.
- BM25Retriever: InMemoryBM25Retriever.
- DenseRetriever: InMemoryEmbeddingRetriever with Sentence Transformers.
- FARMReader: Replaced with a custom lightweight Extractive Reader because the legacy component is not used in this implementation.

The custom Reader performs keyword-overlap sentence selection. It is not a transformer-based or generative question-answering model.

## 8. Limitations

- The experiment uses only five sample PDF documents.
- Only ten predefined questions were evaluated.
- Each question has one expected relevant document.
- The results may not represent performance on large or diverse document collections.
- The custom Reader uses word overlap and may not understand the full meaning or context of a question.
- The Reader selects an existing sentence rather than generating a new answer.

## 9. Future Improvements

- Add more PDF documents and diverse questions.
- Evaluate retrieval using a larger labelled dataset.
- Improve the Reader using a transformer-based question-answering model.
- Explore hybrid retrieval combining BM25 and dense retrieval.
- Test the pipeline on larger document collections.

## 10. Conclusion

The Haystack implementation successfully loads PDF documents and performs retrieval using both BM25 and Dense Retrieval.

Both pipelines are connected to a custom Extractive Reader that selects a relevant sentence and identifies its source document.

Both retrieval approaches achieved 100% Hit Rate@3 and 100% Top-1 Accuracy on the current ten-question evaluation dataset. The experiment demonstrates how lexical retrieval, embedding-based retrieval, and extractive answer selection can be combined using Haystack.
