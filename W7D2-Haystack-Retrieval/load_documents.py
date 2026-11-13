from pathlib import Path
from haystack.components.converters.pypdf import PyPDFToDocument


PDF_DIR = Path("W7D2-Haystack-Retrieval/data/pdfs")


def load_documents():
    converter = PyPDFToDocument()

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {PDF_DIR}")

    documents = []

    for pdf_file in pdf_files:
        result = converter.run(sources=[pdf_file])
        converted_documents = result["documents"]

        for document in converted_documents:
            document.meta["source"] = pdf_file.name
            document.meta["file_type"] = "PDF"

        documents.extend(converted_documents)
        print(f"Loaded: {pdf_file.name}")

    print(f"\nTotal PDFs loaded: {len(pdf_files)}")
    print(f"Total documents created: {len(documents)}")

    return documents


if __name__ == "__main__":
    load_documents()
