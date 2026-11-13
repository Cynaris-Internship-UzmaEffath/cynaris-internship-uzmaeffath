from pathlib import Path
from pypdf import PdfReader
from haystack import Document


def load_pdf_documents(pdf_directory="data/pdfs"):
    documents = []

    pdf_files = sorted(Path(pdf_directory).glob("*.pdf"))

    for pdf_path in pdf_files:
        reader = PdfReader(str(pdf_path))
        text = "\n".join(
            page.extract_text() or "" for page in reader.pages
        ).strip()

        if not text:
            print(f"Skipped empty PDF: {pdf_path.name}")
            continue

        document = Document(
            content=text,
            meta={
                "source": pdf_path.name,
                "file_type": "PDF"
            }
        )

        documents.append(document)
        print(f"Loaded: {pdf_path.name}")

    print(f"\nTotal documents loaded: {len(documents)}")
    return documents


if __name__ == "__main__":
    docs = load_pdf_documents()

    for doc in docs:
        print(f"{doc.meta['source']} | {len(doc.content)} characters")
