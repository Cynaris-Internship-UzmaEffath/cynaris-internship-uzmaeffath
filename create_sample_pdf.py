from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

output = "W5D3-chromadb/sample.pdf"

c = canvas.Canvas(output, pagesize=A4)
text = c.beginText(50, 790)
text.setFont("Helvetica", 11)

content = [
    "Introduction to AI and Cybersecurity",
    "",
    "Machine learning helps computers learn patterns from data.",
    "Deep learning uses neural networks to learn complex patterns.",
    "An embedding represents text as a numerical vector.",
    "",
    "ChromaDB is a vector database used to store embeddings and",
    "retrieve documents based on semantic similarity.",
    "Retrieval-Augmented Generation, or RAG, retrieves relevant",
    "document chunks before passing context to a language model.",
    "",
    "Cybersecurity protects systems, networks, and data from attacks.",
    "Intrusion detection identifies suspicious network activity.",
    "Firewalls filter network traffic according to security rules.",
    "Encryption protects information from unauthorized access.",
    "Multi-factor authentication adds another layer of login security.",
    "",
    "Ollama can run language models and embedding models locally.",
    "A RAG system can use ChromaDB with Ollama to answer questions",
    "using information retrieved from a PDF.",
]

for line in content:
    text.textLine(line)

c.drawText(text)
c.save()

print(f"Created {output}")
