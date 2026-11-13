from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

output = Path("data/pdfs")
output.mkdir(parents=True, exist_ok=True)

documents = {
    "01_artificial_intelligence.pdf": {
        "title": "Artificial Intelligence",
        "paragraphs": [
            "Artificial Intelligence (AI) is a branch of computer science that develops systems capable of performing tasks that normally require human intelligence.",
            "Machine learning is a subset of AI. It enables computers to learn patterns from data and improve their performance without being explicitly programmed for every task.",
            "Deep learning is a subset of machine learning that uses neural networks with multiple layers. It is commonly used in image recognition, speech processing, and natural language processing.",
            "Common applications of AI include recommendation systems, virtual assistants, fraud detection, medical image analysis, and autonomous vehicles."
        ]
    },
    "02_natural_language_processing.pdf": {
        "title": "Natural Language Processing",
        "paragraphs": [
            "Natural Language Processing (NLP) is a field of artificial intelligence that enables computers to understand, interpret, and generate human language.",
            "Tokenization divides text into smaller units called tokens. These units may be words, subwords, or characters.",
            "Stemming reduces words to their root form by removing prefixes or suffixes. Lemmatization converts words into their meaningful dictionary base forms.",
            "Applications of NLP include sentiment analysis, machine translation, text summarization, chatbots, and information extraction.",
            "Transformer models use attention mechanisms to understand relationships between words in a sequence."
        ]
    },
    "03_information_retrieval.pdf": {
        "title": "Information Retrieval",
        "paragraphs": [
            "Information Retrieval (IR) is the process of finding relevant information from a collection of documents in response to a user's query.",
            "BM25 is a keyword-based ranking algorithm. It considers term frequency, inverse document frequency, and document length when ranking documents.",
            "Dense retrieval converts queries and documents into numerical vectors called embeddings. It retrieves documents based on semantic similarity.",
            "Precision measures the proportion of retrieved documents that are relevant. Recall measures the proportion of all relevant documents that were retrieved.",
            "A vector database stores and searches vector embeddings. It is commonly used in semantic search and retrieval-augmented generation systems."
        ]
    },
    "04_cybersecurity.pdf": {
        "title": "Cybersecurity",
        "paragraphs": [
            "Cybersecurity protects computer systems, networks, applications, and data from unauthorized access, attacks, and damage.",
            "The CIA triad represents three important security principles: confidentiality, integrity, and availability.",
            "Confidentiality ensures that information is accessible only to authorized users. Integrity protects information from unauthorized modification. Availability ensures that systems and data remain accessible when needed.",
            "Encryption converts readable information into an encoded form using a cryptographic key. Authentication verifies the identity of a user or system.",
            "Common cybersecurity threats include phishing, malware, ransomware, denial-of-service attacks, and password attacks."
        ]
    },
    "05_cloud_computing.pdf": {
        "title": "Cloud Computing",
        "paragraphs": [
            "Cloud computing provides computing resources such as servers, storage, databases, and software over the internet.",
            "Infrastructure as a Service (IaaS) provides virtualized computing resources. Platform as a Service (PaaS) provides an environment for application development and deployment. Software as a Service (SaaS) provides ready-to-use software over the internet.",
            "Public cloud services are provided over shared infrastructure. Private clouds are dedicated to a single organization. Hybrid cloud combines public and private cloud environments.",
            "Scalability allows computing resources to increase or decrease based on demand. Elasticity automatically adjusts resources according to changing workloads.",
            "Cloud computing can reduce infrastructure management effort and support flexible resource usage."
        ]
    }
}

styles = getSampleStyleSheet()

for filename, content in documents.items():
    path = output / filename
    story = [Paragraph(content["title"], styles["Title"]), Spacer(1, 16)]

    for paragraph in content["paragraphs"]:
        story.extend([Paragraph(paragraph, styles["BodyText"]), Spacer(1, 12)])

    SimpleDocTemplate(str(path), pagesize=A4).build(story)
    print(f"Created: {path}")

print(f"\nTotal PDFs created: {len(documents)}")
