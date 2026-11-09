import re

from haystack import Document, component


@component
class ExtractiveReader:

    STOP_WORDS = {
        "what", "is", "are", "the", "a", "an", "and", "or",
        "how", "does", "do", "in", "on", "of", "to", "for",
        "with", "between", "it", "its", "this", "that",
        "be", "as", "by", "from", "was", "were", "related"
    }

    @component.output_types(
        answer=str,
        source=str,
        documents=list[Document]
    )
    def run(self, query: str, documents: list[Document]):

        query_words = {
            word for word in re.findall(r"\b[a-zA-Z0-9]+\b", query.lower())
            if word not in self.STOP_WORDS
        }

        best_sentence = ""
        best_source = "Unknown"
        best_score = 0

        for document in documents:

            # Join lines broken by PDF extraction.
            clean_text = re.sub(r"\s+", " ", document.content).strip()

            sentences = re.split(r"(?<=[.!?])\s+", clean_text)

            for sentence in sentences:
                sentence = sentence.strip()

                sentence_words = {
                    word for word in re.findall(
                        r"\b[a-zA-Z0-9]+\b", sentence.lower()
                    )
                    if word not in self.STOP_WORDS
                }

                if len(sentence_words) < 4:
                    continue

                score = len(query_words.intersection(sentence_words))

                if score > best_score:
                    best_score = score
                    best_sentence = sentence
                    best_source = document.meta.get("source", "Unknown")

        if not best_sentence:
            best_sentence = "No relevant sentence found."

        return {
            "answer": best_sentence,
            "source": best_source,
            "documents": documents
        }
