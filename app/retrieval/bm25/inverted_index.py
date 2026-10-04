from app.retrieval.tokenizer.base import BaseTokenizer


class InvertedIndex:
    def __init__(self, tokenizer: BaseTokenizer) -> None:
        self._tokenizer = tokenizer
        self._term_doc_counts: dict[str, dict[str, int]] = {}
        self._doc_lengths: dict[str, int] = {}

    def add_document(self, document_id: str, text: str) -> None:
        if document_id in self._doc_lengths:
            raise ValueError(f"Document already indexed: {document_id}")

        tokens = self._tokenizer.tokenize(text)
        self._doc_lengths[document_id] = len(tokens)

        for token in tokens:
            if token not in self._term_doc_counts:
                self._term_doc_counts[token] = {}

            docs = self._term_doc_counts[token]
            docs[document_id] = docs.get(document_id, 0) + 1

    def search(self, query: str) -> list[str]:
        """Return ids of documents that contain ALL the words in the query."""
        terms = self._tokenizer.tokenize(query)
        if not terms:
            return []

        results = set(self._term_doc_counts.get(terms[0], {}))
        for term in terms[1:]:
            results = results & set(self._term_doc_counts.get(term, {}))

        return sorted(results)
