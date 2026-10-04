from app.retrieval.tokenizer.base import BaseTokenizer


class InvertedIndex:
    def __init__(self, tokenizer: BaseTokenizer) -> None:
        self._tokenizer = tokenizer
        self._term_doc_counts: dict[str, dict[str, int]] = {}
        self._doc_lengths: dict[str, int] = {}
        self._total_length = 0

    def add_document(self, document_id: str, text: str) -> None:
        if document_id in self._doc_lengths:
            raise ValueError(f"Document already indexed: {document_id}")

        tokens = self._tokenizer.tokenize(text)
        self._doc_lengths[document_id] = len(tokens)
        self._total_length += len(tokens)

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

    @property
    def document_count(self) -> int:
        """N: how many documents are indexed."""
        return len(self._doc_lengths)

    @property
    def average_document_length(self) -> float:
        """AVGDL: average number of tokens per document (0.0 for an empty index)."""
        if not self._doc_lengths:
            return 0.0

        return self._total_length / len(self._doc_lengths)

    def tokenize(self, text: str) -> list[str]:
        """Split text with the same tokenizer the documents were indexed with."""
        return self._tokenizer.tokenize(text)

    def term_frequency(self, term: str, document_id: str) -> int:
        """TF: how many times `term` appears in the document (0 if it does not)."""
        return self._term_doc_counts.get(term, {}).get(document_id, 0)

    def document_frequency(self, term: str) -> int:
        """DF: how many documents contain `term`."""
        return len(self._term_doc_counts.get(term, {}))

    def document_length(self, document_id: str) -> int:
        """DL: number of tokens in the document."""
        if document_id not in self._doc_lengths:
            raise ValueError(f"Unknown document: {document_id}")

        return self._doc_lengths[document_id]

    def documents_containing(self, term: str) -> list[str]:
        """Ids of the documents that contain `term`."""
        return list(self._term_doc_counts.get(term, {}))
