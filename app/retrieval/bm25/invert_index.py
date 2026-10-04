from app.retrieval.tokenizer.demo_tokenizer import Tokenizer


class InvertedIndex:
    def __init__(self, tokenizer: Tokenizer) -> None:
        self._keyword_doc_map: dict[str, list[str]] = {}
        self._tokenizer = tokenizer

    def add_document(self, document_id: str, text: str) -> None:
        tokens = self._tokenizer.tokenize(text)

        for token in tokens:
            if token not in self._keyword_doc_map:
                self._keyword_doc_map[token] = []

            if document_id not in self._keyword_doc_map[token]:
                self._keyword_doc_map[token].append(document_id)

    def search(self, term: str) -> list[str]:
        return self._keyword_doc_map.get(term.lower(), [])