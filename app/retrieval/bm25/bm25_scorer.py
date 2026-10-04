import math

from app.retrieval.bm25.inverted_index import InvertedIndex


class BM25Scorer:
    """Calculates how relevant a document is for a query, using the index statistics.

    k1: how fast repeated words stop adding score (0 = a word counts once, higher = repeats keep helping).
    b:  how much long documents are penalised (0 = no penalty, 1 = full penalty).
    """

    def __init__(
        self,
        index: InvertedIndex,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> None:
        # Written as "not ... >= 0" (and not "k1 < 0") so that NaN is rejected too.
        if not k1 >= 0:
            raise ValueError(f"k1 must be >= 0, got {k1}")

        if not 0 <= b <= 1:
            raise ValueError(f"b must be between 0 and 1, got {b}")

        self._index = index
        self._k1 = k1
        self._b = b

    def score(self, query: str, document_id: str) -> float:
        """BM25 score of one document for the query (0.0 if no query word is in it)."""
        terms = self._index.tokenize(query)
        return self._score_terms(terms, document_id)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Return up to top_k (document_id, score) pairs, best match first.

        A document is a candidate if it contains at least one query word.
        """
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        terms = self._index.tokenize(query)

        candidates = set()
        for term in terms:
            candidates.update(self._index.documents_containing(term))

        # sorted() makes equal scores come out in a fixed (document id) order.
        results = []
        for document_id in sorted(candidates):
            results.append((document_id, self._score_terms(terms, document_id)))

        results.sort(key=lambda pair: pair[1], reverse=True)
        return results[:top_k]

    def _score_terms(self, terms: list[str], document_id: str) -> float:
        # Called first so an unknown document_id fails loudly instead of scoring 0.
        document_length = self._index.document_length(document_id)

        total = 0.0
        for term in terms:
            total += self._term_score(term, document_id, document_length)

        return total

    def _term_score(self, term: str, document_id: str, document_length: int) -> float:
        tf = self._index.term_frequency(term, document_id)
        if tf == 0:
            return 0.0

        # How much a long document is penalised: 1 means "average length".
        length_ratio = document_length / self._index.average_document_length
        length_norm = 1 - self._b + self._b * length_ratio

        numerator = tf * (self._k1 + 1)
        denominator = tf + self._k1 * length_norm

        return self._idf(term) * numerator / denominator

    def _idf(self, term: str) -> float:
        """Rare words score high, words that are in almost every document score low."""
        n = self._index.document_count
        df = self._index.document_frequency(term)

        return math.log((n - df + 0.5) / (df + 0.5) + 1)
