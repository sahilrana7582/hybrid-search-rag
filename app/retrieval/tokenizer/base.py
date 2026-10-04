from abc import ABC, abstractmethod


class BaseTokenizer(ABC):
    """Turns text into a list of tokens.

    Every tokenizer (our regex one, NLTK, spaCy, a HuggingFace tokenizer...)
    must follow this one contract, so the InvertedIndex never cares which
    one it is using.
    """

    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        """Return the tokens of `text`. Must return [] for empty text."""
