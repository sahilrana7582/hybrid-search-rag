import re

from app.retrieval.tokenizer.base import BaseTokenizer


class RegexTokenizer(BaseTokenizer):
    # \w+ matches letters (any language), digits and "_".
    # The optional group keeps things like gpt-4, v1.2.3 and 10:30 as one token.
    _TOKEN_PATTERN = re.compile(r"\w+(?:[.:+-]\w+)*")

    def tokenize(self, text: str) -> list[str]:
        return self._TOKEN_PATTERN.findall(text.lower())
