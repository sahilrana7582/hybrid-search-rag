import re


class Tokenizer:
    _TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[._:+-][A-Za-z0-9]+)*")

    def tokenize(self, text: str) -> list[str]:
        if not text:
            return []
 
        text = text.lower()

        return self._TOKEN_PATTERN.findall(text)