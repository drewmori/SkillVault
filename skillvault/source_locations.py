"""Resolve verified quotation positions into document-friendly locations."""

from pathlib import Path
import re


def quote_location(filename: str, text: str, quote: str) -> str:
    position = text.find(quote)
    if position < 0 or not quote:
        raise ValueError("Quote is absent from the source")
    prefix = text[:position]
    line = prefix.count("\n") + 1
    markers = list(re.finditer(r"^(SLIDE \d+|PAGE \d+|PARAGRAPH \d+|TABLE \d+ ROW \d+):", prefix, re.MULTILINE))
    if markers:
        marker = markers[-1]
        return f"{marker.group(1).capitalize()} · extracted line {line}"
    suffix = Path(filename).suffix.lower()
    transformed = {".pdf", ".docx", ".pptx", ".json", ".ipynb", ".eml"}
    return f"{'Extracted text line' if suffix in transformed else 'Line'} {line}"
