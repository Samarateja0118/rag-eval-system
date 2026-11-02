"""Lexical normalisation shared by BM25 indexing and querying."""

from __future__ import annotations

import re

import Stemmer  # PyStemmer (Snowball)

_TOKEN = re.compile(r"[a-z0-9]+(?:[.][0-9]+)?")
_NUM_COMMA = re.compile(r"(?<=\d),(?=\d{3})")
_STEM = Stemmer.Stemmer("english")

STOPWORDS = frozenset(
    """a an and are as at be by for from has have in is it its of on or that the this to was were will with
    what which who whom how did does do than then there their they them we our you your i me my he she
    his her not no can could would should may might into over under about all any each per""".split()
)


def tokenize(text: str) -> list[str]:
    text = _NUM_COMMA.sub("", text.lower())
    toks = [t for t in _TOKEN.findall(text) if t not in STOPWORDS]
    return _STEM.stemWords(toks)
