"""Turn a natural-language question into metadata filters.

FinanceBench questions name the company ("...for 3M?", "Is AMD a capital-intensive business...") and
usually the fiscal year ("FY2018"). We match company names/aliases against the known filers and pull
four-digit years. The extracted filter is then evaluated against "no filter" and an oracle filter.
"""

from __future__ import annotations

import re

# hand-written aliases for filers whose question wording differs from the metadata name
ALIASES = {
    "AMD": ["amd", "advanced micro devices"],
    "AES Corporation": ["aes"],
    "Activision Blizzard": ["activision"],
    "American Express": ["american express", "amex"],
    "American Water Works": ["american water"],
    "Best Buy": ["best buy"],
    "Block": ["block inc", "block, inc", "square inc"],
    "McDonalds": ["mcdonald's", "mcdonalds", "mcdonald"],
    "PG&E Corporation": ["pg&e", "pacific gas"],
    "Paypal": ["paypal"],
    "Boeing": ["boeing"],
    "CVS Health": ["cvs"],
    "Coca-Cola": ["coca-cola", "coca cola", "coke"],
    "Corning": ["corning"],
    "Costco": ["costco"],
    "Foot Locker": ["foot locker"],
    "General Mills": ["general mills"],
    "JPMorgan": ["jpmorgan", "jp morgan", "jpm"],
    "Johnson & Johnson": ["johnson & johnson", "johnson and johnson", "j&j", "jnj"],
    "Kraft Heinz": ["kraft heinz", "kraft"],
    "Lockheed Martin": ["lockheed"],
    "MGM Resorts": ["mgm"],
    "Netflix": ["netflix"],
    "Nike": ["nike"],
    "PepsiCo": ["pepsico", "pepsi"],
    "Pfizer": ["pfizer"],
    "Ulta Beauty": ["ulta"],
    "Verizon": ["verizon"],
    "Walmart": ["walmart"],
}

# filer names that are also common English words: only their explicit aliases match
AMBIGUOUS_NAMES = {"Block"}

_YEAR = re.compile(r"(?:FY\s?|fiscal (?:year )?)?((?:19|20)\d{2})\b", re.I)


def _patterns(companies: list[str]) -> list[tuple[re.Pattern, str]]:
    pats = []
    for c in companies:
        names = set(ALIASES.get(c, [])) | ({c.lower()} if c not in AMBIGUOUS_NAMES else set())
        for n in names:
            pats.append((re.compile(r"(?<![a-z])" + re.escape(n) + r"(?![a-z])", re.I), c))
    # longest alias first so "American Water Works" wins over a shorter overlap
    return sorted(pats, key=lambda p: -len(p[0].pattern))


class FilterExtractor:
    def __init__(self, companies: list[str]):
        self.companies = sorted(set(companies))
        self.pats = _patterns(self.companies)

    def companies_in(self, question: str) -> list[str]:
        found = []
        for pat, c in self.pats:
            if c not in found and pat.search(question):
                found.append(c)
        return found

    def years_in(self, question: str) -> list[int]:
        return sorted({int(y) for y in _YEAR.findall(question)})

    def extract(self, question: str, use_year: bool = False) -> dict:
        f: dict = {}
        comps = self.companies_in(question)
        if comps:
            f["company"] = comps
        if use_year:
            ys = self.years_in(question)
            if ys:
                # comparatives live in the next filing and outlook statements in the previous one
                f["doc_period"] = sorted({y + d for y in ys for d in (-1, 0, 1)})
        return f
