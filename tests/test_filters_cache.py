from ragsys.cache import LRUCache, make_key, normalise_question
from ragsys.filters import FilterExtractor

COMPANIES = [
    "3M",
    "AMD",
    "Johnson & Johnson",
    "American Water Works",
    "American Express",
    "Block",
    "McDonalds",
]


def test_company_and_alias_extraction():
    fx = FilterExtractor(COMPANIES)
    assert fx.companies_in("What is the FY2018 capital expenditure amount for 3M?") == ["3M"]
    assert fx.companies_in("Is growth in JnJ's adjusted EPS expected to accelerate?") == ["Johnson & Johnson"]
    assert fx.companies_in("Did American Water Works raise its dividend?") == ["American Water Works"]
    assert fx.companies_in("Does McDonald's own its restaurants?") == ["McDonalds"]
    # common word must not trigger the 'Block' filer
    assert fx.companies_in("Which business segment is the largest building block?") == []


def test_year_filter_window():
    fx = FilterExtractor(COMPANIES)
    f = fx.extract("What was AMD's FY2022 gross margin?", use_year=True)
    assert f == {"company": ["AMD"], "doc_period": [2021, 2022, 2023]}


def test_cache_key_is_order_independent_and_lru_evicts():
    assert make_key("a", x=1, y=2) == make_key("a", y=2, x=1)
    assert normalise_question("  What  IS  capex? ") == "what is capex?"
    c = LRUCache(max_items=2)
    c.set("a", 1)
    c.set("b", 2)
    c.get("a")
    c.set("c", 3)  # evicts b (least recently used)
    assert c.get("b") is None and c.get("a") == 1 and c.get("c") == 3
    assert c.hits == 3 and c.misses == 1
