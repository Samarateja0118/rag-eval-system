from ragsys.ingest.chunking import CONFIGS, chunk_document, chunk_sentences, chunk_tokens


def count(s: str) -> int:  # 1 token per whitespace word
    return len(s.split())


PAGES = ["alpha beta gamma delta. Epsilon zeta eta theta.", "iota kappa lambda mu. Nu xi omicron pi rho."]


def test_token_chunks_respect_size_and_cover_all_words():
    ch = chunk_tokens(PAGES, size=4, overlap=1, count=count)
    assert all(count(c.text) <= 4 for c in ch)
    words = " ".join(PAGES).split()
    assert words[0] in ch[0].text.split() and words[-1] in ch[-1].text.split()
    # overlap: last word of chunk i reappears as first word of chunk i+1
    assert ch[0].text.split()[-1] == ch[1].text.split()[0]


def test_chunks_track_page_spans():
    ch = chunk_tokens(PAGES, size=6, overlap=0, count=count)
    spanning = [c for c in ch if c.page_start != c.page_end]
    assert spanning, "a window crossing the page break should span two pages"
    assert all(c.page_start <= c.page_end for c in ch)
    assert ch[0].page_start == 0 and ch[-1].page_end == 1


def test_sentence_chunks_do_not_split_sentences():
    ch = chunk_sentences(PAGES, size=8, overlap=0, count=count)
    for c in ch:
        assert c.text.endswith(".")


def test_page_strategy_one_chunk_per_nonempty_page():
    ch = chunk_document(PAGES + [""], CONFIGS["page"], count)
    assert [c.page_start for c in ch] == [0, 1]


def test_overlap_must_be_smaller_than_size():
    import pytest

    with pytest.raises(ValueError):
        chunk_tokens(PAGES, size=4, overlap=4, count=count)
