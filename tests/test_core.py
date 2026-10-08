"""Fast unit tests that need no dataset:  python -m pytest -q"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import rag  # noqa: E402
from src.recommender import Recommender  # noqa: E402


def make_listings():
    return pd.DataFrame({
        "Id": [1, 2, 3], "Neighborhood": ["NAmes", "NAmes", "CollgCr"],
        "SalePrice": [150000, 190000, 260000], "EstimatedValue": [170000, 185000, 255000],
        "BedroomAbvGr": [3, 4, 4], "TotalBath": [1.5, 2.0, 2.5], "OverallQual": [5, 7, 8],
        "GrLivArea": [1200, 1800, 2300], "YearBuilt": [1960, 1995, 2005], "GarageCars": [1, 2, 3],
        "Fireplaces": [0, 1, 1], "LotArea": [8000, 9000, 10000],
    })


def test_chunking_keeps_source_and_page():
    text = " ".join(f"word{i}" for i in range(400))
    chunks = rag.chunk_text(text, "guide.md", 3, size=100, overlap=20)
    assert len(chunks) >= 4
    assert all(c.source == "guide.md" and c.page == 3 for c in chunks)
    assert len({c.id for c in chunks}) == len(chunks)


def test_unsupported_file_type_is_rejected(tmp_path):
    bad = tmp_path / "notes.docx"
    bad.write_text("hello")
    with pytest.raises(ValueError):
        rag.read_document(bad)


def test_tfidf_search_finds_relevant_chunk(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "VECTOR_DIR", tmp_path)
    store = rag.TfidfStore()
    store.upsert([
        rag.Chunk("a::p1::0", "closing costs are two to five percent of the price", "a.md", 1),
        rag.Chunk("b::p1::0", "garage and basement add usable space to a home", "b.md", 1),
    ])
    hit = store.search("how much are closing costs", k=1)[0]
    assert hit["source"] == "a.md"


def test_recommender_respects_filters_and_explains():
    rec = Recommender(make_listings())
    out = rec.recommend(budget=200000, min_bedrooms=3, top_k=5)
    assert out["total_matches"] == 2
    assert all(r["price"] <= 200000 for r in out["results"])
    assert all(len(r["reasons"]) >= 3 for r in out["results"])


def test_recommender_gives_hint_when_nothing_matches():
    out = Recommender(make_listings()).recommend(budget=50000)
    assert out["results"] == [] and out["hint"]
