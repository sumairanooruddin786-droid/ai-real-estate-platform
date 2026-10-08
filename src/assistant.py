"""Generative AI layer: prompt design, market-data context, RAG answers and property summaries.

The LLM is Groq (Llama 3.3) when GROQ_API_KEY is set. Without a key, the assistant falls back to
showing the retrieved facts directly so the app is still usable.
"""
import os
import re

import pandas as pd
from dotenv import load_dotenv

from .constants import NEIGHBORHOOD_NAMES
from .rag import RAGService

load_dotenv()

# ----------------------------------------------------------------------------- prompts
SYSTEM_PROMPT = """You are Plotwise, a real estate assistant for home buyers and investors.

Scope and facts
- Your market data comes from historical home sales in Ames, Iowa (2006-2010). It is a learning dataset, not a live market. Say so when someone asks about current prices or about other cities.
- Use only the CONTEXT blocks below for facts and numbers. If the context does not cover the question, say what is missing instead of guessing.
- When you use a document, cite it with its source number, like [1] or [2].

Style
- Give the answer first, then the reasoning. Stay under 180 words unless the user asks for more.
- Write prices in USD with thousands separators. Use a short bullet list for comparisons or steps.
- You give guidance, not legal, tax or financial advice. For contracts or large decisions, add one short sentence suggesting a qualified professional."""

DOCS_ONLY_RULES = """
Documents-only mode
- Answer strictly from the DOCUMENTS block. Do not use outside knowledge or the market data.
- If the documents do not contain the answer, reply exactly: "I couldn't find that in the documents." and suggest what kind of document would help."""

# Two short worked examples teach the format (few-shot prompting).
FEW_SHOT = [
    {"role": "user", "content": "CONTEXT\nMARKET DATA: Overall median price $160,000. College Creek (CollgCr): 150 homes, "
                                "median price $200,000, median quality 7.\n\nQUESTION: Is College Creek expensive?"},
    {"role": "assistant", "content": "Yes, it is above the market. College Creek's median price is $200,000, about 25% higher "
                                     "than the overall median of $160,000, and its typical build quality is 7 out of 10.\n"
                                     "- Expect to pay a premium for newer, higher-quality homes.\n"
                                     "- Remember this is historical Ames data, not current listings."},
    {"role": "user", "content": "CONTEXT\nDOCUMENTS:\n[1] (guide.md, p.1) Closing costs commonly run 2% to 5% of the price.\n\n"
                                "QUESTION: How much should I save for closing costs on a $150,000 home?"},
    {"role": "assistant", "content": "Plan for roughly $3,000 to $7,500, which is 2% to 5% of $150,000 [1]. "
                                     "Ask your lender for a loan estimate to get the exact figure."},
]

SUMMARY_PROMPT = """Write a property summary in exactly three sentences:
1. What the home is (size, bedrooms, neighborhood, age).
2. How its price compares with the estimated value and its neighborhood.
3. One specific thing a buyer should check before making an offer.
Use only the facts below. Do not invent features.

FACTS
{facts}"""


def _best_sentences(text: str, query: str, n: int = 2) -> str:
    """Pick the sentences that share the most words with the question (used when no LLM is available)."""
    words = {w for w in re.findall(r"[a-z]{3,}", query.lower())}
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\s+#+\s+", text) if len(s.strip()) > 25]
    ranked = sorted(sentences, key=lambda s: -len(words & set(re.findall(r"[a-z]{3,}", s.lower()))))
    return " ".join(ranked[:n]) or text[:300]


def _usd(x) -> str:
    return f"${float(x):,.0f}"


class LLM:
    def __init__(self):
        self.model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.client = None
        key = os.getenv("GROQ_API_KEY", "").strip()
        if key:
            try:
                from groq import Groq
                self.client = Groq(api_key=key)
            except Exception as exc:
                print(f"[assistant] Groq client unavailable: {exc}")

    @property
    def available(self) -> bool:
        return self.client is not None

    def chat(self, messages: list[dict], temperature: float = 0.2, max_tokens: int = 500) -> str | None:
        if not self.client:
            return None
        try:
            res = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens)
            return res.choices[0].message.content.strip()
        except Exception as exc:
            print(f"[assistant] LLM call failed: {exc}")
            return None


class Assistant:
    def __init__(self, listings: pd.DataFrame, rag: RAGService):
        self.df = listings
        self.rag = rag
        self.llm = LLM()
        self._aliases = {}
        for code, name in NEIGHBORHOOD_NAMES.items():
            self._aliases[code.lower()] = code
            self._aliases[name.lower()] = code

    # ------------------------------------------------------------------ market context
    def _find_neighborhoods(self, question: str) -> list[str]:
        q = question.lower()
        found = []
        for alias, code in sorted(self._aliases.items(), key=lambda kv: -len(kv[0])):
            if re.search(rf"\b{re.escape(alias)}\b", q) and code not in found:
                found.append(code)
        return found[:3]

    def market_snapshot(self, question: str) -> str:
        df = self.df
        lines = [
            f"Overall: {len(df):,} homes, median price {_usd(df['SalePrice'].median())}, "
            f"median living area {df['GrLivArea'].median():,.0f} sq ft, median price per sq ft {_usd(df['PricePerSqft'].median())}."
        ]
        stats = df.groupby("Neighborhood").agg(n=("SalePrice", "size"), price=("SalePrice", "median"),
                                               ppsf=("PricePerSqft", "median"), qual=("OverallQual", "median"),
                                               year=("YearBuilt", "median"))
        for code in self._find_neighborhoods(question):
            r = stats.loc[code]
            lines.append(f"{NEIGHBORHOOD_NAMES.get(code, code)} ({code}): {r['n']:.0f} homes, median price {_usd(r['price'])}, "
                         f"{_usd(r['ppsf'])} per sq ft, median quality {r['qual']:.0f}/10, typical year built {r['year']:.0f}.")
        ranked = stats.sort_values("price", ascending=False)
        top = ", ".join(f"{NEIGHBORHOOD_NAMES.get(c, c)} {_usd(v)}" for c, v in ranked["price"].head(3).items())
        low = ", ".join(f"{NEIGHBORHOOD_NAMES.get(c, c)} {_usd(v)}" for c, v in ranked["price"].tail(3).items())
        lines.append(f"Highest median prices: {top}. Lowest: {low}.")
        by_year = df.groupby("YrSold")["SalePrice"].median()
        lines.append("Median price by year sold: " + ", ".join(f"{int(y)} {_usd(v)}" for y, v in by_year.items()) + ".")
        return "\n".join(lines)

    # ------------------------------------------------------------------ chat
    def answer(self, question: str, mode: str = "advisor", history: list[dict] | None = None) -> dict:
        question = question.strip()
        docs_only = mode == "docs"
        hits = self.rag.search(question, k=4)
        sources = [{"n": i, "source": h["source"], "page": h["page"], "score": round(h["score"], 2),
                    "snippet": h["text"][:240] + ("..." if len(h["text"]) > 240 else "")}
                   for i, h in enumerate(hits, start=1)]

        if docs_only and not hits:
            return {"answer": "I couldn't find that in the documents. Try rephrasing the question, or upload a document that covers it.",
                    "sources": [], "used_llm": False, "mode": mode}

        doc_block = "\n".join(f"[{s['n']}] ({s['source']}, p.{s['page']}) {h['text']}" for s, h in zip(sources, hits))
        context = []
        if hits:
            context.append("DOCUMENTS:\n" + doc_block)
        if not docs_only:
            context.append("MARKET DATA:\n" + self.market_snapshot(question))
        user_msg = "CONTEXT\n" + "\n\n".join(context) + f"\n\nQUESTION: {question}"

        system = SYSTEM_PROMPT + (DOCS_ONLY_RULES if docs_only else "")
        messages = [{"role": "system", "content": system}]
        if not docs_only:
            messages += FEW_SHOT
        for turn in (history or [])[-6:]:
            if turn.get("role") in ("user", "assistant") and turn.get("content"):
                messages.append({"role": turn["role"], "content": str(turn["content"])[:1500]})
        messages.append({"role": "user", "content": user_msg})

        text = self.llm.chat(messages)
        if text:
            return {"answer": text, "sources": sources if hits else [], "used_llm": True, "mode": mode}
        return {"answer": self._offline_answer(question, hits, docs_only), "sources": sources,
                "used_llm": False, "mode": mode}

    def _offline_answer(self, question, hits, docs_only) -> str:
        note = "No GROQ_API_KEY is set, so I am showing the most relevant information instead of a written answer.\n\n"
        parts = []
        if not docs_only:
            parts.append("**Market data**\n" + self.market_snapshot(question))
        if hits:
            parts.append("**From the documents**\n" + "\n".join(f"- [{i}] {_best_sentences(h['text'], question)}" for i, h in enumerate(hits, 1)))
        return note + "\n\n".join(parts)

    # ------------------------------------------------------------------ property summary
    def summarize_property(self, prop_id: int) -> dict:
        row = self.df[self.df["Id"] == prop_id]
        if row.empty:
            raise KeyError(prop_id)
        r = row.iloc[0]
        name = NEIGHBORHOOD_NAMES.get(r["Neighborhood"], r["Neighborhood"])
        hood_med = self.df.loc[self.df["Neighborhood"] == r["Neighborhood"], "SalePrice"].median()
        gap = (r["EstimatedValue"] - r["SalePrice"]) / r["EstimatedValue"] * 100
        facts = (f"Neighborhood: {name}. Price: {_usd(r['SalePrice'])}. Estimated value: {_usd(r['EstimatedValue'])} "
                 f"({gap:+.0f}% gap). Neighborhood median: {_usd(hood_med)}. Living area: {r['GrLivArea']:,.0f} sq ft. "
                 f"Bedrooms: {int(r['BedroomAbvGr'])}. Bathrooms: {r['TotalBath']:.1f}. Overall quality: {int(r['OverallQual'])}/10. "
                 f"Year built: {int(r['YearBuilt'])}. Remodeled: {'yes' if r['Remodeled'] else 'no'}. "
                 f"Garage cars: {int(r['GarageCars'])}. Fireplaces: {int(r['Fireplaces'])}. Lot: {r['LotArea']:,.0f} sq ft.")
        text = self.llm.chat([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": SUMMARY_PROMPT.format(facts=facts)},
        ], temperature=0.3, max_tokens=220)
        if text:
            return {"summary": text, "used_llm": True}
        position = "below" if gap >= 3 else "above" if gap <= -3 else "in line with"
        text = (f"This is a {int(r['BedroomAbvGr'])}-bedroom, {r['GrLivArea']:,.0f} sq ft home in {name}, built in {int(r['YearBuilt'])}. "
                f"At {_usd(r['SalePrice'])} it is priced {position} its estimated value of {_usd(r['EstimatedValue'])} "
                f"and the {name} median of {_usd(hood_med)}. Before making an offer, have the roof, foundation and heating system inspected.")
        return {"summary": text, "used_llm": False}
