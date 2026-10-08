"""Writes a data-driven neighborhood guide into data/knowledge/ so the RAG assistant can cite it."""
import json

from .config import EDA_SUMMARY, KNOWLEDGE_DIR


def run():
    s = json.loads(EDA_SUMMARY.read_text())
    lines = [
        "# Ames, Iowa Neighborhood Price Guide (generated from the sales data)",
        "",
        f"This guide summarizes {s['homes']:,} historical home sales in Ames, Iowa (2006-2010). "
        f"The median sale price across all homes is ${s['median_price']:,.0f}, the median living area is "
        f"{s['median_area']:,.0f} sq ft and the median price per square foot is ${s['median_ppsf']:,.0f}.",
        "",
    ]
    for n in s["neighborhoods"]:
        lines.append(
            f"## {n['name']} ({n['code']})\n"
            f"{n['name']} has {n['homes']:.0f} sales in the data. The median sale price is ${n['median_price']:,.0f}, "
            f"the median price per square foot is ${n['median_ppsf']:,.0f} and the median overall quality is "
            f"{n['median_quality']:.0f} out of 10.\n"
        )
    (KNOWLEDGE_DIR / "05_area_price_guide.md").write_text("\n".join(lines))
    print("Wrote 05_area_price_guide.md")


if __name__ == "__main__":
    run()
