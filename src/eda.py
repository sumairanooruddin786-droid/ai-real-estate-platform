"""Exploratory data analysis: cleans the data, draws the charts and writes a JSON summary.

Run with:  python -m src.eda
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .config import CHARTS_DIR, EDA_SUMMARY
from .constants import NEIGHBORHOOD_NAMES
from .data_prep import clean_data, load_raw, missing_report, save_clean

PINE, AMBER, SLATE, MIST = "#14574F", "#D9A441", "#17262B", "#D5DDDA"


def _style():
    sns.set_theme(style="whitegrid", rc={
        "axes.edgecolor": MIST, "grid.color": "#E7ECEA", "axes.labelcolor": SLATE,
        "text.color": SLATE, "xtick.color": SLATE, "ytick.color": SLATE,
        "axes.titleweight": "bold", "axes.titlesize": 13, "figure.dpi": 110,
    })


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / name, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def price_distribution(df):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(df["SalePrice"], kde=True, color=PINE, ax=ax[0])
    ax[0].set_title("Sale price is skewed to the right")
    ax[0].set_xlabel("Sale price (USD)")
    sns.histplot(np.log1p(df["SalePrice"]), kde=True, color=AMBER, ax=ax[1])
    ax[1].set_title("After a log transform it is close to normal")
    ax[1].set_xlabel("log(1 + sale price)")
    _save(fig, "price_distribution.png")


def neighborhood_prices(df):
    med = df.groupby("Neighborhood")["SalePrice"].median().sort_values()
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(med.index, med.values, color=PINE)
    ax.axvline(df["SalePrice"].median(), color=AMBER, lw=2, ls="--", label="All homes")
    ax.set_title("Median sale price by neighborhood")
    ax.set_xlabel("Median sale price (USD)")
    ax.legend(loc="lower right")
    _save(fig, "neighborhood_prices.png")


def area_vs_price(df):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sc = ax.scatter(df["GrLivArea"], df["SalePrice"], c=df["OverallQual"], cmap="YlGnBu", s=18, alpha=0.8)
    sns.regplot(x="GrLivArea", y="SalePrice", data=df, scatter=False, color=AMBER, ax=ax)
    ax.set_title("Larger homes cost more, and quality shifts the whole curve")
    ax.set_xlabel("Living area (sq ft)")
    ax.set_ylabel("Sale price (USD)")
    fig.colorbar(sc, ax=ax, label="Overall quality (1-10)")
    _save(fig, "area_vs_price.png")


def quality_boxplot(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x="OverallQual", y="SalePrice", data=df, color=PINE, fliersize=2, ax=ax)
    ax.set_title("Price rises steadily with overall quality")
    ax.set_xlabel("Overall quality (1-10)")
    ax.set_ylabel("Sale price (USD)")
    _save(fig, "quality_boxplot.png")


def correlation_heatmap(df):
    num = df.select_dtypes("number").drop(columns=["Id"], errors="ignore")
    top = num.corr()["SalePrice"].abs().sort_values(ascending=False).head(11).index
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(num[top].corr(), annot=True, fmt=".2f", cmap="YlGnBu", square=True,
                cbar_kws={"shrink": 0.8}, ax=ax, annot_kws={"size": 8})
    ax.set_title("Features most correlated with sale price")
    _save(fig, "correlation_heatmap.png")
    return num.corr()["SalePrice"].drop("SalePrice").abs().sort_values(ascending=False).head(8)


def price_trends(df):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    decade = (df["YearBuilt"] // 10 * 10)
    by_decade = df.groupby(decade)["SalePrice"].median()
    ax[0].plot(by_decade.index, by_decade.values, marker="o", color=PINE, lw=2)
    ax[0].set_title("Median price by decade built")
    ax[0].set_xlabel("Decade built")
    ax[0].set_ylabel("Median sale price (USD)")
    by_year = df.groupby("YrSold")["SalePrice"].median()
    ax[1].bar(by_year.index.astype(str), by_year.values, color=AMBER)
    ax[1].set_title("Median price by year sold")
    ax[1].set_xlabel("Year sold")
    _save(fig, "price_trends.png")
    return by_decade, by_year


def missing_chart(raw):
    miss = missing_report(raw).head(15)
    if miss.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(miss["column"][::-1], miss["percent"][::-1], color=PINE)
    ax.set_title("Missing values in the raw data (top 15 columns)")
    ax.set_xlabel("% of rows missing")
    _save(fig, "missing_values.png")


def run():
    _style()
    raw = load_raw()
    df, log = clean_data(raw)
    save_clean(df)

    price_distribution(df)
    neighborhood_prices(df)
    area_vs_price(df)
    quality_boxplot(df)
    top_corr = correlation_heatmap(df)
    by_decade, by_year = price_trends(df)
    missing_chart(raw)

    nb = df.groupby("Neighborhood").agg(
        homes=("SalePrice", "size"), median_price=("SalePrice", "median"),
        median_ppsf=("PricePerSqft", "median"), median_quality=("OverallQual", "median"),
    ).sort_values("median_price", ascending=False)

    summary = {
        "cleaning": log,
        "homes": int(len(df)),
        "median_price": float(df["SalePrice"].median()),
        "mean_price": float(df["SalePrice"].mean()),
        "median_area": float(df["GrLivArea"].median()),
        "median_ppsf": float(df["PricePerSqft"].median()),
        "price_skew": round(float(df["SalePrice"].skew()), 2),
        "top_correlations": {k: round(float(v), 3) for k, v in top_corr.items()},
        "price_by_decade": {str(int(k)): float(v) for k, v in by_decade.items()},
        "price_by_year_sold": {str(int(k)): float(v) for k, v in by_year.items()},
        "neighborhoods": [
            {"code": code, "name": NEIGHBORHOOD_NAMES.get(code, code), **{k: float(v) for k, v in row.items()}}
            for code, row in nb.iterrows()
        ],
        "charts": [
            {"file": "price_distribution.png", "title": "Price distribution",
             "caption": "Prices are right-skewed, so the model is trained on the log of the price."},
            {"file": "neighborhood_prices.png", "title": "Location comparison",
             "caption": "Neighborhood alone separates budget areas from premium ones."},
            {"file": "area_vs_price.png", "title": "Area versus price",
             "caption": "Living area is the strongest size signal; colour shows quality."},
            {"file": "quality_boxplot.png", "title": "Quality and price",
             "caption": "Each quality step lifts the typical price noticeably."},
            {"file": "correlation_heatmap.png", "title": "Correlations",
             "caption": "Quality, living area and garage size lead the correlations with price."},
            {"file": "price_trends.png", "title": "Property trends",
             "caption": "Median price by decade built and by year sold, to spot age and market-timing effects."},
            {"file": "missing_values.png", "title": "Data quality",
             "caption": "Most gaps mean the feature is absent (no alley, no pool), not unknown."},
        ],
    }
    EDA_SUMMARY.write_text(json.dumps(summary, indent=2))
    print(f"EDA done: {len(df)} homes, {len(summary['charts'])} charts saved to {CHARTS_DIR}")
    return df


if __name__ == "__main__":
    run()
