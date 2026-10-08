"""Content-based property recommender that explains every match."""
import numpy as np
import pandas as pd

from .constants import NEIGHBORHOOD_NAMES


def usd(x) -> str:
    return f"${float(x):,.0f}"


class Recommender:
    def __init__(self, listings: pd.DataFrame):
        self.df = listings.copy()
        self.median_quality = float(self.df["OverallQual"].median())

    def recommend(self, budget: float, neighborhood: str | None = None, min_bedrooms: int = 0,
                  min_bathrooms: float = 0, min_quality: int = 1, min_area: float = 0,
                  garage: bool = False, fireplace: bool = False, top_k: int = 6) -> dict:
        df = self.df
        mask = (df["SalePrice"] <= budget) & (df["BedroomAbvGr"] >= min_bedrooms) \
            & (df["OverallQual"] >= min_quality) & (df["GrLivArea"] >= min_area) \
            & (df["TotalBath"] >= min_bathrooms)
        if neighborhood:
            mask &= df["Neighborhood"] == neighborhood
        if garage:
            mask &= df["GarageCars"] >= 1
        if fireplace:
            mask &= df["Fireplaces"] >= 1
        pool = df[mask].copy()

        if pool.empty:
            return {"total_matches": 0, "results": [], "hint": self._hint(budget, neighborhood, min_bedrooms)}

        # Everything below is scored relative to the homes that passed the filters.
        pool["value_gap"] = (pool["EstimatedValue"] - pool["SalePrice"]) / pool["EstimatedValue"]
        value = (pool["value_gap"].clip(-0.3, 0.3) + 0.3) / 0.6
        budget_fit = 1 - ((pool["SalePrice"] / budget - 0.85).abs() / 0.85).clip(upper=1)
        quality = pool["OverallQual"] / 10
        space = pool["GrLivArea"].rank(pct=True)
        newness = pool["YearBuilt"].rank(pct=True)
        extras = (pool["GarageCars"].clip(upper=3) / 3 + (pool["Fireplaces"] > 0)) / 2
        pool["score"] = 100 * (0.30 * value + 0.15 * budget_fit + 0.25 * quality
                               + 0.15 * space + 0.10 * newness + 0.05 * extras)
        pool["space_pct"] = space
        pool["new_pct"] = newness

        top = pool.sort_values("score", ascending=False).head(top_k)
        results = [self._row(r, budget, neighborhood, min_bedrooms, garage, fireplace) for _, r in top.iterrows()]
        return {"total_matches": int(len(pool)), "results": results, "hint": None}

    # ------------------------------------------------------------------ explanations
    def _row(self, r, budget, neighborhood, min_bedrooms, garage, fireplace) -> dict:
        name = NEIGHBORHOOD_NAMES.get(r["Neighborhood"], r["Neighborhood"])
        reasons = []

        under = (1 - r["SalePrice"] / budget) * 100
        reasons.append(f"{usd(r['SalePrice'])} is {under:.0f}% under your {usd(budget)} budget."
                       if under >= 2 else f"{usd(r['SalePrice'])} fits your {usd(budget)} budget.")
        if neighborhood:
            reasons.append(f"Located in {name}, the area you chose.")
        if min_bedrooms:
            reasons.append(f"{int(r['BedroomAbvGr'])} bedrooms (you asked for at least {int(min_bedrooms)}).")

        gap = r["value_gap"] * 100
        if gap >= 3:
            reasons.append(f"Priced {gap:.0f}% below its estimated market value of {usd(r['EstimatedValue'])}.")
        elif gap <= -3:
            reasons.append(f"Priced about {abs(gap):.0f}% above its estimated value, so there is little room to negotiate.")
        else:
            reasons.append(f"Priced in line with its estimated value of {usd(r['EstimatedValue'])}.")

        if r["OverallQual"] >= self.median_quality + 1:
            reasons.append(f"Build quality {int(r['OverallQual'])}/10, above the typical {self.median_quality:.0f}.")
        elif r["OverallQual"] >= self.median_quality:
            reasons.append(f"Build quality {int(r['OverallQual'])}/10, about typical.")
        if r["space_pct"] >= 0.7:
            reasons.append(f"{int(r['GrLivArea']):,} sq ft of living space, bigger than {r['space_pct'] * 100:.0f}% of matches.")
        if r["new_pct"] >= 0.7:
            reasons.append(f"Built in {int(r['YearBuilt'])}, newer than {r['new_pct'] * 100:.0f}% of matches.")
        extras = []
        if garage and r["GarageCars"] >= 1:
            extras.append(f"a {int(r['GarageCars'])}-car garage")
        if fireplace and r["Fireplaces"] >= 1:
            extras.append("a fireplace")
        if extras:
            reasons.append("Has " + " and ".join(extras) + ", as you requested.")

        return {
            "id": int(r["Id"]),
            "neighborhood": r["Neighborhood"], "neighborhood_name": name,
            "price": int(r["SalePrice"]), "estimate": int(r["EstimatedValue"]),
            "area": int(r["GrLivArea"]), "bedrooms": int(r["BedroomAbvGr"]),
            "bathrooms": float(r["TotalBath"]), "quality": int(r["OverallQual"]),
            "year_built": int(r["YearBuilt"]), "garage": int(r["GarageCars"]),
            "fireplaces": int(r["Fireplaces"]), "lot_area": int(r["LotArea"]),
            "score": round(float(r["score"]), 1), "reasons": reasons[:6],
        }

    def _hint(self, budget, neighborhood, min_bedrooms) -> str:
        """Tell the user which single change would bring results back."""
        df = self.df
        if neighborhood and not df[(df["Neighborhood"] == neighborhood) & (df["SalePrice"] <= budget)].empty:
            return "Nothing matches every filter. Try fewer must-haves or a lower quality minimum."
        cheapest = df[(df["BedroomAbvGr"] >= min_bedrooms)]
        if neighborhood:
            cheapest = cheapest[cheapest["Neighborhood"] == neighborhood]
        if not cheapest.empty:
            return f"Nothing is available at that budget. The lowest price that fits your other filters is {usd(cheapest['SalePrice'].min())}."
        return "Nothing matches these filters. Try fewer bedrooms or a different neighborhood."
