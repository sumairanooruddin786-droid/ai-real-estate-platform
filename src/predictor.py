"""Loads the saved model and turns a handful of form fields into a price estimate."""
import json

import joblib
import numpy as np
import pandas as pd

from .config import LISTINGS, META_PATH, METRICS_PATH, MODEL_PATH
from .constants import (ALL_FEATURES, CAT_FEATURES, KITCHEN_LABELS, KITCHEN_ORDER,
                        NEIGHBORHOOD_NAMES, NUM_FEATURES)


class Predictor:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        self.meta = json.loads(META_PATH.read_text())
        self.metrics = json.loads(METRICS_PATH.read_text())
        self.listings = pd.read_csv(LISTINGS)
        self.neigh_median = self.listings.groupby("Neighborhood")["SalePrice"].median().to_dict()

    # ------------------------------------------------------------------ helpers
    def _frame(self, values: dict) -> pd.DataFrame:
        row = dict(self.meta["defaults"])
        row.update({k: v for k, v in values.items() if v is not None and k in ALL_FEATURES})
        return pd.DataFrame([row], columns=ALL_FEATURES)

    def _price(self, values: dict) -> float:
        return float(self.model.predict(self._frame(values))[0])

    def _what_if(self, values: dict, base: float) -> list:
        """Re-price the home with one change at a time to show what would move the estimate."""
        full = {**self.meta["defaults"], **{k: v for k, v in values.items() if v is not None}}
        rng = self.meta["ranges"]
        scenarios = []

        if full["OverallQual"] < 10:
            scenarios.append(("One step higher overall quality", {"OverallQual": full["OverallQual"] + 1}))
        scenarios.append(("100 sq ft more living area", {"GrLivArea": full["GrLivArea"] + 100}))
        if full["GarageCars"] < rng["GarageCars"]["max"]:
            scenarios.append(("Room for one more car in the garage", {"GarageCars": full["GarageCars"] + 1}))
        if full["Fireplaces"] < 2:
            scenarios.append(("Add a fireplace", {"Fireplaces": full["Fireplaces"] + 1}))
        if full["KitchenQual"] in KITCHEN_ORDER and full["KitchenQual"] != "Ex":
            nxt = KITCHEN_ORDER[KITCHEN_ORDER.index(full["KitchenQual"]) + 1]
            scenarios.append((f"Upgrade the kitchen to {KITCHEN_LABELS[nxt].lower()}", {"KitchenQual": nxt}))
        if full["YearRemodAdd"] < 2005:
            scenarios.append(("Remodel the home", {"YearRemodAdd": 2010}))

        out = []
        for label, change in scenarios:
            delta = self._price({**full, **change}) - base
            out.append({"change": label, "delta": round(delta)})
        return sorted(out, key=lambda d: d["delta"], reverse=True)

    # ------------------------------------------------------------------ public
    def predict(self, values: dict) -> dict:
        price = self._price(values)
        low = price * float(np.exp(self.metrics["residual_q10"]))
        high = price * float(np.exp(self.metrics["residual_q90"]))
        low, high = sorted((low, high))
        hood = values.get("Neighborhood") or self.meta["defaults"]["Neighborhood"]
        hood_med = self.neigh_median.get(hood)
        return {
            "price": round(price),
            "low": round(low),
            "high": round(high),
            "neighborhood": hood,
            "neighborhood_name": NEIGHBORHOOD_NAMES.get(hood, hood),
            "neighborhood_median": round(hood_med) if hood_med else None,
            "vs_neighborhood_pct": round((price / hood_med - 1) * 100, 1) if hood_med else None,
            "model": self.metrics["best_model"],
            "what_if": self._what_if(values, price),
        }
