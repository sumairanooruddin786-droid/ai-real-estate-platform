"""Train, compare and save the price prediction model.

Run with:  python -m src.train_model
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from .config import CHARTS_DIR, LISTINGS, META_PATH, METRICS_PATH, MODEL_PATH
from .constants import ALL_FEATURES, CAT_FEATURES, FEATURE_LABELS, NUM_FEATURES
from .data_prep import load_clean

PINE, AMBER, SLATE = "#14574F", "#D9A441", "#17262B"
RANDOM_STATE = 42


def make_model(regressor, scale: bool) -> TransformedTargetRegressor:
    """Preprocessing + regressor. The target is log-transformed because prices are right-skewed."""
    numeric = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric.append(("scale", StandardScaler()))
    pre = ColumnTransformer([
        ("num", Pipeline(numeric), NUM_FEATURES),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CAT_FEATURES),
    ])
    pipe = Pipeline([("pre", pre), ("model", regressor)])
    return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)


def candidates():
    return {
        "Linear Regression": make_model(LinearRegression(), scale=True),
        "Decision Tree": make_model(DecisionTreeRegressor(max_depth=8, min_samples_leaf=5,
                                                          random_state=RANDOM_STATE), scale=False),
        "Random Forest": make_model(RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1,
                                                          random_state=RANDOM_STATE), scale=False),
        "Gradient Boosting": make_model(GradientBoostingRegressor(n_estimators=400, learning_rate=0.05,
                                                                  max_depth=3, subsample=0.8,
                                                                  random_state=RANDOM_STATE), scale=False),
    }


def evaluate(y_true, y_pred):
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def charts(results, y_test, best_pred, importance, best_name):
    sns.set_theme(style="whitegrid")
    names = [r["name"] for r in results]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    colors = [AMBER if n == best_name else PINE for n in names]
    ax[0].bar(names, [r["r2"] for r in results], color=colors)
    ax[0].set_title("R\u00b2 on unseen homes (higher is better)")
    ax[0].set_ylim(min(0.5, min(r["r2"] for r in results) - 0.05), 1)
    ax[1].bar(names, [r["mae"] for r in results], color=colors)
    ax[1].set_title("Average error in USD (lower is better)")
    for a in ax:
        a.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "model_comparison.png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(y_test, best_pred, s=16, alpha=0.7, color=PINE)
    lim = [min(y_test.min(), best_pred.min()), max(y_test.max(), best_pred.max())]
    ax.plot(lim, lim, color=AMBER, lw=2, label="Perfect prediction")
    ax.set_title(f"{best_name}: predicted vs actual price")
    ax.set_xlabel("Actual price (USD)")
    ax.set_ylabel("Predicted price (USD)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "actual_vs_predicted.png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    top = importance[:10][::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh([t["label"] for t in top], [t["importance"] for t in top], color=PINE)
    ax.set_title("What drives the predicted price")
    ax.set_xlabel("Share of total importance")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "feature_importance.png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run():
    df = load_clean()
    X, y = df[ALL_FEATURES], df["SalePrice"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    results, fitted = [], {}
    for name, model in candidates().items():
        model.fit(X_train, y_train)
        scores = evaluate(y_test, model.predict(X_test))
        cv = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
        results.append({"name": name, **scores, "cv_r2": float(cv.mean()), "cv_r2_std": float(cv.std())})
        fitted[name] = model
        print(f"{name:18s} R2={scores['r2']:.3f}  MAE=${scores['mae']:,.0f}  CV R2={cv.mean():.3f}")

    # Choose by cross-validation on the training split so the test split stays an honest check.
    best_name = max(results, key=lambda r: r["cv_r2"])["name"]
    best = fitted[best_name]
    best_pred = best.predict(X_test)

    # The range shown to users comes from how wrong the model was on homes it had not seen.
    log_resid = np.log(y_test.values / best_pred)
    q10, q90 = float(np.quantile(log_resid, 0.10)), float(np.quantile(log_resid, 0.90))

    perm = permutation_importance(best, X_test, y_test, n_repeats=8, random_state=RANDOM_STATE, scoring="r2")
    raw_imp = np.clip(perm.importances_mean, 0, None)
    share = raw_imp / raw_imp.sum() if raw_imp.sum() > 0 else raw_imp
    importance = sorted(
        ({"feature": f, "label": FEATURE_LABELS[f], "importance": float(s)} for f, s in zip(ALL_FEATURES, share)),
        key=lambda d: d["importance"], reverse=True,
    )
    charts(results, y_test.values, best_pred, importance, best_name)

    # Final model: same recipe, refit on every home so it learns from all the data.
    final = candidates()[best_name]
    final.fit(X, y)
    joblib.dump(final, MODEL_PATH)

    metrics = {
        "best_model": best_name, "models": results, "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "residual_q10": q10, "residual_q90": q90, "target": "SalePrice (USD)",
        "features": ALL_FEATURES, "importance": importance,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    meta = {
        "defaults": {**{c: float(df[c].median()) for c in NUM_FEATURES},
                     **{c: str(df[c].mode().iloc[0]) for c in CAT_FEATURES}},
        "ranges": {c: {"min": float(df[c].min()), "max": float(df[c].max()), "median": float(df[c].median())}
                   for c in NUM_FEATURES},
        "options": {c: sorted(df[c].astype(str).unique().tolist()) for c in CAT_FEATURES},
    }
    META_PATH.write_text(json.dumps(meta, indent=2))

    listings = df.copy()
    listings["EstimatedValue"] = np.round(final.predict(X), 0)
    listings.to_csv(LISTINGS, index=False)

    best_scores = next(r for r in results if r["name"] == best_name)
    print(f"\nBest model: {best_name}  (test R2={best_scores['r2']:.3f}, MAE=${best_scores['mae']:,.0f})")
    print(f"Saved: {MODEL_PATH.name}, metrics.json, feature_meta.json, listings.csv")
    return metrics


if __name__ == "__main__":
    run()
