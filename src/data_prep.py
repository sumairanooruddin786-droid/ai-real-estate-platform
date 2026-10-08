"""Loading, cleaning and feature engineering for the Ames housing data (Kaggle train.csv)."""
import pandas as pd

from .config import CLEAN_DATA, RAW_DATA

# In this dataset a missing value in these columns means "the house does not have this feature",
# not "we do not know". They are filled with the text "None" instead of being treated as unknown.
NONE_MEANS_ABSENT = [
    "Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "PoolQC", "Fence", "MiscFeature", "MasVnrType",
]


def load_raw(path=RAW_DATA) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Download train.csv from the Kaggle 'House Prices - Advanced Regression Techniques' "
            "competition and place it in the data/ folder."
        )
    return pd.read_csv(path)


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Columns that contain missing values, with counts and percentages."""
    miss = df.isna().sum()
    out = pd.DataFrame({"column": miss.index, "missing": miss.values})
    out["percent"] = (out["missing"] / len(df) * 100).round(1)
    return out[out["missing"] > 0].sort_values("missing", ascending=False).reset_index(drop=True)


def clean_data(raw: pd.DataFrame):
    """Return (clean_dataframe, log). The log is a plain dict used in the report and dashboard."""
    df = raw.copy()
    log = {"rows_raw": int(len(df)), "columns_raw": int(df.shape[1]),
           "missing_cells_raw": int(df.isna().sum().sum())}

    if "Id" not in df.columns:
        df.insert(0, "Id", range(1, len(df) + 1))

    before = len(df)
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "Id"])
    log["duplicates_removed"] = int(before - len(df))

    # Two very large houses sold far below the trend; the dataset authors recommend removing them.
    outliers = (df["GrLivArea"] > 4000) & (df["SalePrice"] < 300000)
    log["outliers_removed"] = int(outliers.sum())
    df = df.loc[~outliers].copy()

    for col in NONE_MEANS_ABSENT:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    if "LotFrontage" in df.columns:
        by_area = df.groupby("Neighborhood")["LotFrontage"].transform("median")
        df["LotFrontage"] = df["LotFrontage"].fillna(by_area).fillna(df["LotFrontage"].median())
    if "GarageYrBlt" in df.columns:
        df["GarageYrBlt"] = df["GarageYrBlt"].fillna(df["YearBuilt"])
    if "MasVnrArea" in df.columns:
        df["MasVnrArea"] = df["MasVnrArea"].fillna(0)

    # Whatever is left: median for numbers, most common value for text
    for col in df.columns:
        if df[col].isna().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode().iloc[0])
    log["missing_cells_clean"] = int(df.isna().sum().sum())

    # Engineered features used in the analysis
    df["HouseAge"] = (df["YrSold"] - df["YearBuilt"]).clip(lower=0)
    df["TotalSF"] = df[["TotalBsmtSF", "1stFlrSF", "2ndFlrSF"]].sum(axis=1)
    df["TotalBath"] = (df["FullBath"] + 0.5 * df["HalfBath"]
                       + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"])
    df["PricePerSqft"] = (df["SalePrice"] / df["GrLivArea"]).round(2)
    df["Remodeled"] = (df["YearRemodAdd"] != df["YearBuilt"]).astype(int)

    log["rows_clean"] = int(len(df))
    log["columns_clean"] = int(df.shape[1])
    return df.reset_index(drop=True), log


def save_clean(df: pd.DataFrame) -> None:
    CLEAN_DATA.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_DATA, index=False)


def load_clean() -> pd.DataFrame:
    return pd.read_csv(CLEAN_DATA)
