"""Lookup tables shared by the data pipeline, the model and the web app."""

# Neighborhood codes used in the Kaggle (Ames, Iowa) dataset -> readable names
NEIGHBORHOOD_NAMES = {
    "Blmngtn": "Bloomington Heights", "Blueste": "Bluestem", "BrDale": "Briardale",
    "BrkSide": "Brookside", "ClearCr": "Clear Creek", "CollgCr": "College Creek",
    "Crawfor": "Crawford", "Edwards": "Edwards", "Gilbert": "Gilbert",
    "IDOTRR": "Iowa DOT and Rail Road", "MeadowV": "Meadow Village", "Mitchel": "Mitchell",
    "NAmes": "North Ames", "NoRidge": "Northridge", "NPkVill": "Northpark Villa",
    "NridgHt": "Northridge Heights", "NWAmes": "Northwest Ames", "OldTown": "Old Town",
    "SWISU": "South & West of Iowa State University", "Sawyer": "Sawyer",
    "SawyerW": "Sawyer West", "Somerst": "Somerset", "StoneBr": "Stone Brook",
    "Timber": "Timberland", "Veenker": "Veenker",
}

KITCHEN_LABELS = {"Ex": "Excellent", "Gd": "Good", "TA": "Typical", "Fa": "Fair", "Po": "Poor"}
KITCHEN_ORDER = ["Po", "Fa", "TA", "Gd", "Ex"]

BLDG_LABELS = {
    "1Fam": "Single-family", "2fmCon": "Two-family conversion",
    "Duplex": "Duplex", "Duplx": "Duplex",
    "TwnhsE": "Townhouse (end unit)", "Twnhs": "Townhouse (inside unit)", "TwnhsI": "Townhouse (inside unit)",
}

# Features the price model uses. These are the ones a person can realistically type into a form;
# every other column of the dataset is not needed by the app.
NUM_FEATURES = [
    "OverallQual", "GrLivArea", "TotalBsmtSF", "LotArea", "BedroomAbvGr", "FullBath",
    "TotRmsAbvGrd", "GarageCars", "Fireplaces", "YearBuilt", "YearRemodAdd",
]
CAT_FEATURES = ["Neighborhood", "KitchenQual", "BldgType"]
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES

FEATURE_LABELS = {
    "OverallQual": "Overall quality", "GrLivArea": "Living area", "TotalBsmtSF": "Basement area",
    "LotArea": "Lot size", "BedroomAbvGr": "Bedrooms", "FullBath": "Full bathrooms",
    "TotRmsAbvGrd": "Rooms above ground", "GarageCars": "Garage capacity", "Fireplaces": "Fireplaces",
    "YearBuilt": "Year built", "YearRemodAdd": "Year remodeled", "Neighborhood": "Neighborhood",
    "KitchenQual": "Kitchen quality", "BldgType": "Building type",
}
