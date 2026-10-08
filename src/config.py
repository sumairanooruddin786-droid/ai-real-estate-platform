"""Central place for every path and setting used by the project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RAW_DATA = ROOT / "data" / "train.csv"                      # Kaggle file goes here
CLEAN_DATA = ROOT / "data" / "processed" / "clean.csv"
LISTINGS = ROOT / "data" / "processed" / "listings.csv"      # clean data + model estimate
EDA_SUMMARY = ROOT / "data" / "processed" / "eda_summary.json"

KNOWLEDGE_DIR = ROOT / "data" / "knowledge"                  # starter RAG documents
UPLOAD_DIR = ROOT / "data" / "uploads"                       # documents added from the web UI
VECTOR_DIR = ROOT / "data" / "vectorstore"

MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "price_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
META_PATH = MODEL_DIR / "feature_meta.json"

CHARTS_DIR = ROOT / "app" / "static" / "charts"

for _d in (KNOWLEDGE_DIR, UPLOAD_DIR, VECTOR_DIR, MODEL_DIR, CHARTS_DIR, CLEAN_DATA.parent):
    _d.mkdir(parents=True, exist_ok=True)
