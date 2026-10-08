# Plotwise: AI Real Estate Intelligence Platform

A full-stack AI product that analyzes homes, predicts prices, recommends properties with explanations, and answers questions with a Generative AI assistant backed by RAG. Built on the Kaggle **House Prices: Advanced Regression Techniques** dataset (1,460 home sales in Ames, Iowa, 2006-2010).

| Page | What it does |
|---|---|
| **Market overview** | Data cleaning summary, 7 EDA charts, model comparison table and charts |
| **Price estimate** | Live price prediction with a likely range, neighborhood comparison and "what would change this price" |
| **Find homes** | Recommendations by budget, location, bedrooms and preferences, each with reasons and an AI summary |
| **Assistant** | Market advisor chat and a documents-only RAG chat with cited sources, plus document upload |

## How the project meets the assignment

| Requirement | Where |
|---|---|
| Data cleaning and analysis (Pandas, NumPy, Matplotlib, Seaborn) | `src/data_prep.py`, `src/eda.py` |
| ML price prediction: train, evaluate, save | `src/train_model.py` compares Linear Regression, Decision Tree, Random Forest and Gradient Boosting, then saves the best to `models/` |
| Recommendation system that explains why | `src/recommender.py` |
| Generative AI assistant with prompt engineering | `src/assistant.py` (system prompt, few-shot examples, grounded context, property summary prompt) |
| RAG over property documents | `src/rag.py` with starter documents in `data/knowledge/` and upload from the web UI |
| FastAPI + web interface (prediction, recommendation, chatbot pages) | `app/main.py`, `app/templates/`, `app/static/` |
| GitHub submission | this repository |

## Quick start

You need Python 3.10 or newer.

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 2. Install packages
pip install -r requirements.txt

# 3. Add the dataset
#    Download train.csv from
#    https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data
#    and save it as  data/train.csv

# 4. (Optional but recommended) Add your Groq key for written AI answers
copy .env.example .env         # then paste GROQ_API_KEY inside .env

# 5. Build everything the app needs: cleaning, charts, model, knowledge base
python run_pipeline.py

# 6. Start the app
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000. Interactive API docs are at http://127.0.0.1:8000/docs.

If you skip the Groq key the app still works: the assistant shows the retrieved facts instead of a written answer. If `chromadb` cannot be installed, RAG automatically falls back to TF-IDF keyword search (shown on the Assistant page).

## Project structure

```
run_pipeline.py          one command: EDA -> train -> knowledge guide -> index documents
src/
  config.py              all paths
  constants.py           feature lists, neighborhood names
  data_prep.py           loading, cleaning, feature engineering
  eda.py                 charts and eda_summary.json
  train_model.py         model comparison, evaluation, saving
  predictor.py           price estimate, range and what-if scenarios
  recommender.py         filtering, scoring and explanations
  rag.py                 loaders, chunking, vector store (ChromaDB or TF-IDF)
  assistant.py           prompts, market context, chat, summaries
  build_knowledge.py     generates a neighborhood guide from the data
app/
  main.py                FastAPI routes
  templates/, static/    pages, CSS, JavaScript, generated charts
data/
  train.csv              Kaggle dataset (you add this)
  knowledge/             starter documents for RAG
models/                  saved model, metrics.json, feature_meta.json
tests/                   unit tests
```

## How it works

**Data.** Missing values in fields such as Alley, Pool and Fireplace quality mean the feature does not exist, so they become "None" rather than being guessed. Two famous outliers (very large homes sold cheaply) are removed. Remaining gaps use the median or most common value. Five features are engineered: house age, total square feet, total bathrooms, price per square foot and a remodeled flag.

**Model.** Prices are right-skewed, so models train on `log(1 + price)`. The inputs are 14 features a person can actually enter in a form (quality, living area, basement, lot, bedrooms, bathrooms, rooms, garage, fireplaces, year built, year remodeled, neighborhood, kitchen quality, building type). Four models are compared on a held-out 20% test set and with 5-fold cross-validation. The winner is chosen by cross-validation, then refit on all data and saved. The "likely range" shown to users is the 10th to 90th percentile of the model's error on unseen homes.

**Recommender.** Hard filters (budget, bedrooms, bathrooms, quality, area, garage, fireplace, neighborhood) are applied first. Remaining homes are scored from 0 to 100 on value versus the model's estimate (30%), build quality (25%), budget fit (15%), space (15%), newness (10%) and extras (5%). Each result lists the specific reasons that apply to it.

**Assistant.** The system prompt defines scope (historical Ames data, not a live market), grounding (use only the provided context), citation style and length. Two worked examples teach the answer format. Each question is sent with retrieved document passages and a live market snapshot computed from the data. Documents-only mode refuses to answer when nothing relevant is retrieved.

**RAG.** Documents (PDF, TXT, MD) are split into overlapping 160-word chunks with source name and page number, embedded, and stored in ChromaDB. Answers cite sources as [1], [2] and the UI shows each source with its passage.

## API

| Method | Route | Purpose |
|---|---|---|
| POST | `/api/predict` | Price estimate, range, what-if changes |
| POST | `/api/recommend` | Recommended homes with reasons |
| GET | `/api/property/{id}/summary` | AI summary of one home |
| POST | `/api/chat` | `{message, mode: "advisor" or "docs", history}` |
| GET / POST / DELETE | `/api/documents` | List, upload and remove knowledge documents |
| GET | `/health` | Setup status |

## Results

After running `python run_pipeline.py`, the console prints a table with R², average error and cross-validated R² for each model. The same numbers appear on the Market overview page and in `models/metrics.json`. Paste your results here:

| Model | R² (test) | Average error | CV R² |
|---|---|---|---|
| Linear Regression | | | |
| Decision Tree | | | |
| Random Forest | | | |
| Gradient Boosting | | | |

## Screenshots

Add screenshots of the four pages to a `screenshots/` folder and link them here.

## Tests

```bash
pip install pytest
python -m pytest -q
```

## Limitations

- The data is historical (2006-2010) and from one city, so estimates are for learning and are not appraisals. The assistant says so when asked about current prices or other cities.
- "Properties" in the recommender are past sales treated as listings, and the estimate for each is made by a model that saw that home during training, so the value gap is slightly optimistic.
- Legal and investment documents are general educational material, not advice.

## Deploying a live demo

The app runs anywhere that can run `uvicorn app.main:app --host 0.0.0.0 --port $PORT`, for example Render or Railway. Set the build command to `pip install -r requirements.txt && python run_pipeline.py`, add `GROQ_API_KEY` as an environment variable, and commit `data/train.csv`.
