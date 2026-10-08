"""FastAPI application: web pages + JSON API.

Start with:  uvicorn app.main:app --reload
"""
import json
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from src.config import EDA_SUMMARY, LISTINGS, METRICS_PATH, MODEL_PATH
from src.constants import BLDG_LABELS, KITCHEN_LABELS, NEIGHBORHOOD_NAMES

APP_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))


# ----------------------------------------------------------------------------- services
class Services:
    ready = False
    error = ""
    predictor = recommender = rag = assistant = None
    summary: dict = {}
    metrics: dict = {}


svc = Services()


def load_services():
    needed = [MODEL_PATH, METRICS_PATH, LISTINGS, EDA_SUMMARY]
    if any(not p.exists() for p in needed):
        svc.ready = False
        svc.error = "The model and data files have not been created yet. Run:  python run_pipeline.py"
        return
    from src.assistant import Assistant
    from src.predictor import Predictor
    from src.rag import RAGService
    from src.recommender import Recommender

    svc.predictor = Predictor()
    svc.recommender = Recommender(svc.predictor.listings)
    svc.rag = RAGService()
    svc.assistant = Assistant(svc.predictor.listings, svc.rag)
    svc.summary = json.loads(EDA_SUMMARY.read_text())
    svc.metrics = json.loads(METRICS_PATH.read_text())
    svc.ready = True


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_services()
    yield


app = FastAPI(title="Plotwise - AI Real Estate Intelligence Platform", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")


def require_ready():
    if not svc.ready:
        raise HTTPException(status_code=503, detail=svc.error)


# ----------------------------------------------------------------------------- schemas
class PredictIn(BaseModel):
    Neighborhood: str | None = None
    OverallQual: int | None = Field(None, ge=1, le=10)
    GrLivArea: float | None = Field(None, ge=300, le=6000)
    TotalBsmtSF: float | None = Field(None, ge=0, le=6000)
    LotArea: float | None = Field(None, ge=1000, le=250000)
    BedroomAbvGr: int | None = Field(None, ge=0, le=10)
    FullBath: int | None = Field(None, ge=0, le=5)
    TotRmsAbvGrd: int | None = Field(None, ge=2, le=15)
    GarageCars: int | None = Field(None, ge=0, le=5)
    Fireplaces: int | None = Field(None, ge=0, le=4)
    YearBuilt: int | None = Field(None, ge=1850, le=2025)
    YearRemodAdd: int | None = Field(None, ge=1850, le=2025)
    KitchenQual: str | None = None
    BldgType: str | None = None


class RecommendIn(BaseModel):
    budget: float = Field(..., gt=0, le=5_000_000)
    neighborhood: str | None = None
    min_bedrooms: int = Field(0, ge=0, le=10)
    min_bathrooms: float = Field(0, ge=0, le=6)
    min_quality: int = Field(1, ge=1, le=10)
    min_area: float = Field(0, ge=0, le=6000)
    garage: bool = False
    fireplace: bool = False
    top_k: int = Field(6, ge=1, le=12)


class ChatTurn(BaseModel):
    role: str
    content: str


class ChatIn(BaseModel):
    message: str = Field(..., max_length=1500)
    mode: str = "advisor"
    history: list[ChatTurn] = []


# ----------------------------------------------------------------------------- pages
def page(request: Request, name: str, **ctx):
    ctx.update(ready=svc.ready, error=svc.error, active=name)
    return templates.TemplateResponse(request, f"{name}.html", ctx)


def form_meta() -> dict:
    meta = svc.predictor.meta
    hoods = [{"code": c, "name": NEIGHBORHOOD_NAMES.get(c, c)} for c in meta["options"]["Neighborhood"]]
    hoods.sort(key=lambda h: h["name"])
    return {
        "defaults": meta["defaults"], "ranges": meta["ranges"], "neighborhoods": hoods,
        "kitchen": [{"code": c, "name": KITCHEN_LABELS.get(c, c)} for c in ["Ex", "Gd", "TA", "Fa", "Po"]
                    if c in meta["options"]["KitchenQual"]],
        "bldg": [{"code": c, "name": BLDG_LABELS.get(c, c)} for c in meta["options"]["BldgType"]],
    }


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    best = None
    if svc.ready:
        best = next(m for m in svc.metrics["models"] if m["name"] == svc.metrics["best_model"])
    return page(request, "dashboard", summary=svc.summary, metrics=svc.metrics, best=best)


@app.get("/predict", response_class=HTMLResponse)
def predict_page(request: Request):
    return page(request, "predict", meta=form_meta() if svc.ready else {})


@app.get("/recommend", response_class=HTMLResponse)
def recommend_page(request: Request):
    return page(request, "recommend", meta=form_meta() if svc.ready else {})


@app.get("/assistant", response_class=HTMLResponse)
def assistant_page(request: Request):
    return page(request, "assistant",
                backend=svc.rag.backend if svc.ready else "",
                llm_on=svc.assistant.llm.available if svc.ready else False)


# ----------------------------------------------------------------------------- API
@app.get("/health")
def health():
    return {"status": "ok" if svc.ready else "setup_needed", "detail": svc.error}


@app.post("/api/predict")
def api_predict(body: PredictIn):
    require_ready()
    values = body.model_dump(exclude_none=True)
    meta = svc.predictor.meta["options"]
    if "Neighborhood" in values and values["Neighborhood"] not in meta["Neighborhood"]:
        raise HTTPException(422, "Unknown neighborhood.")
    if "KitchenQual" in values and values["KitchenQual"] not in meta["KitchenQual"]:
        raise HTTPException(422, "Unknown kitchen quality.")
    if "BldgType" in values and values["BldgType"] not in meta["BldgType"]:
        raise HTTPException(422, "Unknown building type.")
    if values.get("YearRemodAdd") and values.get("YearBuilt") and values["YearRemodAdd"] < values["YearBuilt"]:
        values["YearRemodAdd"] = values["YearBuilt"]
    return svc.predictor.predict(values)


@app.post("/api/recommend")
def api_recommend(body: RecommendIn):
    require_ready()
    return svc.recommender.recommend(
        budget=body.budget, neighborhood=body.neighborhood or None, min_bedrooms=body.min_bedrooms,
        min_bathrooms=body.min_bathrooms, min_quality=body.min_quality, min_area=body.min_area,
        garage=body.garage, fireplace=body.fireplace, top_k=body.top_k)


@app.get("/api/property/{prop_id}/summary")
def api_summary(prop_id: int):
    require_ready()
    try:
        return svc.assistant.summarize_property(prop_id)
    except KeyError:
        raise HTTPException(404, "Property not found.")


@app.post("/api/chat")
def api_chat(body: ChatIn):
    require_ready()
    if not body.message.strip():
        raise HTTPException(422, "Type a question first.")
    if body.mode not in ("advisor", "docs"):
        raise HTTPException(422, "Mode must be 'advisor' or 'docs'.")
    return svc.assistant.answer(body.message, body.mode, [t.model_dump() for t in body.history])


@app.get("/api/documents")
def api_documents():
    require_ready()
    return {"documents": svc.rag.documents(), "backend": svc.rag.backend}


@app.post("/api/documents")
async def api_upload(file: UploadFile = File(...)):
    require_ready()
    data = await file.read()
    try:
        chunks = svc.rag.add_upload(file.filename or "document", data)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception:
        raise HTTPException(400, "That file could not be read. Check that it is a valid PDF, TXT or MD file.")
    return {"name": file.filename, "chunks": chunks}


@app.delete("/api/documents/{name}")
def api_delete(name: str):
    require_ready()
    try:
        svc.rag.remove(name)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"removed": name}
