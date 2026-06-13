"""
Early Sepsis Detection API — FastAPI Application

Production-ready ML API serving real-time sepsis risk predictions
from an ensemble model (RF + XGBoost + LightGBM + GBM) with
SHAP explainability and natural language clinical justifications.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router, load_artifacts


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Loads model artifacts once at startup — NOT on every request.
    """
    print("\n🚀 Starting Early Sepsis Detection API...")
    print("📦 Loading model artifacts...")
    load_artifacts()
    print("✅ Startup complete!\n")
    yield
    print("\n👋 Shutting down Early Sepsis Detection API...")


app = FastAPI(
    title="Early Sepsis Detection API",
    description=(
        "Ensemble ML model for early sepsis risk prediction from patient vitals. "
        "Features SHAP explainability and NLG clinical justifications."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins in development, tighten in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
