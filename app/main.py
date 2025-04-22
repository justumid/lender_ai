# app/main.py

from fastapi import FastAPI
from app.api import score_api

app = FastAPI(
    title="AI Credit Scoring API",
    description="Fully AI-based scoring system using deep learning",
    version="1.0.0"
)

# Register scoring route
app.include_router(score_api.router, prefix="/score", tags=["Scoring"])
