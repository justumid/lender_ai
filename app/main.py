from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.score_api import router as score_router

app = FastAPI(
    title="AI Credit Scoring API",
    description="⚙️ Fully Deep Learning-based Credit Risk, Fraud & Limit Scoring API.\n\nPowered by Transformers, LSTM, VAE, SimCLR.",
    version="1.0.0",
    contact={
        "name": "Taqsim Solution",
        "url": "https://taqsimsolution.uz",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT"
    }
)

# Optional: CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(score_router, prefix="/score")

@app.get("/", tags=["Health Check"])
def health_check():
    return {"status": "ok", "message": "AI scoring API is running"}
