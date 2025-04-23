from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.score_api import router as score_router

app = FastAPI(
    title="AI Credit Scoring API",
    description=(
        "⚙️ Fully Deep Learning-based Credit Risk, Fraud & Limit Scoring System.\n"
        "- Built with Transformers, LSTM, VAE, SimCLR.\n"
        "- Includes explainability and segmentation."
    ),
    version="1.0.0",
    contact={
        "name": "Taqsim Solution",
        "url": "https://taqsimsolution.uz",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    }
)

# CORS: allow all origins for now (can be restricted later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routing
app.include_router(score_router, prefix="/score", tags=["Scoring"])

@app.get("/", tags=["Health"])
def health_check():
    """
    Check if API is up and responding.
    """
    return {"status": "ok", "message": "AI Credit Scoring API is live 🚀"}
