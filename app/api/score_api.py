# app/api/score_api.py

from fastapi import APIRouter, HTTPException, Query
from app.schemas.scoring_response import ScoringResponse
from app.services.data_ingestion import fetch_user_data_by_pinfl
from app.services.ai_pipeline import run_full_ai_pipeline

router = APIRouter()


@router.post("/", response_model=ScoringResponse)
async def score_by_pinfl(pinfl: str = Query(..., description="14-digit applicant PINFL")):
    """
    Run full AI-based scoring for the given PINFL.
    """
    try:
        applicant = fetch_user_data_by_pinfl(pinfl)
        if not applicant:
            raise HTTPException(status_code=404, detail="Applicant not found")

        result = run_full_ai_pipeline(applicant)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")
