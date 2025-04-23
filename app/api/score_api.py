from fastapi import APIRouter, Query, HTTPException
from app.services.data_ingestion import fetch_user_data_by_pinfl
from app.services.training_pipeline import run_full_scoring_pipeline
from app.schemas.applicant_schema import ScoringResponse

router = APIRouter(tags=["Scoring"])

@router.get("/score", response_model=ScoringResponse)
def get_score(pinfl: str = Query(..., min_length=10, max_length=20)):
    """
    Run the full AI-based credit scoring pipeline for a given applicant by PINFL.

    Returns risk score, loan limit, fraud risk, approval decision, explanations, etc.
    """
    applicant = fetch_user_data_by_pinfl(pinfl)
    if not applicant:
        raise HTTPException(status_code=404, detail=f"No applicant found for PINFL: {pinfl}")

    result = run_full_scoring_pipeline(applicant)
    return result
