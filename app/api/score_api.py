from fastapi import APIRouter, Query
from app.services.data_ingestion import fetch_user_data_by_pinfl
from app.services.ai_pipeline import run_full_scoring_pipeline
from app.schemas.applicant_schema import ScoringResponse

router = APIRouter()

@router.get("/", response_model=ScoringResponse)
def get_score(pinfl: str = Query(..., min_length=10, max_length=20)):
    """
    Run full AI scoring pipeline for the given user's PINFL.
    """
    applicant = fetch_user_data_by_pinfl(pinfl)
    if not applicant:
        return {"status": "error", "message": "Applicant not found", "pinfl": pinfl}

    result = run_full_scoring_pipeline(applicant)
    return result
