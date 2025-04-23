from fastapi import APIRouter, BackgroundTasks
from train_pipeline import train

router = APIRouter()

@router.post("/train/start")
def start_training(background_tasks: BackgroundTasks):
    """
    Trigger model training in background.
    """
    background_tasks.add_task(train)
    return {"status": "Training started in background"}
