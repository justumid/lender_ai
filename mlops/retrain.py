import logging
from training.trainer import run_training_pipeline
from datetime import datetime

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    logging.info(f"[{datetime.now()}] 🚀 Retraining process started...")
    try:
        run_training_pipeline()
        logging.info(f"[{datetime.now()}] ✅ Retraining completed successfully.")
    except Exception as e:
        logging.error(f"[{datetime.now()}] ❌ Retraining failed: {e}")
