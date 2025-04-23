import typer
from train_pipeline import train

app = typer.Typer()

@app.command()
def run_training(epochs: int = 30, batch_size: int = 32):
    """
    Launch deep model training for credit scoring.
    """
    train(epochs=epochs, batch_size=batch_size)

if __name__ == "__main__":
    app()
