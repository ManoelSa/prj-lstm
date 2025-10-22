from fastapi import FastAPI
from app.router import prediction_router

# Inicialização do FastAPI
app = FastAPI(
    title="PETR4 LSTM Predictor API",
    version="1.0.0",
    description="API para previsão de preços de fechamento da PETR4 usando modelo LSTM."
)

# Incluir o Router de Previsão
app.include_router(prediction_router.router, tags=["Previsão"])

@app.get("/")
def home():
    """Health check endpoint."""
    return {"message": "API de Previsão PETR4 está Online."}

# uvicorn app.main:app --reload