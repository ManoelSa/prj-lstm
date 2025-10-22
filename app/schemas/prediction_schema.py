# app/schemas/prediction_schema.py

from pydantic import BaseModel
from app.config.settings import TIME_STEP

class HistoricalPrices(BaseModel):
    """
    Schema de entrada: lista de 60 preços (R$)
    """
    prices: list[float]

class PredictionResponse(BaseModel):
    """
    Schema de saída da previsão.
    """
    ticker: str
    previsao_proximo_dia: float
    unidade: str = "R$"