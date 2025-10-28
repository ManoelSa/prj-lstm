from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    """
    Schema de entrada: pede apenas o ticker.
    """
    ticker: str = Field(..., example="PETR4.SA", description="Código da ação (Ticker) a ser previsto.")

class PredictionResponse(BaseModel):
    """
    Schema de saída da previsão de preço da ação.
    """
    ticker: str
    ultimo_preco: float
    previsao_proximo_dia: float
    variacao_percentual: float
    unidade: str = "R$"