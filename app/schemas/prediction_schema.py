from pydantic import BaseModel

class PredictionResponse(BaseModel):
    """
    Schema de saída da previsão de preço da ação.
    """
    ticker: str
    ultima_data: str
    ultimo_preco: float
    previsao_proximo_dia: float
    variacao_percentual: float
    unidade: str = "R$"