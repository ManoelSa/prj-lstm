# app/router/prediction_router.py

from fastapi import APIRouter, HTTPException, Depends
from app.schemas.prediction_schema import HistoricalPrices, PredictionResponse
from app.config.settings import MODEL_PATH, SCALER_PATH, TIME_STEP
import numpy as np
import joblib
from tensorflow.keras.models import load_model

router = APIRouter()

# Carregamento Global dos Artefatos de ML
# Acontece uma única vez ao iniciar a API
try:
    MODEL = load_model(MODEL_PATH)
    SCALER = joblib.load(SCALER_PATH)
    print("API: Modelo e Scaler carregados com sucesso para o router.")
except Exception as e:
    print(f"API ERRO CRÍTICO: Falha ao carregar artefatos de ML. {e}")
    MODEL, SCALER = None, None 

def get_ml_artifacts():
    """Dependência para garantir que os artefatos foram carregados."""
    if MODEL is None or SCALER is None:
        raise HTTPException(status_code=503, detail="Serviço indisponível. Artefatos de ML não carregados.")
    return MODEL, SCALER

@router.post("/predict/", response_model=PredictionResponse)
def predict_price(data: HistoricalPrices, artifacts: tuple = Depends(get_ml_artifacts)):
    """
    Endpoint principal para prever o preço de fechamento do próximo dia.
    Requisito 4: API RESTful.
    """
    model, scaler = artifacts

    if len(data.prices) != TIME_STEP:
        raise HTTPException(
            status_code=400,
            detail=f"Requer exatamente {TIME_STEP} preços históricos de entrada."
        )

    # 1. Preparação dos dados
    input_data = np.array(data.prices).reshape(-1, 1)
    
    # 2. Escalonamento
    scaled_input = scaler.transform(input_data)
    
    # 3. Reshape para 3D da LSTM
    X_input = scaled_input.reshape(1, TIME_STEP, 1)

    # 4. Previsão e Desnormalização
    scaled_prediction = model.predict(X_input)
    prediction_original = scaler.inverse_transform(scaled_prediction)[0][0]

    # 5. Retorno com Pydantic Schema
    return PredictionResponse(
        ticker="PETR4.SA",
        previsao_proximo_dia=round(float(prediction_original), 4)
    )