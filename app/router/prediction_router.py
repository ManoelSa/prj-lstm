from fastapi import APIRouter, HTTPException, Depends
from app.schemas.prediction_schema import PredictionRequest, PredictionResponse
from app.config.settings import MODEL_PATH, SCALER_PATH, TIME_STEP
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import yfinance as yf
from datetime import datetime, timedelta
import logging
logger = logging.getLogger(__name__)


router = APIRouter()

# Carregamento Global dos Artefatos de ML (MODEL e SCALER)
try:
    MODEL = load_model(MODEL_PATH)
    SCALER = joblib.load(SCALER_PATH)
    logger.info("API: Modelo e Scaler carregados com sucesso para o router.")
except Exception as e:
    logger.critical(f"API ERRO CRÍTICO: Falha ao carregar artefatos de ML. {e}")
    MODEL, SCALER = None, None 

def get_ml_artifacts():
    """Dependência para garantir que os artefatos foram carregados."""
    if MODEL is None or SCALER is None:
        raise HTTPException(status_code=503, detail="Serviço indisponível. Artefatos de ML não carregados.")
    return MODEL, SCALER

@router.post("/predict/", response_model=PredictionResponse)
def predict_price(request: PredictionRequest, artifacts: tuple = Depends(get_ml_artifacts)):
    """
    Endpoint principal para prever o preço de fechamento do próximo dia,
    buscando os dados históricos necessários via yfinance.
    """
    model, scaler = artifacts
    
    ticker = request.ticker.upper()
    
    # --- 1. Busca e Coleta dos Dados Recentes (via yfinance) ---
    
    # Buscando período seguro (Os últimos 120 dias corridos).
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=100)).strftime('%Y-%m-%d')
    
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        
        if data.empty or len(data) < TIME_STEP:
            raise HTTPException(
                status_code=400,
                detail=f"Não foi possível obter os últimos {TIME_STEP} preços de fechamento (Close) para o ticker {ticker} no período recente."
            )
            
        # Pega a última janela de tempo (60 dias) para previsão
        recent_prices = data['Close'].tail(TIME_STEP).values
        
        if len(recent_prices) != TIME_STEP:
             raise HTTPException(
                status_code=400,
                detail=f"Dados insuficientes. Encontrados apenas {len(recent_prices)} dias úteis, mas {TIME_STEP} são necessários."
            )

    except Exception as e:
        # Erro de conexão, ticker inválido, etc.
        raise HTTPException(
            status_code=500, 
            detail=f"Falha ao buscar dados históricos via yfinance para {ticker}. Erro: {str(e)}"
        )

    # --- 2. Processamento ---
    # 1. Formatar para 2D (scaler)
    input_data = recent_prices.reshape(-1, 1)
    
    # 2. Escalonamento
    scaled_input = scaler.transform(input_data)
    
    # 3. Reshape para 3D da LSTM
    X_input = scaled_input.reshape(1, TIME_STEP, 1)

    # 4. Previsão e Desnormalização
    scaled_prediction = model.predict(X_input)
    prediction_original = scaler.inverse_transform(scaled_prediction)[0][0]

    # 5. Retorno
    ultimo_preco = recent_prices[-1]
    variacao_pct = ((prediction_original - ultimo_preco) / ultimo_preco) * 100

    #logger.info(f"{ticker}: atual={ultimo_preco:.2f}, previsto={prediction_original:.2f}, variação={variacao_pct:.2f}%")

    return PredictionResponse(
        ticker=ticker,
        ultimo_preco=round(float(ultimo_preco), 2),
        previsao_proximo_dia=round(float(prediction_original), 2),
        variacao_percentual=round(float(variacao_pct), 2),
        unidade="R$"
    )