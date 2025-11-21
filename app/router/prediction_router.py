from fastapi import APIRouter, HTTPException, Depends
from app.schemas.prediction_schema import PredictionResponse
from app.config.settings import MODEL_PATH, SCALER_PATH, TIME_STEP
from app.config.security import verify_token
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
from prometheus_client import Counter, Histogram, Gauge
import joblib
import yfinance as yf
import logging
import psutil
import time


logger = logging.getLogger(__name__)


router = APIRouter()
# REQUISITO 5: Escalabilidade e Monitoramento
# --- Métricas Prometheus ---
PREDICT_REQUESTS = Counter(
    "predict_requests_total", "Número total de requisições ao endpoint /predict/petr4"
)
PREDICT_LATENCY = Histogram(
    "predict_latency_seconds", "Tempo de resposta do endpoint /predict/petr4 (segundos)"
)
CPU_USAGE = Gauge("predict_cpu_usage_percent", "Uso de CPU (%) durante a predição")
MEM_USAGE = Gauge("predict_memory_usage_percent", "Uso de memória (%) durante a predição")


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

@router.post("/predict/petr4", response_model=PredictionResponse)
def predict_price(artifacts: tuple = Depends(get_ml_artifacts), token: str = Depends(verify_token)):
    """
    Endpoint responsável por prever o preço de fechamento do próximo pregão (D+1)
    da ação PETR4.SA, utilizando um modelo LSTM previamente treinado com base
    em janelas temporais deslizantes de 60 dias consecutivos de histórico.
    """
    start_time = time.time()
    PREDICT_REQUESTS.inc()  # incrementa contagem de requisições

    model, scaler = artifacts
    
    ticker = "PETR4.SA" # Valor Fixo, Empresa utilizada para o treinamento.

    CPU_USAGE.set(psutil.cpu_percent(interval=None))
    MEM_USAGE.set(psutil.virtual_memory().percent)
       
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

        #Obtendo a ultima data do fechamento
        ultima_data = data.index[-1].strftime("%Y-%m-%d")
        
        if len(recent_prices) != TIME_STEP:
             raise HTTPException(
                status_code=400,
                detail=f"Dados insuficientes. Encontrados apenas {len(recent_prices)} dias úteis, mas {TIME_STEP} são necessários."
            )

    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Falha ao buscar dados históricos via yfinance para {ticker}. Erro: {str(e)}"
        )

    input_data = recent_prices.reshape(-1, 1)
    scaled_input = scaler.transform(input_data)
    X_input = scaled_input.reshape(1, TIME_STEP, 1)

    # Previsão e Desnormalização
    scaled_prediction = model.predict(X_input)
    prediction_original = scaler.inverse_transform(scaled_prediction)[0][0]

    # Retorno da predição
    ultimo_preco = recent_prices[-1]
    variacao_pct = ((prediction_original - ultimo_preco) / ultimo_preco) * 100

    # Finaliza medição de tempo
    elapsed = time.time() - start_time
    PREDICT_LATENCY.observe(elapsed)
    logger.info(f"Tempo de resposta /predict/petr4: {elapsed:.3f}s")

    return PredictionResponse(
        ticker=ticker,
        ultima_data=ultima_data,
        ultimo_preco=round(float(ultimo_preco), 2),
        previsao_proximo_dia=round(float(prediction_original), 2),
        variacao_percentual=round(float(variacao_pct), 2),
        unidade="R$"
    )