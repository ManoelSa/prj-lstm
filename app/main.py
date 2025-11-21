from fastapi import FastAPI
from app.router import prediction_router
from app.config import security
from starlette.responses import Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

# REQUISITO 4.1: Criação de API

# Inicialização do FastAPI
app = FastAPI(
    title="API PETR4",
    version="1.0.0",
    description="API para previsão de preços de fechamento da PETR4 usando modelo LSTM."
)

# Rota de Boas Vindas/Verificação 
@app.get("/", tags=["Health"])
def home():
    """Health endpoint"""
    return {"message": "API de Previsão PETR4 está Online."}

# Endpoint para métricas Prometheus
@app.get("/metrics", tags=["Monitoramento"])
def metrics():
    """
    Endpoint que expõe as métricas coletadas (Prometheus).
    As métricas incluem:
    - Total de requisições /predict/petr4
    - Tempo de resposta
    - Uso de CPU e Memória
    """
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Rota de Previsão
app.include_router(prediction_router.router, tags=["Previsão"])

# Autenticação
app.include_router(security.router)