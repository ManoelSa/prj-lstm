from fastapi import FastAPI
from app.router import prediction_router
from app.config import security

# Inicialização do FastAPI
app = FastAPI(
    title="API PETR4",
    version="1.0.0",
    description="API para previsão de preços de fechamento da PETR4 usando modelo LSTM."
)

#Rota de Boas Vindas/Verificação 
@app.get("/", tags=["Health"])
def home():
    """Health endpoint"""
    return {"message": "API de Previsão PETR4 está Online."}

# Rota de Previsão
app.include_router(prediction_router.router, tags=["Previsão"])

#Autenticação
app.include_router(security.router)