import yfinance as yf
from datetime import datetime
import joblib
from sklearn.preprocessing import MinMaxScaler
from app.utils.helpers import create_sequences
from app.config.settings import TICKER, START_DATE, TIME_STEP, TEST_SIZE_RATIO, SCALER_PATH, MODEL_DIR
import os
import numpy as np

def load_and_preprocess_data() -> tuple[
    np.ndarray   | None,
    np.ndarray   | None,
    np.ndarray   | None,
    np.ndarray   | None,
    MinMaxScaler | None
]:
    """
    Coleta dados, faz o escalonamento, salva o scaler e estrutura 
    os dados em conjuntos de treino e teste.
    Returns:
        (X_treino, X_teste, Y_treino, Y_teste, scaler):
            Arrays numpy com dados de treino/teste e o MinMaxScaler usado.
            Retorna (None, None, None, None, None) em caso de erro.
    """
    print(f"--- 1. Coletando dados para {TICKER} ---")
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    try:
        dados_originais = yf.download(TICKER, start=START_DATE, end=end_date, auto_adjust=True)
        if dados_originais.empty:
            raise ValueError("Dataset vazio. Verifique o ticker.")
    except Exception as e:
        print(f"Erro na coleta de dados: {e}")
        return None, None, None, None, None

    # 1. Seleção da feature e Escalamento
    dados_fechamento = dados_originais[['Close']].copy()
    # Scaler: normaliza os dados para a faixa [0, 1] e melhora o desempenho do modelo
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    dados_escalonados = scaler.fit_transform(dados_fechamento['Close'].values.reshape(-1, 1))

    # 2. Estruturação em Sequências (X e Y)
    X, Y = create_sequences(dados_escalonados, TIME_STEP)

    # 3. Divisão Treino/Teste
    train_size = int(len(X) * (1 - TEST_SIZE_RATIO))
    X_treino, X_teste = X[0:train_size], X[train_size:len(X)]
    Y_treino, Y_teste = Y[0:train_size], Y[train_size:len(Y)]

    # 4. Salvamento do Scaler (Requisito 3)
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler salvo em: {SCALER_PATH}")
    
    return X_treino, X_teste, Y_treino, Y_teste, scaler