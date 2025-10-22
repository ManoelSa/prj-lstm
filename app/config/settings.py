import os
from datetime import datetime, timedelta

# --- Configurações de Dados e Treinamento ---
TICKER = "PETR4.SA"
START_DATE = (datetime.now() - timedelta(days=365 * 3)).strftime('%Y-%m-%d') # Dados de 3 anos
TIME_STEP = 60         # Número de dias usados como janela de entrada
TEST_SIZE_RATIO = 0.20 # 20% para teste
EPOCHS = 20            # Quantas vezes o modelo vê todo o dataset
BATCH_SIZE = 64        # Tamanho do lote de amostras usado a cada atualização de pesos

# --- Caminhos dos Arquivos ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_FILENAME = "modelo_lstm_petr4.keras"
SCALER_FILENAME = "scaler_petr4.pkl"

MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
SCALER_PATH = os.path.join(MODEL_DIR, SCALER_FILENAME)