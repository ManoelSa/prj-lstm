import os

# --- Configurações de Dados e Treinamento ---
TICKER = "PETR4.SA"
START_DATE = "2015-01-01"
TIME_STEP = 60
TEST_SIZE_RATIO = 0.20 # 20% para teste
EPOCHS = 20
BATCH_SIZE = 64

# --- Caminhos dos Arquivos ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_FILENAME = "modelo_lstm_petr4.h5"
SCALER_FILENAME = "scaler_petr4.pkl"

MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
SCALER_PATH = os.path.join(MODEL_DIR, SCALER_FILENAME)