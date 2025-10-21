from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import yfinance as yf
import pandas as pd
import numpy as np

# Definições
TICKER = "PETR4.SA"
START_DATE = "2015-01-01"
END_DATE = datetime.now().strftime('%Y-%m-%d')
TIME_STEP = 60  # Janela de tempo de 60 dias

print(f"1.1 Coletando dados para {TICKER} de {START_DATE} a {END_DATE}...")

# Coleta
try:
    dados_originais = yf.download(TICKER, start=START_DATE, end=END_DATE)
    if dados_originais.empty:
        raise ValueError("Dataset vazio.")
    print(f"Coleta concluída. Total de registros: {len(dados_originais)}")
except Exception as e:
    print(f"Erro na coleta de dados: {e}")
    dados_originais = pd.DataFrame() # Cria um DataFrame vazio em caso de erro

if dados_originais.empty:
    exit() # Interrompe o script se não houver dados



print("\n1.2 Pré-processamento e Estruturação...")

# 1. Seleção e Limpeza
dados_fechamento = dados_originais[['Close']].copy()

# 2. Normalização (Escalonamento)
scaler = MinMaxScaler(feature_range=(0, 1))
dados_escalonados = scaler.fit_transform(dados_fechamento['Close'].values.reshape(-1, 1))

# 3. Criação das Sequências de Tempo (X e Y)
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), 0]) # 60 dias de entrada
        Y.append(data[i + time_step, 0])      # Preço do dia 61 (saída)
    return np.array(X), np.array(Y)

X, Y = create_dataset(dados_escalonados, TIME_STEP)

# 4. Reshape para LSTM: [amostras, time_step, features]
X = X.reshape(X.shape[0], X.shape[1], 1)

print(f"Dados estruturados. Dimensão de X: {X.shape}")
print(f"Dimensão de Y: {Y.shape}")

# 5. Divisão Treino/Teste (80% treino, 20% teste)
train_size = int(len(X) * 0.80)
X_treino, X_teste = X[0:train_size], X[train_size:len(X)]
Y_treino, Y_teste = Y[0:train_size], Y[train_size:len(Y)]

print(f"Treino: {len(X_treino)} amostras. Teste: {len(X_teste)} amostras.")