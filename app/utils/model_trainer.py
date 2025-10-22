from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.preprocessing import MinMaxScaler
from app.utils.helpers import evaluate_predictions
from app.config.settings import TIME_STEP, EPOCHS, BATCH_SIZE, MODEL_PATH
import numpy as np

def create_lstm_model()-> Sequential:
    """
    Cria e compila o modelo LSTM para previsão de séries temporais.

    Returns:
        Sequential: Modelo LSTM compilado com otimizador Adam e perda MSE.
    """
    print("--- REQ: 2.1 Construção do Modelo ---")
    modelo = Sequential([
        Input(shape=(TIME_STEP, 1)),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
 
    modelo.compile(optimizer='adam', loss='mean_squared_error',metrics=['mae'])
    
    return modelo

def train_and_evaluate_model(
        modelo: Sequential,
        X_treino: np.ndarray,
        X_teste: np.ndarray,
        Y_treino: np.ndarray,
        Y_teste: np.ndarray,
        scaler: MinMaxScaler
)-> dict[str, float]:
    """
    Treina o modelo, avalia o desempenho e salva o modelo treinado.
    Args:
        modelo (Sequential): Modelo LSTM já compilado.
        X_treino (np.ndarray): Sequências de treino.
        X_teste (np.ndarray): Sequências de teste.
        Y_treino (np.ndarray): Valores alvo de treino.
        Y_teste (np.ndarray): Valores alvo de teste.
        scaler (MinMaxScaler): Escalonador usado para normalizar os dados.
    Returns:
        dict[str, float]: Dicionário com métricas RMSE, MAE e MAPE.
    """
    print(f"\n--- REQ: 2.2 Treinamento ({EPOCHS} Epochs) ---")
    modelo.fit(
        X_treino, 
        Y_treino, 
        validation_data=(X_teste, Y_teste), 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE, 
        verbose=1,
        shuffle=False  # garante reprodutibilidade
    )

    print("\n--- REQ: 2.3 Avaliação ---")
    
    Y_previsao = modelo.predict(X_teste)

    # Desnormalização
    Y_previsao_original = scaler.inverse_transform(Y_previsao)
    Y_teste_original = scaler.inverse_transform(Y_teste.reshape(-1, 1))

    # Métricas
    metrics = evaluate_predictions(Y_teste_original, Y_previsao_original)
    
    print(f"RMSE: {metrics['rmse']:.4f} R$")
    print(f"MAE: {metrics['mae']:.4f} R$")
    print(f"MAPE: {metrics['mape']:.2f} %")


    print("\n--- REQ: 3.1 Salvar o Modelo ---")
    modelo.save(MODEL_PATH)
    print(f"\nModelo treinado salvo em: {MODEL_PATH}")
    
    return metrics