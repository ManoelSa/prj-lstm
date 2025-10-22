import os
import random
import numpy as np
import tensorflow as tf

from app.utils.data_pipeline import load_and_preprocess_data
from app.utils.model_trainer import create_lstm_model, train_and_evaluate_model
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    print("\n")
    AMBIENTE = os.getenv("PIPELINE_AMBIENTE", "dev")
    
    """
    Número que define o ponto inicial da aleatoriedade,
    garantindo reprodutibilidade dos resultados em diferentes execuções e ambientes durante os testes.
    """
    SEED = int(os.getenv("SEED", 42))

    if AMBIENTE == "dev":
        np.random.seed(SEED)
        random.seed(SEED)
        tf.random.set_seed(SEED)
        os.environ["PYTHONHASHSEED"] = str(SEED)
        print(f"AMBIENTE DEV: Seed global fixada em {SEED}")
    else:
        print("AMBIENTE PRD: Seed não fixada (execuções variáveis)")

    print("\n")
    print("Iniciando Processo de Treinamento e Avaliação de Modelo LSTM.\n")

    # REQUISITO 1: Coleta e Pré-processamento dos Dados
    X_treino, X_teste, Y_treino, Y_teste, scaler = load_and_preprocess_data()
    
    if X_treino is None:
        print("Treinamento não pode ser iniciado devido a erro na coleta de dados.")
    else:
        # REQUISITO 2: Desenvolvimento do Modelo LSTM
        modelo = create_lstm_model()
        
        # REQUISITO 3: Salvamento e Exportação do Modelo
            #Contem partes do req 2
        metrics = train_and_evaluate_model(modelo, X_treino, X_teste, Y_treino, Y_teste, scaler)
        
        print("\n--- Pipeline de Treinamento Concluída ---")
        print(f"Modelo pronto para deploy. MAE final: {metrics['mae']:.4f} R$\n")