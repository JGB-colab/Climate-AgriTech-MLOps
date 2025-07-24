import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
#fgas
def dividir_dados_em_conjuntos(df: pd.DataFrame):
    """
    Separa o DataFrame em features (X) e alvo (y), e depois divide
    em conjuntos de treino (60%), validação (20%) e teste (20%).
    Args:
        df (pd.DataFrame): O DataFrame limpo.
    Retorna:
        tuple: Uma tupla contendo X_train, X_val, X_test, y_train, y_val, y_test.
    """
    print("--- Iniciando Tarefa: Divisão dos Dados ---")

    X = df.drop('Produtividade_t_ha', axis=1)
    y = df['Produtividade_t_ha']

    # Primeiro split: 80% para treino+validação, 20% para teste
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Segundo split: do conjunto de 80%, tira 25% para validação (0.25 * 0.8 = 0.2 do total)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

    print(f"Tamanho do treino: {len(X_train)} ({len(X_train)/len(df):.0%})")
    print(f"Tamanho da validação: {len(X_val)} ({len(X_val)/len(df):.0%})")
    print(f"Tamanho do teste: {len(X_test)} ({len(X_test)/len(df):.0%})")

    return X_train, X_val, X_test, y_train, y_val, y_test


X_train, X_val, X_test, y_train, y_val, y_test = dividir_dados_em_conjuntos(df)

def criar_e_aplicar_preprocessamento(X_train, X_val, X_test):
    """
    Cria um pipeline de pré-processamento que normaliza dados numéricos e
    codifica dados categóricos. Treina o pipeline com os dados de treino e
    o aplica a todos os conjuntos.
    Args:
        X_train, X_val, X_test: DataFrames de features.
    Retorna:
        tuple: Contendo os conjuntos de features processados e o pipeline treinado.
    """
    print("--- Iniciando Tarefa: Pré-processamento Avançado ---")

    # Identifica colunas numéricas e categóricas
    features_numericas = X_train.select_dtypes(include=np.number).columns.tolist()
    features_categoricas = X_train.select_dtypes(exclude=np.number).columns.tolist()

    # Cria os transformadores
    transformador_numerico = StandardScaler()
    transformador_categorico = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # Cria o ColumnTransformer para aplicar transformações diferentes a colunas diferentes
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', transformador_numerico, features_numericas),
            ('cat', transformador_categorico, features_categoricas)
        ],
        remainder='passthrough'
    )

    # Treina o pipeline de pré-processamento APENAS com dados de treino
    preprocessor.fit(X_train)

    # Aplica a transformação a todos os conjuntos
    X_train_proc = preprocessor.transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)

    print("✅ Dados normalizados e codificados.")

    return X_train_proc, X_val_proc, X_test_proc, preprocessor

X_train_proc, X_val_proc, X_test_proc, preprocessor = criar_e_aplicar_preprocessamento(X_train, X_val, X_test)

def ajustar_hiperparametros(X_train_proc, y_train):
    """
    Usa RandomizedSearchCV para encontrar a melhor combinação de hiperparâmetros
    para o modelo RandomForestRegressor.
    Args:
        X_train_proc: Dados de treino processados.
        y_train: Alvo de treino.
    Retorna:
        dict: O dicionário com os melhores parâmetros encontrados.
    """
    print("--- Iniciando Tarefa: Ajuste de Hiperparâmetros ---")

    # Define o espaço de parâmetros para a busca
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', 1.0]
    }

    rf = RandomForestRegressor(random_state=42, n_jobs=-1)

    # Configura a busca aleatória com validação cruzada
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=20,  # Número de combinações a testar
        cv=3,       # Número de folds da validação cruzada
        verbose=2,
        random_state=42,
        n_jobs=-1
    )

    # Executa a busca
    random_search.fit(X_train_proc, y_train)

    print(f"✅ Melhores parâmetros encontrados: {random_search.best_params_}")
    return random_search.best_params_

best_params = ajustar_hiperparametros(X_train_proc, y_train)


def treinar_e_avaliar_modelo_final(best_params, X_train_proc, y_train, X_val_proc, y_val, X_test_proc, y_test):
    """
    Treina o modelo final com os melhores hiperparâmetros no conjunto combinado
    de treino e validação. Depois, avalia no conjunto de teste.
    Args:
        best_params (dict): Melhores hiperparâmetros.
        (outros): Todos os conjuntos de dados.
    Retorna:
        object: O objeto do modelo final treinado.
    """
    print("--- Iniciando Tarefa: Treinamento e Avaliação Final ---")

    # Combina dados de treino e validação para o treino final
    X_train_full = np.vstack((X_train_proc, X_val_proc))
    y_train_full = np.concatenate((y_train, y_val))

    # Cria o modelo final com os melhores parâmetros
    modelo_final = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)

    # Treina o modelo final
    modelo_final.fit(X_train_full, y_train_full)
    print("✅ Modelo final treinado com dados de treino + validação.")

    # Avalia no conjunto de teste (dados nunca vistos)
    y_pred = modelo_final.predict(X_test_proc)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\n--- Avaliação Final no Conjunto de Teste ---")
    print(f"MAE (Erro Médio Absoluto): {mae:.4f}")
    print(f"RMSE (Raiz do Erro Quadrático Médio): {rmse:.4f}")

    return modelo_final

modelo_final = treinar_e_avaliar_modelo_final(best_params, X_train_proc, y_train, X_val_proc, y_val, X_test_proc, y_test)

def salvar_artefatos(modelo, preprocessor):
    """
    Salva o modelo treinado e o pipeline de pré-processamento em arquivos.
    Args:
        modelo: O objeto do modelo treinado.
        preprocessor: O objeto do ColumnTransformer treinado.
    """
    print("--- Iniciando Tarefa: Salvando Artefatos ---")
    joblib.dump(modelo, 'modelo_final.joblib')
    joblib.dump(preprocessor, 'preprocessor.joblib')
    print("✅ Modelo final e pipeline de pré-processamento salvos.")

salvar_artefatos(modelo_final, preprocessor)