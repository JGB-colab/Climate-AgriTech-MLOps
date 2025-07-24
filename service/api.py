# Arquivo: app.py

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

# 1. Inicializa a aplicação Flask
app = Flask(__name__)

# --- Carregamento dos Artefatos de Machine Learning ---
# Carregamos tanto o modelo quanto o pipeline de pré-processamento
try:
    modelo = joblib.load('modelo_final.joblib')
    preprocessor = joblib.load('preprocessor.joblib')
    print("✅ Modelo e pré-processador carregados com sucesso!")
except FileNotFoundError as e:
    print(f"❌ Erro: Não foi possível carregar os arquivos do modelo. Verifique se 'modelo_final.joblib' e 'preprocessor.joblib' existem.")
    print(e)
    modelo, preprocessor = None, None

# --- Endpoint de Previsão ---
@app.route('/prever', methods=['POST'])
def prever_produtividade():
    if modelo is None or preprocessor is None:
        return jsonify({'erro': 'Modelo ou pré-processador não foram carregados.'}), 500

    # Pega os dados JSON da requisição
    dados_entrada = request.get_json()
    if not dados_entrada:
        return jsonify({'erro': 'Nenhum dado foi enviado no corpo da requisição.'}), 400

    try:
        # Converte os dados de entrada em um DataFrame do pandas
        # É crucial que ele tenha as mesmas colunas usadas no treino
        df_input = pd.DataFrame([dados_entrada])

        # Aplica o MESMO pré-processamento treinado anteriormente
        dados_processados = preprocessor.transform(df_input)

        # Faz a previsão com os dados já processados
        previsao = modelo.predict(dados_processados)

        # Formata a resposta
        resposta = {
            'previsao_produtividade_t_ha': round(previsao[0], 4)
        }

        return jsonify(resposta)

    except Exception as e:
        # Captura qualquer outro erro durante o processo
        return jsonify({'erro': f'Ocorreu um erro durante a previsão: {str(e)}'}), 500

# --- Ponto de Entrada do Servidor ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)