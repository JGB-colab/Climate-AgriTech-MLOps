from flask import Flask, request, jsonify
import joblib
import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = Flask(__name__)

def load_artifacts():
    project_root = Path(__file__).parent.parent / 'Climate-AgriTech-MLOps' 
    model_path = project_root / 'artifacts' / 'modelo_final.joblib'
    preprocessor_path = project_root / 'artifacts' / 'preprocessor.joblib'
    try:
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        logging.info("✅ Modelo e pré-processador carregados!")
        return model, preprocessor
    except FileNotFoundError:
        logging.error("❌ Erro: Arquivos de modelo não encontrados.")
        return None, None

modelo, preprocessor = load_artifacts()

@app.route('/prever', methods=['POST'])
def prever_produtividade():
    if modelo is None:
        return jsonify({'erro': 'Modelo não está disponível.'}), 503

    try:
        dados_entrada = request.get_json()
        df_input = pd.DataFrame([dados_entrada])

          
        colunas_renomear = {
            'Irrigation_Access_%': 'Irrigation_Access_%25'
        }
        df_input_corrigido = df_input.rename(columns=colunas_renomear)

        dados_processados = preprocessor.transform(df_input_corrigido)
        
        previsao = modelo.predict(dados_processados)
        resposta = {'previsao_produtividade_t_ha': float(round(previsao[0], 4))}
        logging.info(f"Previsão bem-sucedida: {resposta}")
        return jsonify(resposta)
        
    except Exception as e:
        logging.error(f"Erro inesperado durante a previsão: {e}", exc_info=True)
        return jsonify({'erro': 'Ocorreu um erro interno no servidor.'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)