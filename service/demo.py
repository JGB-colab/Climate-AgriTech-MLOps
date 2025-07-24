# Arquivo: cliente_api.py (Versão Corrigida e Final)

import requests # Biblioteca para fazer requisições HTTP
import json

# 1. URL do nosso endpoint da API que está rodando localmente
url_api = 'http://127.0.0.1:5000/prever'

# 2. Dicionário com dados de exemplo completos para a nova previsão.
#    Agora inclui TODAS as features que o modelo espera, incluindo as de texto.
dados_para_prever = {
    # --- Features Numéricas ---
    "Ano": 2025,
    "Temperatura_C": 22.5,
    "Precipitacao_mm": 1250.0,
    "Emissoes_CO2_MT": 18.5,
    "Eventos_Climaticos_Extremos": 4,
    "Acesso_Irrigacao_Pct": 70.0,
    "Uso_Pesticida_kg_ha": 110.0,
    "Uso_Fertilizante_kg_ha": 350.0,
    "Indice_Saude_Solo": 78.5,
    "Impacto_Economico_USD_Milhoes": 950.0,

    # --- !! Features Categóricas (Agora Obrigatórias) !! ---
    "Pais": "Brazil",
    "Regiao": "Southeast",
    "Tipo_Cultura": "Corn",
    "Estrategias_Adaptacao": "Water Management"
}

# 3. Enviamos a requisição POST para a API, com os dados em formato JSON
print("➡️  Enviando dados para a API...")
try:
    response = requests.post(url_api, json=dados_para_prever)

    # 4. Verificamos se a requisição foi bem-sucedida (código de status 200)
    if response.status_code == 200:
        # Extrai e exibe a previsão recebida do servidor
        resultado = response.json()
        previsao = resultado['previsao_produtividade_t_ha']
        print(f"\n✅ Previsão recebida com sucesso!")
        print(f"   A produtividade prevista para estas condições é de: {previsao} t/ha")
    else:
        # Exibe uma mensagem de erro se algo deu errado
        print(f"\n❌ Erro ao chamar a API.")
        print(f"   Status Code: {response.status_code}")
        print(f"   Resposta: {response.text}")

except requests.exceptions.ConnectionError as e:
    print("\n❌ Erro de Conexão: Não foi possível se conectar à API.")
    print("   Verifique se o servidor Flask (app.py) está rodando no outro terminal.")