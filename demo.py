import requests
import json

dados_para_prever = {
    "Ano": 2025,
    "Pais": "Brazil",
    "Regiao": "Southeast",
    "Tipo_Cultura": "Corn",
    "Temp_C": 22.5,
    "Chuva_mm": 1250.0,
    "CO2_Emissions_MT": 18.5,
    "Eventos_Extremos": 4,
    "Irrigation_Access_%": 70.0,
    "Uso_Pesticida_kg_ha": 110.0,
    "Uso_Fertilizante_kg_ha": 350.0,
    "Indice_Saude_Solo": 78.5,
    "Estrategias_Adaptacao": "Water Management",
    "Impacto_Economico_USD_Milhoes": 950.0
}

api_url = "http://127.0.0.1:5000/prever"
headers = {"Content-Type": "application/json"}

print("➡️  Enviando dados para a API...")
try:
    response = requests.post(api_url, headers=headers, data=json.dumps(dados_para_prever))
    response.raise_for_status() 
    resultado = response.json()
    print("\n✅ Previsão recebida com sucesso!")
    print(json.dumps(resultado, indent=2))
except requests.exceptions.HTTPError as http_err:
    print(f"\n❌ Erro HTTP ao chamar a API.")
    print(f"   Status Code: {http_err.response.status_code}")
    try:
        print(f"   Resposta: {json.dumps(http_err.response.json(), indent=2)}")
    except json.JSONDecodeError:
        print(f"   Resposta (não-JSON): {http_err.response.text}")
except requests.exceptions.RequestException as req_err:
    print(f"\n❌ Erro de conexão com a API: {req_err}")