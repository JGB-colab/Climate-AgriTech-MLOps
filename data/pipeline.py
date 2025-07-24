import mlcroissant as mlc
import pandas as pd
import shutil
import os

class ETL():
    def __init__(self):
        self.df = None

    
    def extrair_e_limpar_dados(self):
        """
        Orquestra o download, carregamento, limpeza e renomeação dos dados.
        Esta função representa a primeira grande etapa do pipeline.
        Retorna:
            pd.DataFrame: O DataFrame limpo e pronto para a próxima etapa.
        """
        print("--- Iniciando Tarefa: Extração e Limpeza ---")

        # Limpeza de cache para garantir dados frescos
        cache_path = os.path.join(os.path.expanduser("~"), ".cache", "mlcroissant")
        if os.path.exists(cache_path):
            shutil.rmtree(cache_path)
            print(f"Cache em {cache_path} removido.")

        # Download via mlcroissant
        croissant_url = 'https://www.kaggle.com/datasets/waqi786/climate-change-impact-on-agriculture/croissant/download'
        croissant_dataset = mlc.Dataset(croissant_url)
        record_set_ref = croissant_dataset.metadata.record_sets[0]
        self.df = pd.DataFrame(croissant_dataset.records(record_set=record_set_ref.uuid))
        print("✅ Download e carregamento concluídos.")

        # Limpeza de nomes de colunas (prefixo)
        self.df.columns = [col.split('/')[-1] for col in self.df.columns]

        # Renomeação para nomes intuitivos em português
        nomes_novos = {
        'Average_Temperature_C': 'Temp_C',
        'Total_Precipitation_mm': 'Chuva_mm',
        'Crop_Yield_MT_per_HA': 'Produtividade_t_ha',
        'Extreme_Weather_Events': 'Eventos_Extremos',
        'Pesticide_Use_KG_per_HA': 'Uso_Pesticida_kg_ha',
        'Fertilizer_Use_KG_per_HA': 'Uso_Fertilizante_kg_ha',
        'Year': 'Ano',
        'Country': 'Pais',
        'Region': 'Regiao',
        'Crop_Type': 'Tipo_Cultura',
        'Irrigation_Access_%': 'Acesso_Irrigacao_Pct',
        'Soil_Health_Index': 'Indice_Saude_Solo',
        'Adaptation_Strategies': 'Estrategias_Adaptacao',
        'Economic_Impact_Million_USD': 'Impacto_Economico_USD_Milhoes'
    }
        self.df = self.df.rename(columns=nomes_novos)

        # Decodificação de valores bytes
        colunas_para_decodificar = ['Pais', 'Regiao', 'Tipo_Cultura', 'Estrategias_Adaptacao']
        for coluna in colunas_para_decodificar:
            self.df[coluna] = self.df[coluna].apply(lambda v: v.decode('utf-8') if isinstance(v, bytes) else v)

        print("✅ Limpeza e renomeação concluídas.")
        return self.df
    