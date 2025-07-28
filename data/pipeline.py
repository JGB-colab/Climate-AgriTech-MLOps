# data/pipeline.py
import mlcroissant as mlc
import pandas as pd
import shutil
import os

class ETL:
    def extrair_e_limpar_dados(self):
        print("--- Iniciando Tarefa: Extração e Limpeza (Otimizada) ---")

        cache_path = os.path.join(os.path.expanduser("~"), ".cache", "mlcroissant")
        if os.path.exists(cache_path):
            shutil.rmtree(cache_path)
            print(f"Cache em {cache_path} removido.")

        croissant_url = 'https://www.kaggle.com/datasets/waqi786/climate-change-impact-on-agriculture/croissant/download'
        croissant_dataset = mlc.Dataset(croissant_url)
        record_set_ref = croissant_dataset.metadata.record_sets[0]
        
        
        chunks_processados = []
        
        #  Quebra em chuncks
        chunk_size = 50000
        current_chunk = []

        print(f"Iniciando processamento em chunks de {chunk_size} linhas...")
        for i, record in enumerate(croissant_dataset.records(record_set=record_set_ref.uuid)):
            current_chunk.append(record)
            if (i + 1) % chunk_size == 0:
                print(f"Processando chunk até a linha {i+1}...")
                df_chunk = pd.DataFrame(current_chunk)
                df_chunk_limpo = self._limpar_chunk(df_chunk)
                chunks_processados.append(df_chunk_limpo)
                current_chunk = [] # Reseta o chunk atual

        # Processa o último chunk, se houver
        if current_chunk:
            print("Processando o chunk final...")
            df_chunk = pd.DataFrame(current_chunk)
            df_chunk_limpo = self._limpar_chunk(df_chunk)
            chunks_processados.append(df_chunk_limpo)

        print("✅ Todos os chunks foram processados. Concatenando...")
        
        # Concatena todos os chunks limpos em um DataFrame final
        df_final = pd.concat(chunks_processados, ignore_index=True)
        
        print("✅ Limpeza e renomeação concluídas.")
        return df_final

    def _limpar_chunk(self, df_chunk: pd.DataFrame) -> pd.DataFrame:
        """Função auxiliar para aplicar a limpeza em um único chunk."""
        # Limpeza de nomes de colunas
        df_chunk.columns = [col.split('/')[-1] for col in df_chunk.columns]

        # Renomeação de colunas
        nomes_novos = {
            'Average_Temperature_C': 'Temp_C', 'Total_Precipitation_mm': 'Chuva_mm',
            'Crop_Yield_MT_per_HA': 'Produtividade_t_ha', 'Extreme_Weather_Events': 'Eventos_Extremos',
            'Pesticide_Use_KG_per_HA': 'Uso_Pesticida_kg_ha', 'Fertilizer_Use_KG_per_HA': 'Uso_Fertilizante_kg_ha',
            'Year': 'Ano', 'Country': 'Pais', 'Region': 'Regiao', 'Crop_Type': 'Tipo_Cultura',
            'Irrigation_Access_%': 'Acesso_Irrigacao_Pct', 'Soil_Health_Index': 'Indice_Saude_Solo',
            'Adaptation_Strategies': 'Estrategias_Adaptacao', 'Economic_Impact_Million_USD': 'Impacto_Economico_USD_Milhoes'
        }
        df_chunk = df_chunk.rename(columns=nomes_novos)

        # Decodificação de valores bytes
        colunas_para_decodificar = ['Pais', 'Regiao', 'Tipo_Cultura', 'Estrategias_Adaptacao']
        for coluna in colunas_para_decodificar:
            if coluna in df_chunk.columns:
                df_chunk[coluna] = df_chunk[coluna].apply(lambda v: v.decode('utf-8') if isinstance(v, bytes) else v)
        
        return df_chunk