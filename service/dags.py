from airflow import DAG
from airflow.operators import PythonOperator
from datetime import datetime
from data.pipeline import ETL


def call_extract():
    ETL.load_data()

def call_import():
    ETL.transform_data()

with DAG(
    dag_id='pipline_ws',
    start_date= datetime(2025,7,22), # Dia da criação da DAG
    schedule_interval="@hourly", # intervalo definido por sintaxe de crobtab
    catchup= False
) as dag:
    extract_transform = PythonOperator(
        task_id = 'extracao_transformacao',
        python_callable = call_extract
    )
    loading = PythonOperator(
        task_id = 'carregamento',
        python_callable = call_import
    )
    extract_transform >> loading