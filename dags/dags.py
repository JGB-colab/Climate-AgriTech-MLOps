# dags/dags.py

from __future__ import annotations
import pendulum
from airflow.models.dag import DAG
from airflow.operators.python import PythonOperator

import pandas as pd
from pathlib import Path
import joblib
import shutil


def _get_run_dir(ti) -> Path:
    """Cria e retorna um diretório temporário único para esta execução da DAG."""
    run_dir = Path(f"/tmp/airflow_runs/{ti.run_id}")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def etl_and_preprocess(**kwargs):
    """
    Tarefa 1: Executa o ETL, carrega os dados e os pré-processa,
    salvando o resultado e o pré-processador para as tarefas seguintes.
    """
    from data.pipeline import ETL
    from model.model import ModelPipeline
    
    # Executa o ETL
    etl_instance = ETL()
    df = etl_instance.extrair_e_limpar_dados()

    # Instancia o pipeline de modelo e executa os primeiros passos
    model_pipeline = ModelPipeline(df)
    model_pipeline.dividir_dados()
    model_pipeline.criar_e_aplicar_preprocessador()
    
    # Salva os artefatos em um diretório temporário
    run_dir = _get_run_dir(kwargs['ti'])
    joblib.dump(model_pipeline, run_dir / 'model_pipeline_preprocessed.joblib')
    
    # Passa o caminho do diretório via XCom
    kwargs['ti'].xcom_push(key='run_dir', value=str(run_dir))

def train_and_evaluate_model(model_trainer_method_name: str, **kwargs):
    """
    Função genérica para treinar um modelo.
    Recebe o nome do método de treinamento a ser chamado.
    """
    # Carrega o pipeline pré-processado
    ti = kwargs['ti']
    run_dir = Path(ti.xcom_pull(key='run_dir', task_ids='etl_and_preprocess_task'))
    model_pipeline = joblib.load(run_dir / 'model_pipeline_preprocessed.joblib')

    # Chama dinamicamente o método de treinamento (ex: 'treinar_modelo_random_forest')
    trainer_method = getattr(model_pipeline, model_trainer_method_name)
    model_info = trainer_method() # Executa o treinamento

    # Salva o modelo treinado
    model_filename = f"{model_info['model_name']}.joblib"
    joblib.dump(model_info['model'], run_dir / model_filename)
    
    # Retorna as métricas e o caminho para a próxima tarefa
    return {
        'model_name': model_info['model_name'],
        'r2_score': model_info['r2_score'],
        'model_path': str(run_dir / model_filename)
    }

def select_and_save_best_model(**kwargs):
    """
    Tarefa final: Compara os modelos e promove o melhor para o diretório de artefatos.
    """
    ti = kwargs['ti']
    # Puxa os resultados das tarefas de treinamento
    rf_info = ti.xcom_pull(task_ids='train_random_forest_task')
    xgb_info = ti.xcom_pull(task_ids='train_xgboost_task')

    # Compara os modelos com base na métrica R²
    best_model_info = rf_info if rf_info['r2_score'] >= xgb_info['r2_score'] else xgb_info
    print(f"🏆 Modelo Vencedor: {best_model_info['model_name']} com R² de {best_model_info['r2_score']:.4f}")

    # Carrega o pré-processador original
    run_dir = Path(ti.xcom_pull(key='run_dir', task_ids='etl_and_preprocess_task'))
    model_pipeline = joblib.load(run_dir / 'model_pipeline_preprocessed.joblib')
    preprocessor = model_pipeline.preprocessor

    # Define o diretório final para os artefatos
    final_artifacts_dir = Path(__file__).parent.parent / "artifacts"
    final_artifacts_dir.mkdir(exist_ok=True)

    # Salva o pré-processador
    joblib.dump(preprocessor, final_artifacts_dir / "preprocessor.joblib")
    
    # Copia o arquivo do melhor modelo para o diretório final com o nome padrão
    shutil.copyfile(best_model_info['model_path'], final_artifacts_dir / "modelo_final.joblib")
    
    print(f"Artefatos finais salvos em: {final_artifacts_dir}")
    # Opcional: Limpa o diretório temporário
    # shutil.rmtree(run_dir)


with DAG(
    dag_id='climate_agritech_pipeline_final',
    start_date=pendulum.datetime(2025, 7, 28, tz="UTC"),
    schedule="@daily",
    catchup=False,
    tags=['mlops', 'structured']
) as dag:

    etl_and_preprocess_task = PythonOperator(
        task_id='etl_and_preprocess_task',
        python_callable=etl_and_preprocess
    )

    train_random_forest_task = PythonOperator(
        task_id='train_random_forest_task',
        python_callable=train_and_evaluate_model,
        op_kwargs={'model_trainer_method_name': 'treinar_modelo_random_forest'}
    )
    
    train_xgboost_task = PythonOperator(
        task_id='train_xgboost_task',
        python_callable=train_and_evaluate_model,
        op_kwargs={'model_trainer_method_name': 'treinar_modelo_xgboost'}
    )
    
    select_best_model_task = PythonOperator(
        task_id='select_and_save_best_model_task',
        python_callable=select_and_save_best_model
    )

    etl_and_preprocess_task >> [train_random_forest_task, train_xgboost_task] >> select_best_model_task