# Climate-AgriTech-MLOps

Este projeto é a implementação de um pipeline de MLOps de ponta a ponta, desenvolvido para a disciplina de Engenharia de Sistemas Inteligentes. O objetivo é prever a produtividade agrícola (`Crop_Yield_MT_per_HA`) com base em variáveis climáticas, geográficas e de insumos agrícolas.

[cite_start]O pipeline é orquestrado com Apache Airflow e é dividido em três módulos principais, conforme a especificação do trabalho.

### [cite_start]1. Módulo de Pipeline de Dados [cite: 13]

Responsável pela extração, limpeza e preparação dos dados.

* **Extração:** Os dados são extraídos diretamente de um dataset do Kaggle sobre o impacto das mudanças climáticas na agricultura, utilizando a biblioteca `mlcroissant`. Essa abordagem garante que os dados sejam carregados seguindo um padrão de metadados padronizado.
* [cite_start]**Limpeza e Tratamento (Racional):** O processo de limpeza foi desenhado para ser eficiente e robusto[cite: 14]:
    * **Processamento em Chunks:** Para lidar com o volume de dados sem sobrecarregar a memória RAM, os registros são processados em lotes (chunks).
    * **Limpeza de Colunas:** Os nomes originais das colunas vinham com prefixos (`source/column_name`). O código os limpa para manter apenas o nome relevante.
    * **Renomeação:** As colunas foram renomeadas para o português e abreviadas (ex: `Average_Temperature_C` para `Temp_C`) para facilitar a manipulação e a legibilidade do código.
    * **Decodificação de Tipos:** Colunas categóricas como `Pais` e `Tipo_Cultura` estavam em formato de `bytes`. Elas foram decodificadas para o padrão `UTF-8` para se tornarem strings legíveis e compatíveis com as bibliotecas de pré-processamento.

### [cite_start]2. Módulo de Pipeline de Modelos [cite: 15]

Responsável por construir, treinar, comparar e salvar o modelo de aprendizagem.

* **Pré-processamento:**
    * Os dados são divididos em conjuntos de treino (60%), validação (20%) e teste (20%).
    * Um `ColumnTransformer` do Scikit-learn é utilizado para aplicar `StandardScaler` (padronização) em features numéricas e `OneHotEncoder` em features categóricas. O `handle_unknown='ignore'` no encoder torna o modelo robusto a novas categorias em produção.
* **Experimentação e Treinamento:**
    * Dois modelos de regressão são treinados em paralelo para competir pelo melhor desempenho: `RandomForestRegressor` e `XGBoost`.
    * A otimização de hiperparâmetros é feita de forma eficiente com `RandomizedSearchCV`, que testa combinações aleatórias de parâmetros.
    * A métrica de avaliação para comparar os modelos é o **R² Score**.
* **Seleção do Campeão e Geração de Binário:**
    * O pipeline, orquestrado pelo Airflow, compara o R² Score dos dois modelos.
    * [cite_start]O modelo com o maior R² (o "modelo campeão") e o pré-processador são salvos como arquivos binários (`.joblib`) no diretório `/artifacts`[cite: 15], prontos para serem usados em produção.

### [cite_start]3. Módulo de Serviço [cite: 17]

Disponibiliza o modelo treinado através de uma API.

* **API:** Foi desenvolvida uma API web usando **Flask**.
* **Endpoint `/prever`:**
    * Recebe dados de entrada via `POST` em formato JSON.
    * Carrega o `modelo_final.joblib` e o `preprocessor.joblib`.
    * Utiliza o pré-processador para transformar os dados de entrada no formato que o modelo espera.
    * Retorna a previsão de produtividade agrícola (`previsao_produtividade_t_ha`) também em formato JSON.
* [cite_start]**Demonstração:** O funcionamento pode ser ilustrado enviando uma requisição JSON para o endpoint `/prever` com os dados de uma safra para obter a previsão de sua produtividade[cite: 18].