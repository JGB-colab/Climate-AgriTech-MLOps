# model/model.py

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import xgboost as xgb

class ModelPipeline:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.preprocessor = None

    def dividir_dados(self):
        """Separa o DataFrame em features e alvo, e depois em treino, validação e teste."""
        print("--- Iniciando Tarefa: Divisão dos Dados ---")
        X = self.df.drop('Produtividade_t_ha', axis=1)
        y = self.df['Produtividade_t_ha']
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.X_train, self.X_val, y_train, self.y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
        self.y_train = y_train.to_frame()
        self.y_val = self.y_val.to_frame()
        self.y_test = self.y_test.to_frame()

    def criar_e_aplicar_preprocessador(self):
        """Cria, treina e aplica o pré-processador aos dados."""
        print("--- Iniciando Tarefa: Pré-processamento ---")
        features_numericas = self.X_train.select_dtypes(include=np.number).columns.tolist()
        features_categoricas = self.X_train.select_dtypes(exclude=np.number).columns.tolist()

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), features_numericas),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), features_categoricas)
            ],
            remainder='passthrough'
        )
        self.preprocessor.fit(self.X_train)
        
        # Transforma os dados
        self.X_train = pd.DataFrame(self.preprocessor.transform(self.X_train), columns=self.preprocessor.get_feature_names_out())
        self.X_val = pd.DataFrame(self.preprocessor.transform(self.X_val), columns=self.preprocessor.get_feature_names_out())
        self.X_test = pd.DataFrame(self.preprocessor.transform(self.X_test), columns=self.preprocessor.get_feature_names_out())
        print("✅ Dados normalizados e codificados.")

    def treinar_modelo_random_forest(self) -> dict:
        """Treina, avalia e retorna as informações do modelo RandomForest."""
        print("--- Iniciando Treinamento: RandomForest ---")
        param_dist = {
            'n_estimators': [100, 200], 'max_depth': [10, 20], 'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2], 'max_features': ['sqrt', 'log2']
        }
        search = RandomizedSearchCV(RandomForestRegressor(random_state=42), param_dist, n_iter=5, cv=3, random_state=42)
        search.fit(self.X_train, self.y_train.values.ravel())
        
        modelo_rf = search.best_estimator_
        y_pred = modelo_rf.predict(self.X_test)
        r2 = r2_score(self.y_test, y_pred)
        print(f"RandomForest - R² no teste: {r2:.4f}")
        
        return {'model': modelo_rf, 'r2_score': r2, 'model_name': 'RandomForest'}

    def treinar_modelo_xgboost(self) -> dict:
        """Treina, avalia e retorna as informações do modelo XGBoost."""
        print("--- Iniciando Treinamento: XGBoost ---")
        param_dist = {'n_estimators': [100, 300], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1]}
        search = RandomizedSearchCV(xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1),
                                    param_dist, n_iter=5, cv=KFold(n_splits=3, shuffle=True, random_state=42))
        search.fit(self.X_train, self.y_train, eval_set=[(self.X_val, self.y_val)], verbose=False)
        
        modelo_xgb = search.best_estimator_
        y_pred = modelo_xgb.predict(self.X_test)
        r2 = r2_score(self.y_test, y_pred)
        print(f"XGBoost - R² no teste: {r2:.4f}")
        
        return {'model': modelo_xgb, 'r2_score': r2, 'model_name': 'XGBoost'}