import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import logging
import argparse
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import ( StandardScaler)

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import (   GridSearchCV

)
from sklearn.metrics import ( mean_squared_error, 
    mean_absolute_error, r2_score
)


import joblib

# Modelos
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR

# Importación de un logger personalizado
from src.utils.logs import get_logger


class ModelTrainer:
    def __init__(self, config_path):
        self.model_dict = {
            'LR': LinearRegression(),
            'kNN': KNeighborsRegressor(),
            'DTree': DecisionTreeRegressor(),
            'RF': RandomForestRegressor(),
            'XGBoost': XGBRegressor(),
            'SVR': SVR()
        }
        self.config = yaml.safe_load(open(config_path))
        self.logger = get_logger('MODEL_TRAINER', log_level=self.config['base']['log_level'])
        self.logger.info('Initialized ModelTrainer')
        self.param_grids = self.config['model']['model_params']  # Leer hiperparámetros del YAML



    def get_data(self):
        self.logger.info('Loading datasets...')
        df_train = pd.read_csv(self.config['data_split']['data_train_path'], sep=';')
        self.Xtrain = df_train.drop(self.config['model']['target'], axis=1)
        self.ytrain = df_train[self.config['model']['target']]
        self.logger.info('Datasets loaded successfully.')

    def scale_data(self):
        self.logger.info('Scaling data...')
        scaler = StandardScaler()
        self.Xtrain_scaled = scaler.fit_transform(self.Xtrain)
        self.logger.info('Data scaling completed.')

    def train_models(self):
        best_model_name = None
        best_model = None
        best_score = float('inf')

        for name, model in self.model_dict.items():
            self.logger.info(f'Evaluating model: {name}')
            
            # Crear un pipeline que incluye escalado y el modelo actual
            pipeline = Pipeline(steps=[('scaler', StandardScaler()), ('model', model)])

            # Obtener el grid de hiperparámetros desde el YAML
            param_grid = self.param_grids.get(name, {})
            
            # Usar GridSearchCV para encontrar los mejores hiperparámetros usando validación cruzada
            grid_search = GridSearchCV(pipeline, param_grid, cv=self.config['model']['n_splits'], 
                                    scoring=self.config['model']['optmise_metric'])
            
            # Entrenar usando GridSearchCV
            grid_search.fit(self.Xtrain, self.ytrain)

            # Obtener el mejor modelo resultante de la búsqueda en grid
            best_model_temp = grid_search.best_estimator_

            # Validación cruzada para calcular predicciones en los datos de entrenamiento
            y_pred_cv = cross_val_predict(best_model_temp, self.Xtrain, self.ytrain, cv=self.config['model']['n_splits'])

            # Calcular métricas de rendimiento basadas en las predicciones de la validación cruzada
            rmse_cv = np.sqrt(mean_squared_error(self.ytrain, y_pred_cv))
            mae_cv = mean_absolute_error(self.ytrain, y_pred_cv)
            r2_cv = r2_score(self.ytrain, y_pred_cv)

            self.logger.info(f'Model: {name}')
            self.logger.info(f'Cross-Validation RMSE: {rmse_cv:.3f}')
            self.logger.info(f'Cross-Validation MAE: {mae_cv:.3f}')
            self.logger.info(f'Cross-Validation R2: {r2_cv:.3f}')

            # Si el modelo tiene el mejor RMSE en validación cruzada, guardarlo como el mejor modelo
            if rmse_cv < best_score:
                best_score = rmse_cv
                best_model_name = name
                best_model = best_model_temp

        # Log del mejor modelo
        self.logger.info(f'Best Model: {best_model_name} with Cross-Validation RMSE: {best_score:.3f}')

        # Guardar el mejor modelo en un archivo
        joblib.dump(best_model, self.config['model']['output_model_path'])

        return best_model_name, best_model
        

    def run(self):
        self.get_data()
        self.scale_data()
        best_model_name, best_model = self.train_models()
        self.logger.info('Model training completed.')
        self.logger.info('Best model saved successfully.')
        
        

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    model_evaluator = ModelTrainer(config_path=args.config)
    model_evaluator.run()