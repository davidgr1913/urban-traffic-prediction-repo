import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import logging


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


class ModelEvaluator:
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
        self.logger = self.get_logger('MODEL_EVALUATOR', log_level=self.config['base']['log_level'])
        self.logger.info('Initialized ModelEvaluator')
        self.param_grids = self.config['model']['model_params']  # Leer hiperparámetros del YAML

    def get_logger(self, name, log_level=logging.INFO):
        logger = logging.getLogger(name)
        logger.setLevel(log_level)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def get_data(self):
        self.logger.info('Loading datasets...')
        df_train = pd.read_csv(self.config['data_split']['data_train_path'], sep=';')
        df_test = pd.read_csv(self.config['data_split']['data_test_path'], sep=';')
        self.Xtrain = df_train.drop(self.config['model']['target'], axis=1)
        self.ytrain = df_train[self.config['model']['target']]
        self.Xtest = df_test.drop(self.config['model']['target'], axis=1)
        self.ytest = df_test[self.config['model']['target']]
        self.logger.info('Datasets loaded successfully.')

    def scale_data(self):
        self.logger.info('Scaling data...')
        scaler = StandardScaler()
        self.Xtrain_scaled = scaler.fit_transform(self.Xtrain)
        self.Xtest_scaled = scaler.transform(self.Xtest)
        self.logger.info('Data scaling completed.')

    def evaluate_models(self):
        best_model_name = None
        best_model = None
        best_score = float('inf')

        for name, model in self.model_dict.items():
            self.logger.info(f'Evaluating model: {name}')
            pipeline = Pipeline(steps=[('scaler', StandardScaler()), ('model', model)])

            # Obtener el grid de hiperparámetros desde el YAML
            param_grid = self.param_grids.get(name, {})

            grid_search = GridSearchCV(pipeline, param_grid, cv=self.config['model']['n_splits'], 
                                       scoring=self.config['model']['optmise_metric'])
            grid_search.fit(self.Xtrain, self.ytrain)

            best_params = grid_search.best_params_
            best_model_temp = grid_search.best_estimator_

            # Evaluar en datos de entrenamiento
            y_pred_train = best_model_temp.predict(self.Xtrain)
            rmse_train = np.sqrt(mean_squared_error(self.ytrain, y_pred_train))
            mae_train = mean_absolute_error(self.ytrain, y_pred_train)
            r2_train = r2_score(self.ytrain, y_pred_train)

            # Evaluar en datos de prueba
            y_pred_test = best_model_temp.predict(self.Xtest)
            rmse_test = np.sqrt(mean_squared_error(self.ytest, y_pred_test))
            mae_test = mean_absolute_error(self.ytest, y_pred_test)
            r2_test = r2_score(self.ytest, y_pred_test)

            self.logger.info(f'Model: {name}')
            self.logger.info(f'Best Params: {best_params}')
            self.logger.info(f'Train RMSE: {rmse_train:.3f}, Test RMSE: {rmse_test:.3f}')
            self.logger.info(f'Train MAE: {mae_train:.3f}, Test MAE: {mae_test:.3f}')
            self.logger.info(f'Train R2: {r2_train:.3f}, Test R2: {r2_test:.3f}')


            if rmse_test < best_score:
                best_score = rmse_test
                best_model_name = name
                best_model = best_model_temp
                best_r2 = r2_test

                best_r2_train = r2_train
                best_rmse_train = rmse_train
        

        self.logger.info(f'Best Model: {best_model_name} with and R2: {best_r2:.3f} RMSE: {best_score:.3f} ')
        self.logger.info(f'Best Model Train R2: {best_r2_train:.3f} and RMSE: {best_rmse_train:.3f}')

        # exportar el mejor modelo

        joblib.dump(best_model, self.config['model']['output_model_path'])



        return best_model_name, best_model

    def run(self):
        self.get_data()
        self.scale_data()
        best_model_name, best_model = self.evaluate_models()
        return best_model_name, best_model

# Uso de la clase
# evaluator = ModelEvaluator('config.yaml')
# best_model_name, best_model = evaluator.run()
