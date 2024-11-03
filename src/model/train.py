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
    """
    A class used to train various machine learning models for urban traffic prediction.
    Attributes
    ----------
    model_dict : dict
        A dictionary containing model names as keys and their corresponding scikit-learn model instances as values.
    config : dict
        Configuration parameters loaded from a YAML file.
    logger : logging.Logger
        Logger instance for logging messages.
    param_grids : dict
        Hyperparameter grids for each model loaded from the configuration file.
    Xtrain : pd.DataFrame
        Training features.
    ytrain : pd.Series
        Training target variable.
    Xtrain_scaled : np.ndarray
        Scaled training features.
    Methods
    -------
    get_data():
        Loads the training dataset from the specified path in the configuration file.
    scale_data():
        Scales the training features using StandardScaler.
    train_models():
        Trains multiple models using GridSearchCV to find the best hyperparameters and evaluates them using cross-validation.
    run():
        Executes the full training pipeline: loading data, scaling data, training models, and saving the best model.
    """
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
        """
        Loads the training dataset from the specified file path in the configuration,
        splits it into features and target variables, and assigns them to instance variables.

        The method performs the following steps:
        1. Logs the start of the dataset loading process.
        2. Reads the training dataset from the CSV file specified in the configuration.
        3. Splits the dataset into features (Xtrain) and target (ytrain) based on the configuration.
        4. Logs the successful loading of the datasets.

        Raises:
            FileNotFoundError: If the specified file path does not exist.
            KeyError: If the specified target column is not found in the dataset.

        Returns:
            None
        """
        self.logger.info('Loading datasets...')
        df_train = pd.read_csv(self.config['data_split']['data_train_path'], sep=';')
        self.Xtrain = df_train.drop(self.config['model']['target'], axis=1)
        self.ytrain = df_train[self.config['model']['target']]
        self.logger.info('Datasets loaded successfully.')

    def scale_data(self):
        """
        Scales the training data using StandardScaler.

        This method scales the training data stored in `self.Xtrain` using
        the `StandardScaler` from scikit-learn. The scaled data is stored
        in `self.Xtrain_scaled`. Logs the start and completion of the scaling process.
        """
        self.logger.info('Scaling data...')
        scaler = StandardScaler()
        self.Xtrain_scaled = scaler.fit_transform(self.Xtrain)
        self.logger.info('Data scaling completed.')

    def train_models(self):
        """
        Trains multiple models using cross-validation and hyperparameter tuning, and selects the best model based on RMSE.
        This method iterates over a dictionary of models, performs hyperparameter tuning using GridSearchCV, 
        evaluates each model using cross-validation, and selects the model with the best RMSE score. 
        The best model is then saved to a file.
        Returns:
            tuple: A tuple containing the name of the best model and the best model object.
        Attributes:
            best_model_name (str): The name of the best performing model.
            best_model (object): The best performing model object.
            best_score (float): The best RMSE score obtained during cross-validation.
        Raises:
            KeyError: If the model name is not found in the parameter grid dictionary.
            ValueError: If the scoring metric specified in the configuration is not valid.
        Logs:
            Logs information about the evaluation of each model, including RMSE, MAE, and R2 scores.
            Logs the name and RMSE of the best model.
        Saves:
            The best model to the path specified in the configuration file.
        """
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