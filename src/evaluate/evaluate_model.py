import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import argparse
import os
import json
from src.utils.logs import get_logger

class ModelEvaluator:
    def __init__(self, config_path):
        self.config = yaml.safe_load(open(config_path))
        self.logger = get_logger('MODEL_EVALUATOR', log_level=self.config['base']['log_level'])
        self.logger.info('Initialized ModelEvaluator')


    def get_data(self):
        self.logger.info('Loading test dataset...')
        df_test = pd.read_csv(self.config['data_split']['data_test_path'], sep=';')
        self.Xtest = df_test.drop(self.config['model']['target'], axis=1)
        self.ytest = df_test[self.config['model']['target']]
        self.logger.info('Test data loaded successfully.')


    def evaluate_model(self):
        self.logger.info('Loading best model...')
        model = joblib.load(self.config['model']['output_model_path'])

        self.logger.info('Evaluating model on test data...')
        y_pred_test = model.predict(self.Xtest)
        rmse_test = np.sqrt(mean_squared_error(self.ytest, y_pred_test))
        mae_test = mean_absolute_error(self.ytest, y_pred_test)
        r2_test = r2_score(self.ytest, y_pred_test)

        self.logger.info(f'Test RMSE: {rmse_test:.3f}')
        self.logger.info(f'Test MAE: {mae_test:.3f}')
        self.logger.info(f'Test R2: {r2_test:.3f}')
        self.save_metrics_for_dvc(rmse_test, mae_test, r2_test)
        self.plot_prediction_quality(self.ytest, y_pred_test)

    def plot_prediction_quality(self, y_true, y_pred):

        plt.figure(figsize=(8, 6))
        plt.scatter(y_true, y_pred, alpha=0.6, color='blue', label='Predicciones')
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', lw=2, label='Referencia (y=x)')
        plt.xlabel('Valores Reales')
        plt.ylabel('Valores Predichos')
        plt.title('Comparación de Valores Reales vs Predichos')
        plt.legend()

        # Guardar el gráfico en la carpeta
        plot_path = os.path.join(self.config['evaluate']['output_fig_prediction_true_path'])
        plt.savefig(plot_path)
        self.logger.info(f'Gráfico de calidad de predicción guardado en {plot_path}')
        plt.close()


    def save_metrics_for_dvc(self, rmse, mae, r2):
    
        # Guardar métricas en un archivo JSON
        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }

        metrics_path = os.path.join(self.config['evaluate']['output_metrics_path'])
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)

        self.logger.info(f'Métricas guardadas en {metrics_path}')


    def run(self):
        self.get_data()
        self.evaluate_model()


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    evaluator = ModelEvaluator(config_path=args.config)
    evaluator.run()
