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
    """
    A class used to evaluate a machine learning model.
    Attributes
    ----------
    config : dict
        Configuration parameters loaded from a YAML file.
    logger : logging.Logger
        Logger instance for logging information.
    Methods
    -------
    get_data():
        Loads the test dataset and prepares the features and target variables.
    evaluate_model():
        Loads the trained model, evaluates it on the test data, and logs the performance metrics.
    plot_prediction_quality(y_true, y_pred):
        Plots the comparison between true and predicted values and saves the plot.
    save_metrics_for_dvc(rmse, mae, r2):
        Saves the evaluation metrics (RMSE, MAE, R2) to a JSON file.
    save_predictions_to_csv(y_true, y_pred):
        Saves the true and predicted values to a CSV file.
    run():
        Executes the data loading and model evaluation process.
    """
    def __init__(self, config_path):
        self.config = yaml.safe_load(open(config_path))
        self.logger = get_logger('MODEL_EVALUATOR', log_level=self.config['base']['log_level'])
        self.logger.info('Initialized ModelEvaluator')


    def get_data(self):
        """
        Loads the test dataset from the specified file path in the configuration.

        This method reads the test dataset from a CSV file, splits it into features (Xtest) and target (ytest),
        and logs the process.

        Attributes:
            Xtest (pd.DataFrame): The features of the test dataset.
            ytest (pd.Series): The target variable of the test dataset.

        Raises:
            FileNotFoundError: If the test dataset file is not found at the specified path.
            KeyError: If the target column specified in the configuration is not found in the dataset.
        """
        self.logger.info('Loading test dataset...')
        df_test = pd.read_csv(self.config['data_split']['data_test_path'], sep=';')
        self.Xtest = df_test.drop(self.config['model']['target'], axis=1)
        self.ytest = df_test[self.config['model']['target']]
        self.logger.info('Test data loaded successfully.')


    def evaluate_model(self):
        """
        Evaluate the trained model on the test dataset.
        This method loads the best model from the specified path, evaluates it on the test data,
        and logs the performance metrics including RMSE, MAE, and R2 score. It also saves the 
        evaluation metrics for DVC tracking, saves the predictions to a CSV file, and plots the 
        prediction quality.
        Returns:
            None
        """
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
        self.save_predictions_to_csv(self.ytest, y_pred_test)
        self.plot_prediction_quality(self.ytest, y_pred_test)

    def plot_prediction_quality(self, y_true, y_pred):
        """
        Plots the quality of predictions by comparing true values with predicted values.
        This function creates a scatter plot of the true values versus the predicted values,
        along with a reference line (y=x) to visualize the accuracy of the predictions. The plot
        is saved to a specified directory.
        Args:
            y_true (array-like): Array of true values.
            y_pred (array-like): Array of predicted values.
        Returns:
            None
        """

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
        """
        Save evaluation metrics to a JSON file for DVC tracking.
        Args:
            rmse (float): Root Mean Squared Error of the model.
            mae (float): Mean Absolute Error of the model.
            r2 (float): R-squared value of the model.
        Saves:
            A JSON file containing the evaluation metrics at the path specified in the configuration.
        """
    
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

    def save_predictions_to_csv(self, y_true, y_pred):
        """
        Save the true and predicted values to a CSV file.
        Args:
            y_true (array-like): The true values.
            y_pred (array-like): The predicted values.
        Saves:
            A CSV file containing the true and predicted values at the path specified
            in the configuration under 'evaluate' -> 'output_predictions_csv_path'.
        Logs:
            A message indicating the file path where the true and predicted values were saved.
        """
        # Guardar valores reales y predichos en un archivo CSV
        predictions_df = pd.DataFrame({
            'True Values': y_true,
            'Predicted Values': y_pred
        })

        csv_path = os.path.join(self.config['evaluate']['output_predictions_csv_path'])
        predictions_df.to_csv(csv_path, index=False)
        self.logger.info(f'Valores reales y predichos guardados en {csv_path}')



    def run(self):
        self.get_data()
        self.evaluate_model()


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    evaluator = ModelEvaluator(config_path=args.config)
    evaluator.run()
