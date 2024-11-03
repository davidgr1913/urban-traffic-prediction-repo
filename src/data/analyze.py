import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import yaml

from src.utils.logs import get_logger

class DataFrameAnalyzer:
    """
    DataFrameAnalyzer is a class for performing various data analysis tasks on a pandas DataFrame.
    Attributes:
        config (dict): Configuration loaded from a YAML file.
        df (pd.DataFrame): DataFrame loaded from a CSV file specified in the configuration.
        logger (Logger): Logger instance for logging messages.
    Methods:
        show_info():
            Mostrar información general del DataFrame.
            Returns:
                pd.DataFrame: The DataFrame with general information printed.
        descriptive_statistics():
            Mostrar estadísticas descriptivas.
            Returns:
                pd.DataFrame: Descriptive statistics of the DataFrame.
        identify_null_values():
            Identificar y mostrar los valores nulos en cada columna.
            Prints the number of null values in each column.
        plot_target_distribution():
            Mostrar la distribución de la variable objetivo.
            Plots the distribution of the target variable 'slowness_in_traffic'.
        plot_numeric_distributions():
            Mostrar la distribución de todas las variables numéricas.
            Plots the distribution of all numeric variables in the DataFrame.
        correlation_heatmap():
            Generar un mapa de calor de las correlaciones entre variables numéricas.
            Plots a heatmap of the correlations between numeric variables in the DataFrame.
        full_analysis():
            Realizar un análisis completo del DataFrame.
            Performs a full analysis by calling all the above methods sequentially.
    """

    def __init__(self, config_path):

        self.config = yaml.safe_load(open(config_path))
        self.df = pd.read_csv(self.config['data_process']['data_processed_path'], sep=';')
        self.logger = get_logger('DATA_ANALYSIS', log_level=self.config['base']['log_level'])

    def show_info(self):
        """Mostrar información general del DataFrame."""
        print("\nInformación del dataset:")
        print(self.df.info())
        return self.df

    def descriptive_statistics(self):
        """Mostrar estadísticas descriptivas."""
        print("\nEstadística descriptiva:")
        return self.df.describe()

    def identify_null_values(self):
        """Identificar y mostrar los valores nulos en cada columna."""
        print("\nValores nulos por columna:")
        print(self.df.isnull().sum())


    def plot_target_distribution(self):
        """Mostrar la distribución de la variable objetivo."""
        plt.figure(figsize=(6, 3))
        sns.histplot(self.df['slowness_in_traffic'], kde=True)
        plt.title('Distribución de la variable objetivo: Slowness in traffic (%)')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()
        print('Se observa una distribución asimétrica hacia la derecha, teniendo la moda a 7.5%')

    def plot_numeric_distributions(self):
        """Mostrar la distribución de todas las variables numéricas."""
        self.df.hist(bins=50, figsize=(20, 15))
        plt.show()

    def correlation_heatmap(self):
        """Generar un mapa de calor de las correlaciones entre variables numéricas."""
        numeric_df = self.df.select_dtypes(include=[np.number])
        plt.figure(figsize=(10, 7))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
        plt.title('Mapa de correlación entre variables')
        plt.show()
        print('Se visualiza una correlación alta en la variable "Hour" y "Slowness in traffic".')
        print('Se visualiza una correlación alta en "Semaphore off" y "lack of electricity".')
    
    def full_analysis(self):
        """Realizar un análisis completo del DataFrame."""
        self.show_info()
        self.descriptive_statistics()
        self.identify_null_values()
        self.plot_target_distribution()
        self.plot_numeric_distributions()
        self.correlation_heatmap()





if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config", type=str, required=True,dest='config', help="Path to the configuration file")
    args = arg_parser.parse_args()
    processor = DataFrameAnalyzer(args.config)
    processor.full_analysis()



  