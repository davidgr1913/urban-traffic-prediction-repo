import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import yaml

from src.utils.logs import get_logger

class DataFrameAnalyzer:

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



  