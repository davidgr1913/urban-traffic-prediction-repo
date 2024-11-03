import pandas as pd
import argparse
import yaml

from src.utils.logs import get_logger

days_to_code = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5}

class DataFrameProcessor:
    """
    A class used to preprocess a DataFrame based on a configuration file.
    Attributes
    ----------
    config : dict
        Configuration loaded from the provided YAML file.
    df : pandas.DataFrame
        DataFrame loaded from the CSV file specified in the configuration.
    fill_na_method : str
        Method to fill missing values ('mean', 'median', 'mode').
    logger : logging.Logger
        Logger instance for logging messages.
    Methods
    -------
    fill_missing_values():
        Fills missing values in the DataFrame based on the specified method.
    transform_days():
        Transforms day names to numerical values based on the configuration.
    export_data():
        Exports the processed DataFrame to a CSV file.
    full_process():
        Executes the full data processing pipeline: filling missing values, transforming days, and exporting data.
    """

    def __init__(self, config_path):

        self.config = yaml.safe_load(open(config_path))
        self.df = pd.read_csv(self.config['data_preprocess']['data_preprocessed_path'], sep=';')
        self.fill_na_method = self.config['data_process']['fill_na_method']
        self.logger = get_logger('DATA_PROCESS', log_level=self.config['base']['log_level'])

    

    def fill_missing_values(self):
        """
        Fills missing values in the 'slowness_in_traffic' column of the dataframe using the specified method.

        This method checks for NaN values in the 'slowness_in_traffic' column and fills them using the method specified 
        by the 'fill_na_method' attribute. The available methods are 'mean', 'median', and 'mode'.

        Logging:
            - Logs an info message indicating the start of the missing values check.
            - Logs a warning message if NaN values are found in the 'slowness_in_traffic' column.
            - Logs an info message indicating the method used to fill NaN values.

        Attributes:
            fill_na_method (str): The method to use for filling NaN values. Can be 'mean', 'median', or 'mode'.
            df (pandas.DataFrame): The dataframe containing the 'slowness_in_traffic' column.
            logger (logging.Logger): The logger instance for logging messages.

        Raises:
            ValueError: If 'fill_na_method' is not one of 'mean', 'median', or 'mode'.
        """
        self.logger.info("Checking for missing values")
        if self.df['slowness_in_traffic'].isnull().any():
            self.logger.warning(f"There are NaN values in 'Slowness in traffic (%)'. They will be filled with the {self.fill_na_method }")
             # Rellenar valores faltantes
            self.logger.info(f"Filling NaN values with {self.fill_na_method}")
            if self.fill_na_method == 'mean':
                self.df.fillna(self.df.mean(), inplace=True)
            elif self.fill_na_method == 'median':
                self.df.fillna(self.df.median(), inplace=True)
            elif self.fill_na_method == 'mode':
                self.df.fillna(self.df.mode().iloc[0], inplace=True)

    def transform_days(self):
        """
        Transforms the 'day' column in the dataframe from day names to numerical values.
        This method first assigns day names to the 'day' column based on the index positions
        specified in the configuration. Then, it converts these day names to numerical values
        using a predefined mapping (days_to_code).
        Raises:
            AssertionError: If the dataframe does not contain all the day names before converting
                            them to numerical values.
        Notes:
            Ensure that the dataframe is first transformed to contain day names before applying
            the numerical transformation.
        """
        self.logger.info("Transforming days to numerical values")
    
        self.df['day'] = '0'
        
        for idx in self.df.index:
            if idx <= self.config['data_process']['monday_position']:
                self.df.loc[idx, 'day'] = 'Monday'
            elif idx <= self.config['data_process']['tuesday_position']:
                self.df.loc[idx, 'day'] = 'Tuesday'
            elif idx <= self.config['data_process']['wednesday_position']:
                self.df.loc[idx, 'day'] = 'Wednesday'
            elif idx <= self.config['data_process']['thursday_position']:
                self.df.loc[idx, 'day'] = 'Thursday'
            elif idx <= self.config['data_process']['friday_position']:
                self.df.loc[idx, 'day'] = 'Friday'

        df_values = self.df["day"].unique()
        for key, value in days_to_code.items():
            assert key in df_values, "First transform your data into weekday by setting to_numerical=False, then apply the numerical transformation"
            self.df.loc[(self.df.day == key), 'day'] = value
        self.df['day'] = self.df['day'].astype(int)
    
    
    
    def export_data(self):
        self.df.to_csv(self.config['data_process']['data_processed_path'], index=False, sep=';')
        self.logger.info(f"Data exported to {self.config['data_process']['data_processed_path']}")

    def full_process(self):
        self.fill_missing_values()
        self.transform_days()
        self.export_data()
    


    



if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config", type=str, required=True,dest='config', help="Path to the configuration file")
    args = arg_parser.parse_args()
    processor = DataFrameProcessor(args.config)
    processor.full_process()

    

  