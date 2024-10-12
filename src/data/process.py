import pandas as pd
import argparse
import yaml

from src.utils.logs import get_logger

days_to_code = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5}

class DataFrameProcessor:

    def __init__(self, config_path):

        self.config = yaml.safe_load(open(config_path))
        self.df = pd.read_csv(self.config['data_preprocess']['data_preprocessed_path'], sep=';')
        self.fill_na_method = self.config['data_process']['fill_na_method']
        self.logger = get_logger('DATA_PROCESS', log_level=self.config['base']['log_level'])

    

    def fill_missing_values(self):
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

    

  