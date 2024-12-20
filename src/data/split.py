import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Text
import yaml

from src.utils.logs import get_logger


def data_split(config_path: Text) -> None:
    """
    Splits the dataset into training and testing sets based on the configuration provided.
    Args:
        config_path (Text): Path to the configuration file in YAML format.
    Returns:
        None
    The function performs the following steps:
    1. Reads the configuration file to load the settings.
    2. Initializes a logger with the specified log level.
    3. Loads the processed dataset from the path specified in the configuration.
    4. Splits the dataset into training and testing sets based on the test size, random state, and stratification settings from the configuration.
    5. Saves the training and testing sets to the paths specified in the configuration.
    """


    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger('DATA_SPLIT', log_level=config['base']['log_level'])

    dataset = pd.read_csv(config['data_process']['data_processed_path'],sep=';')
    logger.info('Split features into train and test sets')
    train_dataset, test_dataset = train_test_split(
        dataset,
        test_size=config['data_split']['test_size'],
        random_state=config['base']['random_state'],
        stratify=dataset[config['data_split']['stratify']]
    )

    logger.info('Save train and test sets')
    train_csv_path = config['data_split']['data_train_path']
    test_csv_path = config['data_split']['data_test_path']
    train_dataset.to_csv(train_csv_path, index=False,sep=';')
    test_dataset.to_csv(test_csv_path, index=False,sep=';')


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    data_split(config_path=args.config)