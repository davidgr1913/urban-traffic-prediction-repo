import yaml
import pandas as pd
import argparse

from src.utils.logs import get_logger
from src.utils.general import to_snake_case


# Diccionario para decodificar las horas
HOURS = {
    0: "7:00", 1: "7:30", 2: "8:00", 3: "8:30", 4: "9:00", 5: "9:30", 6: "10:00", 
    7: "10:30", 8: "11:00", 9: "11:30", 10: "12:00", 11: "12:30", 12: "13:00", 
    13: "13:30", 14: "14:00", 15: "14:30", 16: "15:00", 17: "15:30", 18: "16:00", 
    19: "16:30", 20: "17:00", 21: "17:30", 22: "18:00", 23: "18:30", 24: "19:00",
    25: "19:30", 26: "20:00"
}

def preprocess_data(config_path):

    config = yaml.safe_load(open(config_path))

    logger = get_logger('DATA_PREPROCESS', log_level=config['base']['log_level'])
    logger.info('Get dataset')

    df = pd.read_csv(config['data_preprocess']['data_raw_path'], sep=';', engine='python')

    logger.info('Preprocess dataset')
    df = to_snake_case(df)
    df['hour_decoded'] = df['hour_coded'].map(HOURS)

    # Convertir horas decodificadas a número 
    df['hour_decoded'] = df['hour_decoded'].str.split(':').str[0].astype(float).fillna(-1).astype(int)

    # Reemplazar delimitador de variable objetivo por . para convertir a tipo numérico
    df['slowness_in_traffic'] = df['slowness_in_traffic'].str.replace(',', '.')

    # Convertir la variable objetivo a tipo numérico
    df['slowness_in_traffic'] = pd.to_numeric(df['slowness_in_traffic'], errors='coerce')


    df.to_csv(config['data_preprocess']['data_preprocessed_path'], index=False, sep=';')

    logger.info('Preprocess Done')



if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config", type=str, required=True,dest='config', help="Path to the configuration file")
    args = arg_parser.parse_args()
    preprocess_data(args.config)

  
