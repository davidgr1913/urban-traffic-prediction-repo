import yaml
import pandas as pd
import argparse

from src.utils.logs import get_logger
from src.utils.general import to_snake_case


# Diccionario para decodificar las horas
HOURS = {
    1: "7:00", 2: "7:30", 3: "8:00", 4: "8:30", 5: "9:00", 6: "9:30", 7: "10:00", 
    8: "10:30", 9: "11:00", 10: "11:30", 11: "12:00", 12: "12:30", 13: "13:00", 
    14: "13:30", 15: "14:00", 16: "14:30", 17: "15:00", 18: "15:30", 19: "16:00", 
    20: "16:30", 21: "17:00", 22: "17:30", 23: "18:00", 24: "18:30", 25: "19:00",
    26: "19:30", 27: "20:00"
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

  
