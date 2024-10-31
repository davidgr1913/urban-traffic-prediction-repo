import pytest
import pandas as pd
from src.data.process import DataFrameProcessor
import yaml
import os

# Crear un archivo de configuración temporal para las pruebas
config_data = {
    'base': {'log_level': 'INFO'},
    'data_preprocess': {'data_preprocessed_path': 'tests/test_data_preprocessed.csv'},
    'data_process': {'fill_na_method': 'mean'}
}

with open("tests/test_config.yaml", "w") as f:
    yaml.dump(config_data, f)

@pytest.fixture
def processor():
    # Crear datos de prueba con valores faltantes y guardarlos en un CSV temporal
    data = {
        'slowness_in_traffic': [None, 5.3, 7.8, None, 6.2, None],
        'hour_decoded': [7, 8, 9, 10, 11, 12]
    }
    df = pd.DataFrame(data)
    df.to_csv(config_data['data_preprocess']['data_preprocessed_path'], index=False, sep=';')
    
    processor = DataFrameProcessor("tests/test_config.yaml")
    yield processor
    
    # Eliminar archivos después de la prueba
    if os.path.exists(config_data['data_preprocess']['data_preprocessed_path']):
        os.remove(config_data['data_preprocess']['data_preprocessed_path'])
    if os.path.exists("tests/test_config.yaml"):
        os.remove("tests/test_config.yaml")

def test_fill_missing_values(processor):
    processor.fill_missing_values()
    # Verifica que no haya valores nulos después del procesamiento
    assert not processor.df['slowness_in_traffic'].isnull().any()
