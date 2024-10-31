# tests/test_preprocess.py
import pandas as pd
from src.data.preprocess import HOURS

def test_preprocess_logic():
    # Verificar el diccionario HOURS
    assert len(HOURS) == 27, "El diccionario HOURS debería tener 27 valores"
    assert HOURS[1] == "7:00", "La hora decodificada de '1' debería ser '7:00'"
    assert HOURS[27] == "20:00", "La hora decodificada de '27' debería ser '20:00'"
    
    # Probar la conversión de 'slowness_in_traffic'
    data = {'slowness_in_traffic': ['1,5', '2,3', '0,9']}
    df = pd.DataFrame(data)
    df['slowness_in_traffic'] = df['slowness_in_traffic'].str.replace(',', '.').astype(float)
    
    assert df['slowness_in_traffic'].dtype == float, "La columna debería ser de tipo float"
    assert df['slowness_in_traffic'].iloc[0] == 1.5, "El primer valor debería ser 1.5"
    assert df['slowness_in_traffic'].iloc[1] == 2.3, "El segundo valor debería ser 2.3"
    assert df['slowness_in_traffic'].iloc[2] == 0.9, "El tercer valor debería ser 0.9"

