
import pandas as pd
from src.utils.general import to_snake_case
import yaml

# Diccionario para decodificar las horas
HOURS = {
    1: "7:00", 2: "7:30", 3: "8:00", 4: "8:30", 5: "9:00", 6: "9:30", 7: "10:00", 
    8: "10:30", 9: "11:00", 10: "11:30", 11: "12:00", 12: "12:30", 13: "13:00", 
    14: "13:30", 15: "14:00", 16: "14:30", 17: "15:00", 18: "15:30", 19: "16:00", 
    20: "16:30", 21: "17:00", 22: "17:30", 23: "18:00", 24: "18:30", 25: "19:00",
    26: "19:30", 27: "20:00"
}

days_to_code = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5}

def preprocess_data(df):

 
    df = to_snake_case(df)
    df['hour_decoded'] = df['hour_coded'].map(HOURS)

    # Convertir horas decodificadas a número 
    df['hour_decoded'] = df['hour_decoded'].str.split(':').str[0].astype(float).fillna(-1).astype(int)



    return df




class DataFrameProcessor:


    def __init__(self, df):

        self.df = df


    def transform_days(self):
 
    
        self.df['day'] = self.df['day'].map(days_to_code)
    
    
    


    def full_process(self):
        self.transform_days()
        return self.df  

    