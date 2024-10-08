import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys

def preprocess_data(data_path, X_train_path, X_test_path, y_train_path, y_test_path):

    
    df = pd.read_csv(data_path, sep=';', engine='python')
    print("Datos cargados. Primeras filas:")
    print(df.head())  # Muestra las primeras filas del DataFrame

    # Diccionario para decodificar las horas
    HOURS = {
        0: "7:00", 1: "7:30", 2: "8:00", 3: "8:30", 4: "9:00", 5: "9:30", 6: "10:00", 
        7: "10:30", 8: "11:00", 9: "11:30", 10: "12:00", 11: "12:30", 12: "13:00", 
        13: "13:30", 14: "14:00", 15: "14:30", 16: "15:00", 17: "15:30", 18: "16:00", 
        19: "16:30", 20: "17:00", 21: "17:30", 22: "18:00", 23: "18:30", 24: "19:00",
        25: "19:30", 26: "20:00"
    }
    
     
    df['Hour_decoded'] = df['Hour (Coded)'].map(HOURS)


    if  df['Hour_decoded'].isnull().any():
        print("Advertencia: Algunos valores en 'Hour (Coded)' no se encontraron en el diccionario HOURS.")
        print("Valores NaN en 'Hour_decoded':", df['Hour_decoded'][df['Hour_decoded'].isnull()])

    # Convertir horas decodificadas a número 
    df['Hour_decoded'] = df['Hour_decoded'].str.split(':').str[0].astype(float).fillna(-1).astype(int)

      
   
    df.rename(columns={'Slowness in traffic (%),': 'Slowness in traffic (%)'}, inplace=True)

    # Reemplazar delimitador de variable objetivo por . para convertir a tipo numérico
    df['Slowness in traffic (%)'] = df['Slowness in traffic (%)'].str.replace(',', '.')

    # Convertir la variable objetivo a tipo numérico
    df['Slowness in traffic (%)'] = pd.to_numeric(df['Slowness in traffic (%)'], errors='coerce')

    # Comprobar si hay valores Nulos en la variable objetivo
    if df['Slowness in traffic (%)'].isnull().any():
        print("Advertencia: Hay valores NaN en 'Slowness in traffic (%)'. Se rellenarán con la media.")
    
    # Rellenar valores faltantes
    df.fillna(df.mean(), inplace=True)

    # Drop de la variable objetivo
    X = df.drop(['Slowness in traffic (%)'], axis=1, errors='ignore') 
    y = df['Slowness in traffic (%)']
    
    print("Características (X):")
    print(X.head())  # Muestra las primeras filas de X
    print("Tipos de datos en características (X):")
    print(X.dtypes)  # Muestra los tipos de datos en las características

    # División de los datos en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Escalar las características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    
    # Guardar los datos procesados en archivos CSV
    pd.DataFrame(X_train_scaled).to_csv(X_train_path, index=False)
    pd.DataFrame(X_test_scaled).to_csv(X_test_path, index=False)
    pd.DataFrame(y_train).to_csv(y_train_path, index=False)
    pd.DataFrame(y_test).to_csv(y_test_path, index=False)

if __name__ == "__main__":
    # Obtener las rutas de los archivos de entrada y salida
    data_path = sys.argv[1]
    X_train_path = sys.argv[2]
    X_test_path = sys.argv[3]
    y_train_path = sys.argv[4]
    y_test_path = sys.argv[5]
    
    # Llama a la función de procesamiento
    preprocess_data(data_path, X_train_path, X_test_path, y_train_path, y_test_path)
