from fastapi import FastAPI, HTTPException
import uvicorn
import json
from src.app import data_process
import pandas as pd
import joblib
# Load the saved model
with open("./models/best_model.pkl", "rb") as f:
    model = joblib.load(f)



# Initialize FastAPI app
app = FastAPI(title="Traffic Prediction API", description="API for traffic prediction model", version="1.0",
              description='''This is a simple API for traffic prediction model. It takes in 18 features and returns the predicted traffic volume.
              ### funcionalidades:
              # - **/predict**: endpoint para hacer predicciones
              # - **/**: endpoint para obtener un mensaje de bienvenida''',
              license_info={"name": "MIT", "url": "https://opensource.org/licenses/MIT"})

# Define prediction endpoint
@app.post("/predict")

def predict(data_json: dict):

    data_dict = data_json

    # Validate input length
    if len(data_dict) != 19:
        raise HTTPException(
            status_code=400,
            detail=f"Input must contain {model.n_features_in_} features."
        )
    # dict to dataframe
    df = pd.DataFrame(data_dict, index=[0])
    # Preprocess the input data
    df = data_process.preprocess_data(df)
    data_processor = data_process.DataFrameProcessor(df)
    df = data_processor.full_process()

    ## ordenar las columnas del dataframe en el mismo orden que las columnas del modelo

    df = df[model.feature_names_in_]

    # Make a prediction
    prediction = model.predict(df)[0]


    
    return {"prediction": prediction.item()}

# Define a root endpoint
@app.get("/")   
def read_root():
    return {"message": "Traffic prediction model API"}

# Run the server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000
    )
