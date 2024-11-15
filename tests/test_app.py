import pytest
from fastapi.testclient import TestClient
from app import app  # Importa la aplicación FastAPI

# Crea un cliente de prueba
client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Traffic prediction model API"}

def test_predict_endpoint():
    # Datos de ejemplo para enviar a /predict
    data = {
    "hour_coded": 1,
    "immobilized_bus": 0,
    "broken_truck": 0,
    "vehicle_excess": 0,
    "accident_victim": 0,
    "running_over": 0,
    "fire_vehicles": 0,
    "occurrence_involving_freight": 0,
    "incident_involving_dangerous_freight": 0,
    "lack_of_electricity": 0,
    "fire": 0,
    "point_of_flooding": 0,
    "manifestations": 0,
    "defect_in_the_network_of_trolleybuses": 0,
    "tree_on_the_road": 0,
    "semaphore_off": 0,
    "intermittent_semaphore": 0,
    "slowness_in_traffic": 4.1,
    "day": "Monday"  
}
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert "prediction" in response.json()
