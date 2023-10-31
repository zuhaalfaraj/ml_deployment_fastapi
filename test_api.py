from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200

def test_predict_with_valid_data():
    response = client.post(
        "/predict/",
        json={'age': 24,
    'workclass': 'private',
    'fnlgt': 172496,
    'education': 'bachelors',
    'education-num': 13,
    'marital-status': 'never-married',
    'occupation': 'sales',
    'relationship': 'not-in-family',
    'race': 'white',
    'sex': 'male',
    'capital-gain': 0,
    'capital-loss': 0,
    'hours-per-week': 30,
    'native-country': 'united-states'},
    )
    assert response.status_code == 200


def test_predict_with_invalid_data():
    response = client.post(
        "/predict/",
        json={
            'age': "thirty",
            'relationship': 'not-in-family',
            'race_val': 'white',
            'sex_val': 'male',
            'capital': 0,
            'capital': 0,
        },
    )
    assert response.status_code != 200