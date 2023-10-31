from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message":"Welcome to the model inference API!"}

def test_predict_with_cls_1():
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
    assert response.json() == {"prediction": 1}


def test_predict_with_cls_0():
    response = client.post(
        "/predict/",
        json={'age': 43,
         'workclass': 'private',
         'fnlgt': 70055,
         'education': 'some-college',
         'education-num': 10,
         'marital-status': 'married-civ-spouse',
         'occupation': 'adm-clerical',
         'relationship': 'husband',
         'race': 'white',
         'sex': 'male',
         'capital-gain': 0,
         'capital-loss': 0,
         'hours-per-week': 40,
         'native-country': 'united-states'},
    )
    assert response.status_code == 200
    assert response.json() == {"prediction": 0}