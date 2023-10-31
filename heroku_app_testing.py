import requests

url = 'https://ml-deployment-fastapi-a1a7a9f97a62.herokuapp.com/predict/'

data = {'age': 24,
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
    'native-country': 'united-states'}

# POST request
response = requests.post(url, json=data)
print("Status Code:", response.status_code)
print("Response Body:", response.json())