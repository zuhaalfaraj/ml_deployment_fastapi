# Project Title: Machine Learning Model Deployment with FastAPI

## Overview
This project showcases the deployment of a machine learning model using FastAPI. It includes a REST API for making predictions using a pre-trained model, with support for both GET and POST requests. The GET request greets the user, and the POST request performs model inference based on the provided data.

## Features
- **FastAPI Framework**: Utilizes FastAPI for high performance and easy-to-create APIs with automatic interactive documentation.
- **Machine Learning Integration**: Includes a pre-trained RandomForestClassifier model to make predictions.
- **Input Validation**: Uses Pydantic models for request body validation and automatic documentation generation.
- **Example Input**: Provides an example payload for ease of use in testing the POST endpoint.

## Installation

### Prerequisites
- Python 3.8+
- pip

### Steps
1. **Clone the Repository**
   ```sh
   git clone https://github.com/zuhaalfaraj/ml_deployment_fastapi.git
   cd your-repo
   ```

2. **Set Up a Virtual Environment (Optional but Recommended)**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**
   ```sh
   pip install -r requirements.txt
   ```

4. **Run the Application**
   ```sh
   uvicorn main:app --reload
   ```

## Usage
Once the application is running, you can perform API requests either through the interactive documentation at `http://127.0.0.1:8000/docs` or using tools like `curl` or Postman.

### Endpoints
- `GET /`: Returns a welcome message.
- `POST /predict/`: Accepts a JSON payload to make predictions. The payload format is as per the `DataItem` Pydantic model.

### Example POST Request
```json
{
    "age": 30,
    "workclass": "Private",
    "fnlgt": 78934,
    "education": "Masters",
    "education-num": 14,
    "marital-status": "Never-married",
    "occupation": "Professional",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Female",
    "capital-gain": 10000,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
}
```

## Testing
- Run `pytest` to execute unit tests.
- Test coverage is provided for different API endpoints.

## Deployment
- This project is set up for deployment on Heroku with a Procfile and necessary configuration.

---
