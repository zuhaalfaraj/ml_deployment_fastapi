# Script to train machine learning model.

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from ml.data import process_data, load_data
from ml.model import train_model
import joblib

# Add the necessary imports for the starter code.

# Add code to load in the data.
pth = 'data/census_clean.csv'
data = load_data(pth)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, encoder=encoder, label="salary", training=False)


# Train and save a model.
model = train_model(X_train, y_train)