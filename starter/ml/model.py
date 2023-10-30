import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import fbeta_score, precision_score, recall_score
import joblib
from starter.ml.data import load_data, process_data

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """


    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save the trained model and encoders
    joblib.dump(model, 'model/trained_model.pkl')

    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    predictions = model.predict(X)

    return predictions


def evaluate_model_on_slices(model, df, X, y, slice_feature, encoder, lb):
    """
    Evaluate the model on slices of the data based on a categorical feature.

    Args:
    - model (sklearn model): Trained machine learning model.
    - df (DataFrame): Original DataFrame before processing.
    - X (np.array): Processed features of the dataset.
    - y (np.array): Processed target values.
    - slice_feature (str): Feature name on which to slice the data.
    - encoder (OneHotEncoder): The OneHotEncoder used in data processing.
    - lb (LabelBinarizer): The LabelBinarizer used in data processing.
    """

    categories = df[slice_feature].unique()

    for category in categories:
        # Create mask for the category
        mask = df[slice_feature] == category

        # Apply mask to X and y
        X_slice = X[mask]
        y_slice = y[mask]

        # Check if there are samples in the slice
        if len(X_slice) == 0:
            print(f"No samples for category '{category}' in {slice_feature}.")
            continue

        # Make predictions
        predictions = model.predict(X_slice)

        # Assuming y_slice needs to be binarized based on the unique values
        # This step depends on the format of y; adjust as needed.
        y_slice_binarized = lb.transform(y_slice)

        # Calculate metrics using compute_model_metrics
        precision, recall, f1 = compute_model_metrics(y_slice_binarized, predictions)

        # Calculate accuracy separately as it's not included in compute_model_metrics
        accuracy = accuracy_score(y_slice_binarized, predictions)

        print(f"Metrics for {slice_feature} = {category}:")
        print(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}\n")


if __name__ == '__main__':
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
        test, categorical_features=cat_features,encoder=encoder, label="salary", training=False
    )

    model = train_model(X_train, y_train)
    preds = inference(model,X_test)


    precision, recall, fbeta = compute_model_metrics(y_test.map({'<=50k': 0, '>50k':1 }),preds)
    print('precision: ', precision)
    print('recall: ', recall)
    print('fbeta: ', fbeta)

    evaluate_model_on_slices(model, test, X_test, y_test, 'salary', encoder, lb)



