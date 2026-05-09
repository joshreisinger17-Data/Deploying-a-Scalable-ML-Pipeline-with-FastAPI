import pytest
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model
from ml.model import compute_model_metrics
from ml.model import process_data

# Implement the first test.
def test_train_model():
    """
    Testing if train_model returns a RandomForestClassifier
    """
    X_train = pd.DataFrame({
        'age': [20, 30, 40, 50]
    })

    y_train = [0, 1, 0, 1]
    model = train_model(X_train, y_train)

    assert isinstance(model, RandomForestClassifier) 


# Implement the second test.
def test_compute_model_metrics():
    """
    Testing the correct metric values are returned.
    """
    y = [0, 1, 0, 1]
    preds = [1, 0, 0, 1]

    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)


# Implement the third test.
def test_process_data():
    """
    Testing the process data has the expected shape.
    """
    test_data = pd.DataFrame({'age': [20, 30, 40], 'workclass': ['Private',
        'Self-emp', 'Government'], 'salary': [0, 1, 1]
    })

    cat_features = ['workclass']
    X, y, encoder, lb = process_data(test_data, categorical_features=cat_features,
        label='salary', training=True
    )

    assert X.shape[0] == 3
    assert len(y) == 3
