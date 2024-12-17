import joblib
import pandas as pd
import os

def test_model_load():
    """
    Test if the model file loads successfully.
    """
    # Get the model version from environment variables
    model_version = os.getenv("MODEL_VERSION", "v1")  # Default to v1 if not provided
    model_path = f"{model_version}/model_{model_version}.pkl"
    print(f"DEBUG: Testing model load with path: {model_path}")

    try:
        model = joblib.load(model_path)
    except FileNotFoundError as e:
        assert False, f"Model file not found: {e}"
    except Exception as e:
        assert False, f"Model failed to load due to an unexpected error: {e}"

def test_valid_predictions():
    """
    Test if the model produces valid predictions for given input data.
    """
    # Get the model version from environment variables
    model_version = os.getenv("MODEL_VERSION", "v1")  # Default to v1 if not provided
    model_path = f"{model_version}/model_{model_version}.pkl"
    print(f"DEBUG: Testing predictions with model path: {model_path}")

    # Load the model
    model = joblib.load(model_path)

    # Input data
    test_data = pd.DataFrame({
        "G1": [9, 13, 12],
        "studytime": [2, 2, 4],
        "famsup_no": [False, True, True],
        "famsup_yes": [True, False, False]
    })

    predictions = model.predict(test_data)

    # Check that the number of predictions matches the input size
    assert len(predictions) == len(test_data), "Prediction count mismatch!"

    # Check that predictions are non-negative
    assert all(pred >= 0 for pred in predictions), "Predictions should be non-negative!"
