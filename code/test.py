# test_serve.py
from fastapi.testclient import TestClient
from serve import app

client = TestClient(app)


def test_predict_veracity():
    response = client.post("/claim/v1/predict", json={"claim_text": "COVID-19 vaccines are safe."})

    # Ensure the request was successful
    assert response.status_code == 200

    # Ensure the response contains the predicted veracity
    json_response = response.json()
    assert "veracity" in json_response
    assert isinstance(json_response["veracity"], int)  # Veracity should be an integer


def test_empty_claim():
    response = client.post("/claim/v1/predict", json={"claim_text": ""})

    # Ensure the request was successful but veracity is handled appropriately
    assert response.status_code == 200

    json_response = response.json()
    assert "veracity" in json_response
    assert isinstance(json_response["veracity"], int)  # Even for an empty claim, veracity is returned
