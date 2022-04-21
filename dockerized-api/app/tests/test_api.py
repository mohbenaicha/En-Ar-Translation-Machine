import math
from typing import List

from fastapi.testclient import TestClient


def test_make_prediction(client: TestClient, test_data: List[List[str]]) -> None:
    # Given
    payload = {
        "inputs": [{"input_sentence": test_data[0][i]}
                   for i in range(len(test_data[0]))]
    }

    # When
    response = client.post(
        "http://localhost:8001/api/v1/predict",
        json=payload,
    )

    # Then
    assert response.status_code == 200
    result = response.json()

    expected_no_predictions = 2160

    predictions = result.get("predictions")
    assert isinstance(predictions, list)
    assert isinstance(predictions[0], str)
    assert result.get("errors") is None
    assert len(predictions) == expected_no_predictions
    # make sure the translation isn't unreasonably long
    assert math.isclose(
                        len(test_data[0][0].split(" ")),
                        len(predictions[0]),
                        abs_tol=4 * len(test_data[0][0].split(" ")),
    )
