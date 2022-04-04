import math
from translator_model.translate import make_translation


def test_make_translation(test_input_data):
    # Given
    expected_no_predictions = 2160

    # When
    result = make_translation(input_data=test_input_data[0])

    # Then
    predictions = result.get("predictions")
    assert isinstance(predictions, list)
    assert isinstance(predictions[0], str)
    assert result.get("errors") is None
    assert len(predictions) == expected_no_predictions
    # make sure the translation isn't unreasonably long
    assert math.isclose(
        len(test_input_data[0][0].split(" ")),
        len(predictions[0]),
        abs_tol=4 * len(test_input_data[0][0].split(" ")),
    )
