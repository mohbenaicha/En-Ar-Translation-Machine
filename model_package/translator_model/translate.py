from typing import List

from translator_model import __version__ as _version
from translator_model.utilities.data_manager import load_pipeline
from translator_model.utilities.validation import validate_inputs

translation_pipe = load_pipeline()


def make_translation(*, input_data: List[str]) -> dict:
    """Make a prediction using a saved model pipeline."""

    validated_data, errors = validate_inputs(input_data=input_data)
    validated_data = [element.get("input_sentence") for element in validated_data]

    results = {"predictions": None, "version": _version, "errors": errors}

    if not errors:
        predictions = translation_pipe.translate(query=validated_data)
        results = {
            "predictions": predictions,
            "version": _version,
            "errors": errors,
        }

    return results
