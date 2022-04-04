from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ValidationError


def validate_inputs(
    *,
    input_data: List[str],
) -> Tuple[List[Dict[str, str]], Optional[Any]]:

    """Validate inputs are as expected according to a defined
    Pydantic schema."""
    validated_data = [{"input_sentence": sent} for sent in input_data]
    errors = None

    try:
        MultipleTranslationDataInputs(inputs=validated_data)
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors


class TranslationDataInputSchema(BaseModel):
    """Single-record schema"""

    input_sentence: str


class MultipleTranslationDataInputs(BaseModel):
    """Applying schema to input data structure type
    In this case, it's a list of strings
    """

    inputs: List[TranslationDataInputSchema]
