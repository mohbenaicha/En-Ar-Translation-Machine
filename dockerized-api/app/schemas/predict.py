from typing import Any, List, Optional

from pydantic import BaseModel
from translator_model.utilities.validation import TranslationDataInputSchema


class PredictionResults(BaseModel):
    predictions: Optional[List[str]]
    version: str
    errors: Optional[Any]


class MultipleTranslationDataInputs(BaseModel):
    inputs: List[TranslationDataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {"input_sentence": "Translate this sentence for me, please?"}
                ]
            }
        }
