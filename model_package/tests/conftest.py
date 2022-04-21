import pytest

from translator_model.config.base import config
from translator_model.utilities.data_manager import load_dataset


@pytest.fixture()
def test_input_data():
    return load_dataset(
        file_name=config.app_config.test_data_file,
        file_type=config.app_config.test_data_file_type,
        pairs=True,
        delimiter="\t",
    )
