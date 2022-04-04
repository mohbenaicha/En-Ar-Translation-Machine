from typing import Generator, List

import pytest
from fastapi.testclient import TestClient
from translator_model.config.base import config
from translator_model.utilities.data_manager import load_dataset

from app.main import app


@pytest.fixture(scope="module")
def test_data() -> List[List[str]]:
    return load_dataset(
        file_name=config.app_config.test_data_file,
        file_type=config.app_config.test_data_file_type,
        pairs=True,
        delimiter="\t",
    )


@pytest.fixture()
def client() -> Generator:
    with TestClient(app) as _client:
        yield _client
        app.dependency_overrides = {}
