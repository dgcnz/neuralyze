import pytest
from ..mocks.mocks import DatasetMock, ModelMock

@pytest.fixture
def fx_small_dataset():
    return DatasetMock(input_size=10, output_size=2, num_samples=1000)


@pytest.fixture
def fx_small_model(fx_small_dataset: DatasetMock):
    return ModelMock(
        input_size=fx_small_dataset.input_size, output_size=fx_small_dataset.output_size
    )