from neuralyze.tools import get_hessian_max_spectrum
from .mocks.mocks import DatasetMock, ModelMock
from torch.utils.data import DataLoader
import torch
import pytest


@pytest.mark.parametrize("weight_decay", [None, 0.01])
def test_get_hessian_max_spectrum(
    weight_decay: float, fx_small_dataset: DatasetMock, fx_small_model: ModelMock
):
    train_dataloader = DataLoader(fx_small_dataset, batch_size=10, shuffle=True)
    top_k = 5

    criterion = torch.nn.CrossEntropyLoss()
    res = get_hessian_max_spectrum(
        model=fx_small_model,
        criterion=criterion,
        train_dataloader=train_dataloader,
        weight_decay=weight_decay,
        top_k=top_k,
        cuda=False,
        verbose=False,
    )
    assert res is not None
