from neuralyze.tools import get_hessian_max_spectrum
from .mocks.mocks import DatasetMock, ModelMock
from torch.utils.data import DataLoader
import torch
import pytest


@pytest.mark.parametrize("weight_decay", [None, 0.01])
def test_get_hessian_max_spectrum(
    weight_decay: float, fx_small_dataset: DatasetMock, fx_small_model: ModelMock
):
    # initialize model weights with gaussian large values
    for param in fx_small_model.parameters():
        param.data = torch.randn_like(param.data) * 1000
    top_k = 50

    criterion = torch.nn.CrossEntropyLoss()
    res = get_hessian_max_spectrum(
        model=fx_small_model,
        criterion=criterion,
        train_dataset=fx_small_dataset,
        batch_size=16,
        percentage_data=0.1,
        weight_decay=weight_decay,
        hessian_top_k=top_k,
        hessian_tol=1e-1,
        hessian_max_iter=200,
        cuda=False,
        verbose=False,
    )
    assert res is not None
