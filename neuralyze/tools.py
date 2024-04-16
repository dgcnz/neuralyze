from typing import Optional

import torch
from pyhessian import hessian
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm


class L2RegularizedLossWithFrozenWeights(torch.nn.Module):
    """
    Wrapper for a loss function that adds L2 regularization
    """

    def __init__(
        self, criterion: torch.nn.Module, weight_decay: float, params: list[Tensor]
    ):
        super().__init__()

        # References for 0.5 factor:
        # - https://discuss.pytorch.org/t/a-bug-of-pytorch-about-optim-sgd-weight-decay/55490
        # - https://d2l.ai/chapter_linear-regression/weight-decay.html
        self.l2_penalty = 0.5 * weight_decay * sum(param.square() for param in params)
        self.criterion = criterion

    def forward(self, *args, **kwargs):
        return self.criterion(*args, **kwargs) + self.l2_penalty


def get_hessian_max_spectrum(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    train_dataloader: DataLoader,
    weight_decay: Optional[float] = None,
    top_k: int = 5,
    cuda: bool = False,
    verbose: bool = True,
) -> list[float]:
    """
    Compute batched top-k hessian max eigenvalues of a model with respect to a criterion and a dataloader
    using [power-iteration](https://en.wikipedia.org/wiki/Power_iteration) as per
    [Park et al. 2021](https://arxiv.org/abs/2105.12639) and
    [Park et al. 2022](https://github.com/xxxnell/how-do-vits-work/issues/12#issuecomment-1113991017)


    :param model: Neural network model
    :param criterion: Loss function
    :param train_dataloader: DataLoader of the form (inputs, targets). Must encode data augmentations used in training.
    :param weight_decay: L2 regularization strength used in training
    :param top_k: Number of top eigenvalues per batch to compute. Not the total size of the returned list.
    :param cuda: Whether to use GPU
    """

    max_eigens: list[float] = []
    for xs, ys in tqdm(train_dataloader, disable=not verbose):
        hessian_comp = hessian(model, criterion, data=(xs, ys), weight_decay=weight_decay, cuda=cuda)
        top_eigenvalues, _ = hessian_comp.eigenvalues(top_n=top_k)
        max_eigens.extend(top_eigenvalues)
    return max_eigens
