from typing import Optional

import torch
from pyhessian import hessian
from torch.utils.data import DataLoader
from tqdm import tqdm


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
        hessian_comp = hessian(
            model, criterion, data=(xs, ys), weight_decay=weight_decay, cuda=cuda
        )
        top_eigenvalues, _ = hessian_comp.eigenvalues(top_n=top_k)
        max_eigens.extend(top_eigenvalues)
    return max_eigens
