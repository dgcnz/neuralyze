# Neuralyze

[wip] This is a package that will contain useful tools when analyzing neural networks.

## Setup

For pip:
```sh
pip install git+https://github.com/dgcnz/neuralyze
```
For poetry:
```sh
poetry add git+https://github.com/dgcnz/neuralyze
```

## Usage

```py
from neuralyze import get_hessian_max_spectrum
spectrum = get_hessian_max_spectrum(
    model=model,
    criterion=loss_fn,
    train_dataloader=train_dataloader,
    weight_decay=weight_decay,
    top_k=top_k,
    cuda=False,
    verbose=False,
)
# plotting etc...
```


Features:
- `get_hessian_max_spectrum`: Retrieves the top eigenvalue spectrum of the hessian as used in:
```
    @inproceedings{park2022how,
        title={How Do Vision Transformers Work?},
        author={Namuk Park and Songkuk Kim},
        booktitle={International Conference on Learning Representations},
        year={2022}
    }
```