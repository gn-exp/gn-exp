import typing

import torch
import numpy as np


def repeat_tensor(input: torch.Tensor, repeats: torch.LongTensor, dim: int = 0) -> torch.Tensor:
    """ Repeats each entry of a tensor along a given dimension according to a tensor of repetitions,
    gradients can be computed w.r.t. `tensor`, but not w.r.t. `repeats`

    Args:
        input: a tensor to repeat, e.g. [x, y, z]
        repeats: the non-negative number of repetition of each entry of the tensor, e.g. [2, 3, 1]
        dim: the dimension used to repeat the tensor

    Returns:
        A tensor with repeated entries that has the same type and placement as `tensor`

    Examples:
        Each element of `x` is repeated according to the corresponding number of repetitions in `repeats`

        >>> x = torch.tensor([a, b, c, d])
        >>> repeats = torch.tensor([2, 3, 0, 1])
        >>> repeat_tensor(x, repeats, dim=0)
        tensor([a, a, b, b, b, d])

        Gradient information can be propagated through the repetition

        >>> x = torch.tensor([a, b, c, d], requires_grad=True)
        >>> repeats = torch.tensor([2, 3, 0, 1])
        >>> repeat_tensor(x, repeats, dim=0).sum().backward()
        >>> x.grad
        tensor([2., 3., 0., 1.])
    """
    if repeats.dim() != 1:
        raise ValueError(f'`repeats` should have a single dimension, got shape {repeats.shape}')
    if (repeats < 0).any():
        raise ValueError(f'All entries in `repeats` should be non-negative')
    if len(repeats) != input.shape[dim]:
        raise ValueError(f'`input.shape[dim]` should match `len(repeats)`, got {input.shape[dim]} and {len(repeats)}')

    index = input.new_tensor(np.arange(len(repeats)).repeat(repeats.cpu().numpy()), dtype=torch.long)
    return torch.index_select(input, index=index, dim=dim)


def segment_lengths_to_ids(segment_lengths: torch.LongTensor) -> torch.LongTensor:
    """
    Args:
        segment_lengths: Non-negative lengths of the tensor segments

    Returns:
        A tensor containing ids for every element in the tensor to be segmented

    Examples:
        >>> segments = torch.tensor([2, 4, 3, 1])
        >>> segment_lengths_to_slices(segments)
        tensor([0, 0, 1, 1, 1, 1, 2, 2, 2, 3])
    """
    if segment_lengths.dim() != 1:
        raise ValueError(f'`segment_lengths` should have a single dimension, got shape {segment_lengths.shape}')
    if (segment_lengths < 0).any():
        raise ValueError(f'All entries in `segment_lengths` should be non-negative')

    return segment_lengths.new_tensor(np.arange(len(segment_lengths)).repeat(segment_lengths.cpu().numpy()))


def segment_lengths_to_slices(segment_lengths: torch.LongTensor) -> typing.Iterator[slice]:
    """
    Args:
        segment_lengths: Non-negative lengths of the tensor segments

    Yields:
        Slices to slice the tensor according to the segments

    Examples:
        >>> segments = torch.tensor([2, 4, 3, 1])
        >>> list(segment_lengths_to_slices(segments))
        [0:2, 2:6, 6:9, 9:10]
    """
    assert segment_lengths.dim() == 1
    assert (segment_lengths >= 0).all()

    indexes = segment_lengths.cumsum(dim=0)
    yield slice(indexes.new_tensor(0), indexes[0])
    for start, end in zip(indexes[:-1], indexes[1:]):
        yield slice(start, end)
