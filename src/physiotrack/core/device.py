"""Device-argument normalisation for the PyTorch-backed predictors."""
from typing import Union

__all__ = ["torch_device"]


def torch_device(device: Union[str, int]) -> str:
    """Normalise a predictor ``device`` argument to a PyTorch device string.

    Accepts the same values as the YOLO-backed predictors, so every predictor takes the
    same ``device`` spelling. The request is honoured as given: asking for CUDA on a
    machine without it fails in PyTorch rather than silently running on the CPU.

    Args:
        device (str | int): ``"cpu"``, ``"cuda"``, ``"cuda:<i>"``, ``"mps"``, or a CUDA
            device index such as ``0``.

    Returns:
        str: The PyTorch device string, e.g. ``"cuda:0"``.

    Raises:
        ValueError: If ``device`` is a boolean, a negative index, or names several
            devices (e.g. ``"0,1"``): these predictors run on one device.
    """
    if isinstance(device, bool):
        raise ValueError(f"device must be a string or CUDA index, not {device!r}.")
    if isinstance(device, int):
        if device < 0:
            raise ValueError(f"CUDA device index must be >= 0, got {device}.")
        return f"cuda:{device}"
    token = str(device).strip().lower()
    if "," in token:
        raise ValueError(f"These predictors run on a single device; got {device!r}.")
    return f"cuda:{token}" if token.isdigit() else token
