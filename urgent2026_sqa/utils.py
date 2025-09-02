from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import scipy.stats
import torch


def lens2mask(lens: int["b"]) -> int["b n"]:
    """
    Make padding mask from lengths.
    1 = valid, 0 = pad
    """
    seq = torch.arange(lens.amax(), device=lens.device)
    return (seq[None, :] < lens[:, None]).long()


def mask2lens(mask: int["b n"]) -> int["b"]:
    return mask.sum(dim=1)


def default(value, default_value):
    return value if value is not None else default_value


def exists(value):
    return value is not None


def override(d: dict, **kwargs):
    """
    Override dictionary d with non-None keyword arguments.
    """
    for k, v in kwargs.items():
        if v is not None:
            d[k] = v
    return d


def calculate_metrics(preds: list[dict], refs: list[dict]) -> dict[str, dict[str, float]]:
    """
    preds: list of {"sample_id": str, "system_id": str, "value": float}
    lables: list of {"sample_id": str, "system_id": str, "value": float}
    """
    df_pred = pd.DataFrame(preds).rename(columns={"value": "pred"})
    df_ref = pd.DataFrame(refs).rename(columns={"value": "ref"})

    df_pred = df_pred.merge(df_ref, on=["sample_id", "system_id"], how="left")
    if df_pred["ref"].isna().any():
        missing_rows = df_pred[df_pred["ref"].isna()]
        raise ValueError(f"Missing refs for some predictions:\n{missing_rows}")

    utt_pred = df_pred.sort_values(by=["sample_id"])["pred"].to_numpy(dtype=float)
    utt_ref = df_pred.sort_values(by=["sample_id"])["ref"].to_numpy(dtype=float)

    sys_df = df_pred.groupby("system_id", as_index=False).agg(pred=("pred", "mean"), ref=("ref", "mean"))

    sys_pred = sys_df.sort_values(by=["system_id"])["pred"].to_numpy(dtype=float)
    sys_ref = sys_df.sort_values(by=["system_id"])["ref"].to_numpy(dtype=float)

    def metrics(preds: float["b"], refs: float["b"]) -> dict[str, float]:
        mse = np.mean((preds - refs) ** 2)
        lcc = np.corrcoef(preds, refs)[0, 1]
        srcc = scipy.stats.spearmanr(preds, refs).statistic
        ktau = scipy.stats.kendalltau(preds, refs).statistic
        return {"mse": mse, "lcc": lcc, "srcc": srcc, "ktau": ktau}

    return {"utt": metrics(utt_pred, utt_ref), "sys": metrics(sys_pred, sys_ref)}


# https://github.com/facebookresearch/fairseq2/blob/077ac04e89a4ebfdc0691ee0bdb84883391e8c2a/src/fairseq2/nn/utils/grad.py#L54
def scale_grad(x: torch.nn.Tensor, scale: float) -> torch.nn.Tensor:
    """Scale the gradient of ``x`` during backpropagation.

    This is typically used to allow one part of a model to learn at a lower rate
    than the rest.

    :param x:
        The input tensor.
    :param scale:
        The scale factor of the gradient.
    """
    return _GradScaleFunction.apply(x, scale)


class _GradScaleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.nn.Tensor, scale: float) -> torch.nn.Tensor:
        if not x.dtype.is_floating_point:
            raise TypeError(f"`x` must be a float tensor, but is a `{x.dtype}` tensor instead.")

        ctx.scale = scale

        return x.detach().clone()

    @staticmethod
    def backward(ctx: Any, grad_output: torch.nn.Tensor) -> tuple[torch.nn.Tensor, None]:
        return grad_output * ctx.scale, None
