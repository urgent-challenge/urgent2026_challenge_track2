from __future__ import annotations

import contextlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import AutoModel

from urgent2026_sqa.utils import lens2mask, scale_grad


class SSLFeatureExtractor(nn.Module):
    def __init__(self, model_name: str, freeze: bool = True, gradient_scale: float = 0.1):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.weights = nn.Parameter(torch.zeros(self.model.config.num_hidden_layers + 1))
        self.freeze = freeze
        self.gradient_scale = gradient_scale
        self.freeze = freeze
        if self.freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, audio: float["b t"], audio_lengths: int["b"]):
        attn_mask = lens2mask(audio_lengths)
        context = contextlib.nullcontext() if not self.freeze else torch.no_grad()
        with context:
            hidden_states = self.model(
                input_values=audio, attention_mask=attn_mask, output_hidden_states=True
            ).hidden_states
            hidden_states = torch.stack(hidden_states, dim=0)  # [l, b, t, d]
            hidden_states = scale_grad(hidden_states, self.gradient_scale)
        output = hidden_states * rearrange(F.softmax(self.weights, dim=0), "l -> l 1 1 1")
        output = output.sum(dim=0)  # [b ,t ,d]
        output_lengths = self.model._get_feat_extract_output_lengths(audio_lengths)
        return output, output_lengths

    @property
    def output_dim(self):
        return self.model.config.hidden_size


class RangeActivation(nn.Module):
    """
    Range bounds:
      - Two-sided: scaled sigmoid
      - Lower-only: softplus lower clamp
      - Upper-only: softplus upper clamp
      - Identity if both bounds are infinite
    """

    def __init__(self, min_value=-math.inf, max_value=math.inf):
        super().__init__()
        self.register_buffer("min_value", torch.as_tensor(float(min_value)))
        self.register_buffer("max_value", torch.as_tensor(float(max_value)))
        if math.isfinite(min_value) and math.isfinite(max_value):
            if not (max_value > min_value):
                raise ValueError("max_value must be larger than min_value")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        min_is_finite = torch.isfinite(self.min_value)
        max_is_finite = torch.isfinite(self.max_value)
        max_value = self.max_value.to(x.dtype)  # amp friendly
        min_value = self.min_value.to(x.dtype)

        if not min_is_finite and not max_is_finite:
            return x
        if min_is_finite and max_is_finite:
            return min_value + (max_value - min_value) * torch.sigmoid(x)
        elif min_is_finite:
            return min_value + F.softplus(x - min_value)
        else:
            return max_value - F.softplus(max_value - x)


class UniVersaExt(nn.Module):
    def __init__(
        self,
        feature_extractor: nn.Module,
        encoder: nn.Module,
        metrics: dict[str, dict[str, float]],
    ):
        """
        metrics: list of dict mapping metric name to {"min": float, "max": float, "weight": float}
        """
        super().__init__()
        self.feature_extractor = feature_extractor

        if isinstance(encoder, nn.TransformerEncoder):
            encoder_dim = encoder.layers[0].linear1.in_features
        else:
            raise ValueError(f"Define your own encoder_dim for {type(encoder)}")

        self.feat_proj = nn.Sequential(
            nn.Linear(feature_extractor.output_dim, encoder_dim),
            nn.LayerNorm(encoder_dim),
        )
        self.encoder = encoder

        self.metrics = metrics
        self.metric_proj = nn.Linear(encoder_dim, len(metrics))
        self.metric2idx = {name: i for i, name in enumerate(metrics.keys())}
        self.metric2act = nn.ModuleDict(
            {name: RangeActivation(rng["min"], rng["max"]) for name, rng in metrics.items()}
        )
        self.metric2weight = {name: metric.get("weight", 1.0) for name, metric in metrics.items()}

    def forward(
        self, audio: float["b t"], audio_lengths: int["b"], metrics: dict[str, float["b"]], **kwargs
    ) -> tuple[torch.Tensor, dict[str, float]]:
        metric2pred = self.predict(audio, audio_lengths)
        loss, loss_detail = 0, {}
        for name in metric2pred.keys():
            pred = metric2pred[name]
            ref_mask: bool["b"] = ~torch.isnan(metrics[name])
            if ref_mask.any():
                loss_metric = (
                    self.metric2weight[name]
                    * (F.mse_loss(pred.float()[ref_mask], metrics[name].float()[ref_mask], reduction="none")).sum()
                    / ref_mask.sum()
                )
                loss_detail[f"loss_{name}"] = loss_metric.detach().cpu().item()
                loss += loss_metric

        loss /= len(loss_detail)

        other = {
            "info": loss_detail,
            "metric2pred": metric2pred,
        }

        return loss, other

    def predict(self, audio: float["b t"], audio_lengths: int["b"], **kwargs) -> dict[str, torch.Tensor]:
        feats, feats_lengths = self.feature_extractor(audio, audio_lengths)
        feats = self.feat_proj(feats)
        padding_mask = lens2mask(feats_lengths)
        x = self.encoder(feats, src_key_padding_mask=(1 - padding_mask).bool()) * padding_mask.unsqueeze(-1)

        pooled = x.sum(dim=1) / feats_lengths.unsqueeze(-1)
        metric_logits = self.metric_proj(pooled)
        metric2pred: dict[str, float["b"]] = {
            name: self.metric2act[name](metric_logits[:, self.metric2idx[name]]) for name in self.metric2idx
        }
        return metric2pred
