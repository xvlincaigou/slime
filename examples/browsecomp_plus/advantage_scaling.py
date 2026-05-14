from __future__ import annotations

from typing import Any

import torch


def apply_turn_hit_scaling(
    args,
    *,
    pg_loss: torch.Tensor,
    loss_masks: list[torch.Tensor],
    metadata: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    if metadata is None:
        raise ValueError("BrowseComp-Plus turn-hit scaling requires per-sample metadata.")

    weight_tensors = []
    for sample_metadata, loss_mask in zip(metadata, loss_masks, strict=True):
        weights = sample_metadata.get("advantage_weights")
        if weights is None:
            raise ValueError("Missing `advantage_weights` in sample metadata.")
        if len(weights) != loss_mask.numel():
            raise ValueError(
                f"advantage_weights length {len(weights)} does not match response length {loss_mask.numel()}."
            )
        weight_tensors.append(torch.tensor(weights, dtype=pg_loss.dtype, device=pg_loss.device))

    token_weights = torch.cat(weight_tensors, dim=0)
    scaled_pg_loss = pg_loss * token_weights
    metrics = {
        "browsecomp_plus_adv_weight": token_weights.clone().detach(),
    }
    return scaled_pg_loss, loss_masks, metrics
