"""Native PyTorch Muon with auxiliary AdamW for language-model fine-tuning.

Muon updates hidden linear weight matrices; AdamW updates embeddings, the output
head, biases, normalization weights, and other parameters. Update math is delegated
to torch.optim.Muon and torch.optim.AdamW, rather than reimplemented here.

References:
    Liu et al., "Muon is Scalable for LLM Training" (2025).
    https://arxiv.org/abs/2502.16982
    https://docs.pytorch.org/docs/2.9/generated/torch.optim.Muon.html
"""

from collections.abc import Callable
from typing import Any, overload

import torch
from torch import nn
from torch.optim import AdamW, Muon, Optimizer
from transformers import PreTrainedModel
from transformers.pytorch_utils import Conv1D
from typing_extensions import override


def partition_parameters(model: PreTrainedModel) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Split trainable parameters into Muon and AdamW sets by module identity.

    Only weights belonging to Linear or Hugging Face Conv1D modules are Muon
    candidates. All embedding modules and the model's output head take precedence
    over that rule, including when their parameters are tied to another module.
    Frozen parameters are omitted and shared parameters occur only once.
    """
    excluded: set[int] = set()
    candidates: set[int] = set()
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            excluded.update(id(param) for param in module.parameters())
        elif isinstance(module, nn.Linear | Conv1D):
            candidates.add(id(module.weight))

    for module in (model.get_input_embeddings(), model.get_output_embeddings()):
        if module is not None:
            excluded.update(id(param) for param in module.parameters())

    muon_params: list[nn.Parameter] = []
    adamw_params: list[nn.Parameter] = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 2 and id(param) in candidates and id(param) not in excluded:
            muon_params.append(param)
        else:
            adamw_params.append(param)
    return muon_params, adamw_params


class MuonAdamW(Optimizer):
    """Expose native Muon and AdamW as one scheduler/checkpoint-compatible optimizer.

    Both child optimizers share this optimizer's parameter-group dictionaries and
    state mapping. Consequently, Trainer's scheduler, gradient clearing and state
    serialization operate on both algorithms through the standard Optimizer API.
    """

    def __init__(self, muon: Muon, adamw: AdamW | None) -> None:
        """Combine freshly constructed native optimizers with disjoint parameters."""
        self._optimizers: tuple[Optimizer, ...] = (muon,) if adamw is None else (muon, adamw)
        groups: list[dict[str, Any]] = []
        for optimizer in self._optimizers:
            if optimizer.state:
                raise ValueError("MuonAdamW requires optimizers that have not taken any steps.")
            for group in optimizer.param_groups:
                group["optimizer_type"] = "muon" if optimizer is muon else "adamw"
                groups.append(group)
        super().__init__(groups, defaults={})
        self._bind_child_optimizers()

    def _bind_child_optimizers(self) -> None:
        """Rebind shared groups/state after initialization or checkpoint restoration."""
        offset = 0
        for optimizer in self._optimizers:
            count = len(optimizer.param_groups)
            optimizer.param_groups = self.param_groups[offset : offset + count]
            optimizer.state = self.state
            offset += count

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @override
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Run a closure once, then update both disjoint parameter sets."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for optimizer in self._optimizers:
            optimizer.step()
        return loss

    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore both algorithms and reconnect their groups to the scheduler."""
        saved_groups = state_dict["param_groups"]
        if [group.get("optimizer_type") for group in saved_groups] != [
            group["optimizer_type"] for group in self.param_groups
        ]:
            raise ValueError("Checkpoint optimizer groups do not match the Muon/AdamW parameter routing.")
        super().load_state_dict(state_dict)
        self._bind_child_optimizers()


def build_muon_optimizer(
    model: PreTrainedModel,
    *,
    decay_parameter_names: set[str],
    learning_rate: float,
    weight_decay: float,
    momentum: float = 0.95,
    nesterov: bool = True,
    ns_steps: int = 5,
    adjust_lr_fn: str = "match_rms_adamw",
    adamw_learning_rate: float | None = None,
    adamw_betas: tuple[float, float] = (0.9, 0.999),
    adamw_eps: float = 1e-8,
) -> MuonAdamW:
    """Build Muon and AdamW using the trainer's existing weight-decay exclusions.

    Each algorithm gets separate decay/no-decay groups. An omitted auxiliary
    learning rate shares the main rate; a supplied rate retains its ratio under
    Trainer's learning-rate schedule. Unsupported hidden-module types fall back
    to AdamW rather than being classified from their shape alone.
    """
    muon_params, adamw_params = partition_parameters(model)
    if not muon_params:
        raise ValueError("Muon requires at least one trainable hidden Linear or Conv1D weight matrix.")

    decay_ids = {id(param) for name, param in model.named_parameters() if name in decay_parameter_names}

    def make_groups(params: list[nn.Parameter]) -> list[dict[str, Any]]:
        groups: list[dict[str, Any]] = []
        for decay in (True, False):
            selected = [param for param in params if (id(param) in decay_ids) == decay]
            if selected:
                groups.append({"params": selected, "weight_decay": weight_decay if decay else 0.0})
        return groups

    muon = Muon(
        make_groups(muon_params),
        lr=learning_rate,
        momentum=momentum,
        nesterov=nesterov,
        ns_steps=ns_steps,
        adjust_lr_fn=adjust_lr_fn,
    )
    adamw = (
        AdamW(
            make_groups(adamw_params),
            lr=learning_rate if adamw_learning_rate is None else adamw_learning_rate,
            betas=adamw_betas,
            eps=adamw_eps,
        )
        if adamw_params
        else None
    )
    return MuonAdamW(muon, adamw)
