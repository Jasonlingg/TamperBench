"""Optimizer selection overrides shared by the attack benchmark CLIs."""

from argparse import ArgumentParser
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from tamperbench.whitebox.utils.names import AttackName


def add_optimizer_argument(parser: ArgumentParser) -> None:
    """Add an optional optimizer override; omission preserves the YAML configuration."""
    parser.add_argument(
        "--optim",
        type=str,
        default=None,
        help=(
            "Override the YAML optimizer (e.g. adamw_torch or muon). "
            "Muon supports full_parameter_finetune and benign_full_parameter_finetune."
        ),
    )


def validate_optimizer_override(attack_names: Sequence[AttackName], optim: str | None) -> None:
    """Check Muon attack selection before launching any experiments.

    Raises:
        ValueError: If Muon is requested for an attack outside the full-finetuning scope.
    """
    if optim != "muon":
        return

    supported = {AttackName.FULL_PARAMETER_FINETUNE, AttackName.BENIGN_FULL_PARAMETER_FINETUNE}
    unsupported = [str(name) for name in attack_names if name not in supported]
    if unsupported:
        raise ValueError(
            "Muon supports full-parameter fine-tuning only; unsupported attacks: " + ", ".join(unsupported)
        )


def override_optimizer(config: Mapping[str, Any], optim: str | None) -> dict[str, Any]:
    """Copy a config and apply an explicit CLI optimizer override.

    Raises:
        ValueError: If an override is requested for a configuration without an optimizer field.
    """
    result = deepcopy(dict(config))
    if optim is not None:
        if "optim" not in result:
            raise ValueError("--optim requires an attack configuration with an 'optim' field.")
        result["optim"] = optim
    return result
