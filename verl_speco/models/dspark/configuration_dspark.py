from __future__ import annotations

import json
import os
from typing import Optional

from verl_speco.models.dflash import DFlashConfig


class DSparkConfig(DFlashConfig):
    """Configuration for the DSpark draft model.

    DSpark uses the same target-hidden-state context backbone as DFlash and adds
    a Markov head plus an optional confidence head.
    """

    model_type = "dspark"

    def __init__(
        self,
        *args,
        block_size: int = 7,
        num_anchors: int = 512,
        markov_rank: int = 256,
        markov_head_type: str = "vanilla",
        enable_confidence_head: Optional[bool] = None,
        confidence_head_alpha: float = 0.0,
        confidence_head_with_markov: bool = True,
        ce_loss_alpha: float = 1.0,
        l1_loss_alpha: float = 0.0,
        loss_decay_gamma: float = 7.0,
        **kwargs,
    ):
        architectures = kwargs.pop("architectures", None)
        super().__init__(*args, **kwargs)
        self.architectures = architectures or ["DSparkDraftModel"]
        self.block_size = int(block_size)
        self.num_anchors = int(num_anchors)
        self.markov_rank = int(markov_rank)
        self.markov_head_type = str(markov_head_type)
        self.confidence_head_alpha = float(confidence_head_alpha)
        self.enable_confidence_head = (
            bool(enable_confidence_head)
            if enable_confidence_head is not None
            else self.confidence_head_alpha > 0.0
        )
        self.confidence_head_with_markov = bool(confidence_head_with_markov)
        self.ce_loss_alpha = float(ce_loss_alpha)
        self.l1_loss_alpha = float(l1_loss_alpha)
        self.loss_decay_gamma = float(loss_decay_gamma)

    @classmethod
    def from_dspark_pretrained(cls, model_path: str):
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        config["model_type"] = cls.model_type
        config["architectures"] = ["DSparkDraftModel"]
        if "enable_confidence_head" not in config:
            config["enable_confidence_head"] = float(config.get("confidence_head_alpha", 0.0)) > 0.0
        return cls.from_dict(config)
