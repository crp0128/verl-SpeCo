# Licensed under the Apache License, Version 2.0. See LICENSE.
"""Consolidated multi-node capabilities. Hardware adapters load only when explicitly selected."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Mapping
import torch
from torch import nn
from verl_speco.trainer.distributed_metrics import observe_training_acceptance
from verl_speco.backends.dspark_multinode_backend import MultiNodeDSparkTrainingModel as DSparkTrainingModel
from verl_speco.models.dflash.modeling_dflash import DFlashRotaryEmbedding

@dataclass
class DSparkVeOmniModelConfig:
    """Minimal model config consumed by VeOmniEngine for a custom DSpark module."""
    trust_remote_code: bool = False
    use_remove_padding: bool = False
    enable_gradient_checkpointing: bool = False
    enable_activation_offload: bool = False
    lora_rank: int = 0

    def get_processor(self) -> None:
        return None

class VeOmniDSparkModule(nn.Module):
    """Give VeOmni a model-level init hook for randomly initialized DSpark."""
    _no_split_modules = ['DFlashDecoderLayer']

    def __init__(self, training_model: DSparkTrainingModel):
        super().__init__()
        self.training_model = training_model
        self.config = training_model.draft_model.config

    def init_weights(self) -> None:
        self.training_model.draft_model.init_weights()
        for module in self.modules():
            if isinstance(module, DFlashRotaryEmbedding):
                module._set_cos_sin_cache(module.max_seq_len_cached, module.inv_freq.device, torch.float32)

    def forward(self, **batch: Any):
        return self.training_model(**batch)

def validate_dp_only_engine_config(engine_config: Any) -> None:
    """Reject unsupported draft parallelism before VeOmni creates a process mesh."""
    if engine_config is None or engine_config.init_device != 'meta':
        raise ValueError("VeOmni DSpark data parallelism requires init_device='meta'")
    if engine_config.fsdp_size != 1:
        raise ValueError('DSpark draft trainer requires a DP-only mesh (fsdp_size=1)')
    if engine_config.ulysses_parallel_size != 1 or engine_config.expert_parallel_size != 1:
        raise ValueError('first-release DSpark supports data parallelism only')

def build_veomni_dspark_engine_class():
    """Load VeRL lazily so CPU protocol tests need no accelerator engine."""
    from veomni.arguments import MixedPrecisionConfig
    from veomni.distributed.offloading import build_activation_offloading_context
    from veomni.distributed.torch_parallelize import build_parallelize_model
    from verl.workers.engine.veomni.transformer_impl import VeOmniEngine

    class VeOmniDSparkEngine(VeOmniEngine):

        def __init__(self, *, model_factory: Callable[[], DSparkTrainingModel], **kwargs: Any):
            validate_dp_only_engine_config(kwargs.get('engine_config'))
            self._speco_model_factory = model_factory
            super().__init__(**kwargs)

        def _build_model_optimizer(self) -> None:
            with torch.device('meta'):
                module = VeOmniDSparkModule(self._speco_model_factory())
            module = build_parallelize_model(module, weights_path=None, init_device='meta', enable_full_shard=self.engine_config.enable_full_shard, mixed_precision=MixedPrecisionConfig(enable=bool(self.engine_config.mixed_precision)), enable_gradient_checkpointing=False, enable_fsdp_offload=self.engine_config.enable_fsdp_offload, basic_modules=['DFlashDecoderLayer'], enable_reentrant=False, enable_forward_prefetch=False)
            self.module = module
            self.optimizer = self._build_optimizer(module)
            self.lr_scheduler = self._build_lr_scheduler(self.optimizer)
            self.model_fwd_context, self.model_bwd_context = build_activation_offloading_context(False, False, 0.0)

        @observe_training_acceptance
        def train_dspark_window_shard(self, shard) -> tuple[float, float]:
            """Update once from pair-weighted windows assigned to this DP rank."""
            from verl_speco.trainer.distributed_batches import backward_dspark_window_shard
            self.module.train()
            self.optimizer.zero_grad(set_to_none=True)
            loss_value = backward_dspark_window_shard(self.module, shard)
            grad_norm = self.optimizer_step()
            self.lr_scheduler.step()
            return (loss_value, float(grad_norm))

        @observe_training_acceptance
        def train_dspark_swa_part_shard(self, shard) -> tuple[float, float]:
            """Opt-in DP mean-part backward over complete-prefill SWA captures."""
            from verl_speco.trainer.distributed_batches import backward_dmtp_swa_parts
            from verl_speco.backends.dflash_trainer_backend import _resolve_sliding_windows
            if getattr(self.module.training_model, 'anchor_sampling_policy', None) != 'dmtp_cpu':
                raise ValueError('SWA parts require the explicit DMTP anchor policy')
            windows = _resolve_sliding_windows(self.module.config)
            if not windows or any((window is None for window in windows)):
                raise ValueError('SWA parts require sliding-window attention on every layer')
            if getattr(self.module.config, 'draft_response_plan', None) != 'dmtp_swa_v1' or getattr(self.module.config, 'draft_loss_reduction', None) != 'part_mean':
                raise ValueError('SWA parts require explicit checkpoint plan/reduction identity')
            self.module.train()
            self.optimizer.zero_grad(set_to_none=True)
            loss_value = backward_dmtp_swa_parts(self.module, shard)
            grad_norm = self.optimizer_step()
            self.lr_scheduler.step()
            return (loss_value, float(grad_norm))

        @observe_training_acceptance
        def train_dspark_batch(self, batch: Mapping[str, torch.Tensor], *, window_size: int=0, loss_scale: float=1.0) -> tuple[float, float]:
            """Run one VeOmni-managed optimizer update after a complete feature batch."""
            import math
            if not math.isfinite(loss_scale) or loss_scale < 0:
                raise ValueError('DSpark gradient loss scale must be finite and non-negative')
            self.module.train()
            self.optimizer.zero_grad(set_to_none=True)
            if window_size:
                from verl_speco.trainer.distributed_batches import backward_dspark_windows
                loss_value = backward_dspark_windows(self.module, batch, window_size, loss_scale=loss_scale)
            else:
                output = self.module(**dict(batch))
                loss = output[0]
                if not torch.isfinite(loss):
                    raise RuntimeError('DSpark loss is not finite')
                (loss * loss_scale).backward()
                loss_value = float(loss.detach().item())
            grad_norm = self.optimizer_step()
            self.lr_scheduler.step()
            return (loss_value, float(grad_norm))
    return VeOmniDSparkEngine
