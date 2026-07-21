"""NPU compatibility for verl release/v0.8.0 vLLM imports and FSDP checkpoints."""

from __future__ import annotations

import functools
import gc
import importlib
import importlib.util
import logging
import sys
import time
from typing import Any, Callable

from packaging import version

logger = logging.getLogger(__name__)

_VERL_NPU_VLLM_PATCH_MODULE = "verl.utils.vllm.npu_vllm_patch"
_VLLM_FUSED_MOE_MODULE = "vllm.model_executor.layers.fused_moe"
_VERL_FSDP_ENGINE_MODULE = "verl.workers.engine.fsdp.transformer_impl"
_IMPORT_COMPAT_APPLIED = False
_NPU_FSDP2_CHECKPOINT_COMPAT_APPLIED = False


def _module_available(module_name: str) -> bool:
    if module_name in sys.modules:
        return True
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _unused_factory_weight_loader(*args, **kwargs):
    del args, kwargs
    raise RuntimeError("FusedMoE factory compatibility weight_loader must never be called")


def install_verl_npu_vllm_import_compat(
    module_importer: Callable[[str], Any] = importlib.import_module,
) -> bool:
    """Import verl's NPU patch without applying its obsolete class-only MoE hook.

    vLLM >= 0.18 exposes ``FusedMoE`` as a factory function. The verl v0.8.0
    patch still accesses ``FusedMoE.weight_loader`` during import. A temporary
    attribute lets the rest of verl's NPU initialization run; it is removed
    immediately because factory instances use their own runner weight loaders.
    """

    global _IMPORT_COMPAT_APPLIED
    if _IMPORT_COMPAT_APPLIED or _VERL_NPU_VLLM_PATCH_MODULE in sys.modules:
        return False
    # Match verl's own guard: its failing import path is enabled by torch_npu,
    # even before vllm_ascend itself has necessarily been imported.
    if not _module_available("torch_npu"):
        return False

    vllm = module_importer("vllm")
    if version.parse(str(getattr(vllm, "__version__", "0"))) < version.parse("0.18.0"):
        return False

    fused_moe_module = module_importer(_VLLM_FUSED_MOE_MODULE)
    fused_moe = getattr(fused_moe_module, "FusedMoE", None)
    if fused_moe is None or isinstance(fused_moe, type) or hasattr(fused_moe, "weight_loader"):
        return False

    fused_moe.weight_loader = _unused_factory_weight_loader
    try:
        module_importer(_VERL_NPU_VLLM_PATCH_MODULE)
    finally:
        if hasattr(fused_moe, "weight_loader"):
            delattr(fused_moe, "weight_loader")

    _IMPORT_COMPAT_APPLIED = True
    logger.warning(
        "Applied verl release/v0.8.0 NPU import compatibility for the vLLM FusedMoE factory"
    )
    return True


def _can_save_npu_fsdp2_checkpoint_from_cpu(engine: Any, fsdp_utils: Any) -> bool:
    if not bool(getattr(engine, "_is_offload_param", False)):
        return False
    if bool(getattr(engine, "_uses_fsdp2_cpu_offload_policy", False)):
        return False

    module = getattr(engine, "module", None)
    if module is None or fsdp_utils.fsdp_version(module) != 2:
        return False
    parameters = list(module.parameters())
    if not parameters or any(parameter.device.type != "cpu" for parameter in parameters):
        return False
    buffers = getattr(module, "buffers", lambda: ())()
    if any(buffer.device.type != "cpu" for buffer in buffers):
        return False

    checkpoint_manager = getattr(engine, "checkpoint_manager", None)
    if checkpoint_manager is None:
        return False
    if bool(getattr(checkpoint_manager, "should_save_optimizer", False)):
        if not bool(getattr(engine, "_is_offload_optimizer", False)):
            return False
        optimizer = getattr(engine, "optimizer", None)
        if optimizer is None:
            return False
        for state in optimizer.state.values():
            values = state.values() if isinstance(state, dict) else (state,)
            if any(getattr(value, "device", None) is not None and value.device.type != "cpu" for value in values):
                return False
    return True


def install_verl_npu_fsdp2_checkpoint_compat(
    module_importer: Callable[[str], Any] = importlib.import_module,
) -> bool:
    """Keep offloaded FSDP2 actor shards on CPU while saving on NPU.

    verl release/v0.8.0 moves an already offloaded FSDP2 model to NPU before
    serializing local shards, then immediately moves it back to CPU. On NPU,
    saving the existing CPU shards avoids that transfer and prevents NPU tensor
    serialization from retaining additional host-side staging memory.
    """

    global _NPU_FSDP2_CHECKPOINT_COMPAT_APPLIED
    if _NPU_FSDP2_CHECKPOINT_COMPAT_APPLIED or not _module_available("torch_npu"):
        return False

    device_module = module_importer("verl.utils.device")
    if device_module.get_device_name() != "npu":
        return False

    engine_module = module_importer(_VERL_FSDP_ENGINE_MODULE)
    engine_cls = getattr(engine_module, "FSDPEngine", None)
    if engine_cls is None:
        return False
    original_save_checkpoint = getattr(engine_cls, "save_checkpoint", None)
    if original_save_checkpoint is None or getattr(
        original_save_checkpoint,
        "_speco_npu_fsdp2_cpu_checkpoint",
        False,
    ):
        return False

    torch_module = module_importer("torch")
    fsdp_utils = module_importer("verl.utils.fsdp_utils")

    @functools.wraps(original_save_checkpoint)
    def save_checkpoint_from_cpu(
        self,
        local_path: str,
        hdfs_path: str | None = None,
        global_step: int = 0,
        max_ckpt_to_keep: int | None = None,
        **kwargs,
    ):
        if not _can_save_npu_fsdp2_checkpoint_from_cpu(self, fsdp_utils):
            return original_save_checkpoint(
                self,
                local_path,
                hdfs_path=hdfs_path,
                global_step=global_step,
                max_ckpt_to_keep=max_ckpt_to_keep,
                **kwargs,
            )

        started = time.perf_counter()
        if getattr(self, "rank", 0) == 0:
            from verl_speco.trainer.checkpoint import format_checkpoint_memory_snapshot

            logger.warning(
                "[actor checkpoint] NPU FSDP2 CPU-shard save step=%s phase=start %s",
                global_step,
                format_checkpoint_memory_snapshot(),
            )
        try:
            result = self.checkpoint_manager.save_checkpoint(
                local_path=local_path,
                hdfs_path=hdfs_path,
                global_step=global_step,
                max_ckpt_to_keep=max_ckpt_to_keep,
            )
            torch_module.distributed.barrier()
        finally:
            gc.collect()

        if getattr(self, "rank", 0) == 0:
            from verl_speco.trainer.checkpoint import format_checkpoint_memory_snapshot

            logger.warning(
                "[actor checkpoint] NPU FSDP2 CPU-shard save step=%s elapsed=%.2fs %s",
                global_step,
                time.perf_counter() - started,
                format_checkpoint_memory_snapshot(),
            )
        return result

    save_checkpoint_from_cpu._speco_npu_fsdp2_cpu_checkpoint = True
    engine_cls.save_checkpoint = save_checkpoint_from_cpu
    _NPU_FSDP2_CHECKPOINT_COMPAT_APPLIED = True
    logger.warning("Enabled NPU FSDP2 CPU-shard checkpoint compatibility for verl release/v0.8.0")
    return True


class VerlNPUVLLMImportCompatMixin:
    """Install import compatibility when WorkerDict constructs the worker."""

    def __init__(self, *args, **kwargs):
        install_verl_npu_vllm_import_compat()
        install_verl_npu_fsdp2_checkpoint_compat()
        super().__init__(*args, **kwargs)
