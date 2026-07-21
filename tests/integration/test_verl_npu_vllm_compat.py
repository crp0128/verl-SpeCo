from __future__ import annotations

import importlib
import sys
import types

from verl_speco.integration import verl_npu_vllm_compat as compat


def test_factory_fused_moe_survives_verl_npu_patch_import(monkeypatch) -> None:
    vllm = types.ModuleType("vllm")
    vllm.__version__ = "0.23.0"
    fused_moe_module = types.ModuleType(compat._VLLM_FUSED_MOE_MODULE)

    def fused_moe_factory(*args, **kwargs):
        return args, kwargs

    fused_moe_module.FusedMoE = fused_moe_factory
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "torch_npu", types.ModuleType("torch_npu"))
    monkeypatch.setitem(sys.modules, "vllm_ascend", types.ModuleType("vllm_ascend"))
    monkeypatch.setitem(sys.modules, compat._VLLM_FUSED_MOE_MODULE, fused_moe_module)
    monkeypatch.setattr(compat, "_IMPORT_COMPAT_APPLIED", False)

    def module_importer(module_name: str):
        if module_name == compat._VERL_NPU_VLLM_PATCH_MODULE:
            assert hasattr(fused_moe_factory, "weight_loader")
            original = fused_moe_factory.weight_loader

            def wrapped_weight_loader(*args, **kwargs):
                return original(*args, **kwargs)

            fused_moe_factory.weight_loader = wrapped_weight_loader
            module = types.ModuleType(module_name)
            monkeypatch.setitem(sys.modules, module_name, module)
            return module
        return importlib.import_module(module_name)

    assert compat.install_verl_npu_vllm_import_compat(module_importer) is True
    assert not hasattr(fused_moe_factory, "weight_loader")
    assert compat._IMPORT_COMPAT_APPLIED is True


def test_worker_mixin_installs_compat_before_base_init(monkeypatch) -> None:
    events = []
    monkeypatch.setattr(compat, "install_verl_npu_vllm_import_compat", lambda: events.append("compat"))
    monkeypatch.setattr(
        compat,
        "install_verl_npu_fsdp2_checkpoint_compat",
        lambda: events.append("checkpoint"),
    )

    class BaseWorker:
        def __init__(self):
            events.append("base")

    class WrappedWorker(compat.VerlNPUVLLMImportCompatMixin, BaseWorker):
        pass

    WrappedWorker()
    assert events == ["compat", "checkpoint", "base"]


def test_npu_fsdp2_checkpoint_saves_existing_cpu_shards(monkeypatch) -> None:
    events = []
    engine_module = types.ModuleType(compat._VERL_FSDP_ENGINE_MODULE)

    class FSDPEngine:
        def save_checkpoint(self, *args, **kwargs):
            events.append(("original", args, kwargs))

    engine_module.FSDPEngine = FSDPEngine
    device_module = types.SimpleNamespace(get_device_name=lambda: "npu")
    fsdp_utils = types.SimpleNamespace(fsdp_version=lambda module: module.fsdp_version)
    torch_module = types.SimpleNamespace(
        distributed=types.SimpleNamespace(barrier=lambda: events.append(("barrier",)))
    )
    modules = {
        compat._VERL_FSDP_ENGINE_MODULE: engine_module,
        "verl.utils.device": device_module,
        "verl.utils.fsdp_utils": fsdp_utils,
        "torch": torch_module,
    }
    monkeypatch.setitem(sys.modules, "torch_npu", types.ModuleType("torch_npu"))
    monkeypatch.setattr(compat, "_NPU_FSDP2_CHECKPOINT_COMPAT_APPLIED", False)

    assert compat.install_verl_npu_fsdp2_checkpoint_compat(modules.__getitem__) is True

    class Model:
        fsdp_version = 2

        @staticmethod
        def parameters():
            parameter = types.SimpleNamespace(device=types.SimpleNamespace(type="cpu"))
            return iter([parameter])

    class CheckpointManager:
        should_save_optimizer = True

        @staticmethod
        def save_checkpoint(**kwargs):
            events.append(("manager", kwargs))
            return "saved"

    engine = FSDPEngine()
    engine.rank = 1
    engine.module = Model()
    engine.checkpoint_manager = CheckpointManager()
    engine.optimizer = types.SimpleNamespace(state={})
    engine._is_offload_param = True
    engine._is_offload_optimizer = True
    engine._uses_fsdp2_cpu_offload_policy = False

    assert engine.save_checkpoint("/tmp/actor", global_step=20, max_ckpt_to_keep=1) == "saved"
    assert [event[0] for event in events] == ["manager", "barrier"]

    engine.module.fsdp_version = 1
    engine.save_checkpoint("/tmp/actor", global_step=40)
    assert events[-1][0] == "original"

    engine.module.fsdp_version = 2
    engine._is_offload_optimizer = False
    engine.save_checkpoint("/tmp/actor", global_step=60)
    assert events[-1][0] == "original"
