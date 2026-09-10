"""Thin lifecycle integration for the verl V1 PPO trainers."""

from __future__ import annotations

import logging
import json
import inspect
from typing import Any

logger = logging.getLogger(__name__)


def _plain_config(value):
    """Convert an OmegaConf/plain mapping to a JSON-safe value."""
    try:
        from omegaconf import OmegaConf

        return OmegaConf.to_container(value, resolve=True)
    except Exception:  # noqa: BLE001
        if isinstance(value, dict) or hasattr(value, "items"):
            return {str(key): _plain_config(item) for key, item in value.items()}
        return value


def _unwrap_remote(cls):
    return getattr(cls, "__ray_actor_class__", cls)


def _remotify_like(original, cls):
    import ray

    return ray.remote(cls) if hasattr(original, "__ray_actor_class__") else cls


class SpecoV1Mixin:
    """Add SPECO worker/runtime wiring while preserving the V1 PPO loop.

    Phase 1 deliberately does not alter V1's ``KVBatchMeta`` pipeline.  The
    V1 loop owns sampling and policy updates; the shared SPECO scheduler,
    collector, feature store, and publication facade are delegated to the
    legacy adapter implementation.
    """

    speco_worker_cls = None

    def _speco_init_state(self):
        from verl_speco.trainer.scheduler import DrafterRuntimeState, DrafterScheduler

        self.device_name = getattr(
            self, "device_name", self.config.trainer.get("device", "cuda")
        )
        self.drafter_wg = None
        self._drafter_scheduler = DrafterScheduler()
        self._drafter_runtime_state = DrafterRuntimeState()
        self._pending_drafter_publish_refs = None
        self._pending_drafter_checkpoint_refs = []
        self._pending_target_lm_head_sync = None
        self._speco_last_raw_drafter_samples = 0
        self._speco_last_collected_samples = 0
        self._speco_last_oldlogprob_candidate_samples = 0
        self._speco_last_oldlogprob_short_response_skipped = 0
        self._speco_last_oldlogprob_planned_samples = 0
        self._speco_last_oldlogprob_collected_samples = 0
        self._speco_last_oldlogprob_collected_rows = 0
        self._speco_last_oldlogprob_payload_mib = 0.0
        self._speco_last_oldlogprob_select_elapsed_sec = 0.0
        self._speco_last_oldlogprob_sp_merge_elapsed_sec = 0.0
        self._speco_last_oldlogprob_concat_elapsed_sec = 0.0
        self._speco_last_oldlogprob_cpu_copy_elapsed_sec = 0.0
        self._speco_last_oldlogprob_ray_put_elapsed_sec = 0.0
        self._speco_last_oldlogprob_prepare_elapsed_sec = 0.0
        self._speco_last_oldlogprob_compute_elapsed_sec = 0.0
        self._speco_last_oldlogprob_collect_elapsed_sec = 0.0
        self._speco_last_oldlogprob_collect_rpc_elapsed_sec = 0.0
        self._speco_last_oldlogprob_total_elapsed_sec = 0.0
        self._speco_last_collect_interval_matched = 0
        self._speco_last_collection_outcome = None

    def __getattr__(self, name):
        """Lazily expose the stable SPECO facade implemented by the legacy adapter.

        The V1 trainer owns the PPO loop, while the scheduler, feature-store,
        and worker RPC facade are shared with the legacy trainer.  Lazy binding
        avoids importing the legacy Ray trainer during V1 module discovery and
        keeps one implementation of the runtime protocol.
        """
        if (
            name.startswith("_speco")
            or name.startswith("speco_")
            or name.startswith("is_drafter_")
            or name
            in {
                "_ray_get_if_needed",
                "_first_non_null",
                "_require_speco_worker_group",
            }
        ):
            from verl_speco.trainer.speco_ray_trainer import SpecoRayPPOTrainer

            descriptor = inspect.getattr_static(SpecoRayPPOTrainer, name, None)
            if descriptor is not None:
                return descriptor.__get__(self, type(self))
        raise AttributeError(name)

    @staticmethod
    def _speco_flatten_checkpoint_results(value: Any) -> list[dict[str, Any]]:
        if isinstance(value, dict):
            return [value]
        if isinstance(value, (list, tuple)):
            flattened: list[dict[str, Any]] = []
            for item in value:
                flattened.extend(SpecoV1Mixin._speco_flatten_checkpoint_results(item))
            return flattened
        return []

    def attach_speco_worker_group(self, worker_group):
        """Bind V1 drafter workers to the shared SPECO adapter facade."""
        from verl_speco.trainer.speco_ray_trainer import SpecoRayPPOTrainer

        return SpecoRayPPOTrainer.attach_speco_worker_group(self, worker_group)

    @staticmethod
    def _speco_online_enabled_from_config(config) -> bool:
        drafter = config.actor_rollout_ref.rollout.get("drafter", {}) or {}
        return bool(drafter.get("enable", False) and drafter.get("enable_drafter_training", False))

    @staticmethod
    def _speco_validate_v1_training_config(config) -> None:
        drafter = config.actor_rollout_ref.rollout.get("drafter", {}) or {}
        training = drafter.get("training", {}) or {}
        if not bool(drafter.get("enable", False)) or not bool(
            drafter.get("enable_drafter_training", False)
        ):
            return
        if bool(training.get("collect_hidden_states_from_sgl", False)):
            raise ValueError(
                "verl V1 online SPECO training currently requires "
                "collect_hidden_states_from_sgl=false; use old-logprob collection"
            )
        if not bool(training.get("collect_hidden_states_from_old_logprob", False)):
            raise ValueError(
                "verl V1 online SPECO training requires "
                "training.collect_hidden_states_from_old_logprob=true"
            )
        rollout_correction = config.algorithm.get("rollout_correction", None)
        if rollout_correction and bool(rollout_correction.get("bypass_mode", False)):
            raise ValueError(
                "Phase-1 V1 online SPECO training is incompatible with "
                "algorithm.rollout_correction.bypass_mode=true because hidden "
                "states must be collected from actor old-logprob inference"
            )
        if str(config.trainer.v1.get("trainer_mode", "sync")).lower() != "sync":
            raise ValueError(
                "Phase-1 V1 online SPECO training supports trainer.v1.trainer_mode=sync only. "
                "The asynchronous V1 modes remain available for fixed-drafter serving."
            )
        if str(config.actor_rollout_ref.rollout.get("name", "")).lower() != "vllm":
            raise ValueError(
                "Phase-1 V1 online SPECO training supports rollout.name=vllm only"
            )

    def _setup(self):
        from verl_speco.integration.vllm_runtime import (
            configure_vllm_runtime_from_config,
        )

        self._speco_validate_v1_training_config(self.config)
        self._speco_init_state()
        if self._speco_online_enabled_from_config(self.config):
            # Resolve a resumed draft checkpoint before V1 creates the rollout
            # server.  The server reads drafter.model_path during construction.
            self._speco_prepare_drafter_checkpoint_for_worker_init()

        configure_vllm_runtime_from_config(self.config)
        self._speco_drafter_config_for_worker_wrap = self.config.actor_rollout_ref.rollout.get(
            "drafter", None
        )
        # SPECO translates this extension field into vLLM-native
        # ``engine_kwargs.vllm.speculative_config`` above.  Keep the source
        # field present as well: custom/third-party V1 launchers may read it,
        # and removing it here would make their rollout replicas silently lose
        # speculative decoding.
        result = super()._setup()
        if self._speco_online_enabled_from_config(self.config):
            # The actor/rollout workers and their placement group now exist, so
            # the drafter can safely share their bundles before the first
            # checkpoint-manager weight update in on_init_end().
            self._init_v1_speco_drafter_workers()
        self._speco_v1_state = {"drafter_version": None, "features_collected": 0}
        return result

    def _init_resource_pool_mgr(self):
        result = super()._init_resource_pool_mgr()
        self._wrap_v1_actor_worker()
        return result

    def _wrap_v1_actor_worker(self):
        rollout = self.config.actor_rollout_ref.rollout
        drafter_config = getattr(self, "_speco_drafter_config_for_worker_wrap", None)
        if str(rollout.get("name", "")).lower() != "vllm":
            return
        from verl.trainer.ppo.utils import Role

        from verl_speco.integration.rollout_publish import DraftWeightPublishMixin
        from verl_speco.integration.verl_npu_vllm_compat import (
            VerlNPUVLLMImportCompatMixin,
        )

        for role, worker_cls in list(self.role_worker_mapping.items()):
            if role not in {Role.ActorRollout, Role.ActorRolloutRef}:
                continue
            raw = _unwrap_remote(worker_cls)
            bases = []
            if not issubclass(raw, VerlNPUVLLMImportCompatMixin):
                bases.append(VerlNPUVLLMImportCompatMixin)
            if (
                bool((drafter_config or rollout.get("drafter", {})).get("enable", False))
                and not issubclass(raw, DraftWeightPublishMixin)
            ):
                bases.append(DraftWeightPublishMixin)
            if not bases:
                continue
            wrapped = type(
                f"SpecoV1{raw.__name__}",
                tuple(bases) + (raw,),
                {
                    "__module__": __name__,
                    "__doc__": raw.__doc__,
                    "_speco_drafter_config_env": json.dumps(
                        _plain_config(drafter_config or rollout.get("drafter", {})), sort_keys=True
                    ),
                },
            )
            self.role_worker_mapping[role] = _remotify_like(worker_cls, wrapped)
            logger.info("SPECO V1 wrapped actor worker: %s", wrapped.__name__)

    def on_init_end(self):
        result = super().on_init_end()
        online_drafter = bool(
            self.config.actor_rollout_ref.rollout.get("drafter", {})
            .get("enable_drafter_training", False)
        )
        logger.info(
            "SPECO V1 trainer initialized: mode=%s, online_drafter=%s",
            self.trainer_mode,
            online_drafter,
        )
        return result

    def on_step_end(self):
        result = super().on_step_end()
        if getattr(self, "_speco_v1_pending_training", False):
            publish_metrics = self._speco_publish_drafter_weights(
                True, getattr(self, "_speco_v1_training_plan", None), after_weight_update=True
            )
            self._pending_sync_metrics = {
                **(getattr(self, "_pending_sync_metrics", None) or {}),
                **publish_metrics,
            }
            self._speco_v1_pending_training = False
        return result

    def _save_checkpoint(self):
        if self._speco_online_enabled_from_config(self.config):
            self._speco_wait_pending_drafter_publish()
            self._speco_save_drafter_checkpoint(wait=True)
        return super()._save_checkpoint()

    def on_sample_end(self):
        return super().on_sample_end()

    def _init_v1_speco_drafter_workers(self):
        from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
        from verl.trainer.ppo.utils import Role
        from verl_speco.workers import SpecoWorker

        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
        worker_cls = self.speco_worker_cls or SpecoWorker
        remote_worker_cls = (
            worker_cls if hasattr(worker_cls, "__ray_actor_class__") else __import__("ray").remote(worker_cls)
        )
        drafter_cls = RayClassWithInitArgs(
            cls=remote_worker_cls,
            config=self.config.actor_rollout_ref,
            role="drafter",
            device_name=self.config.trainer.device,
        )
        worker_group = RayWorkerGroup(
            resource_pool=resource_pool,
            ray_cls_with_init=drafter_cls,
            name_prefix="speco_v1_drafter",
            device_name=self.config.trainer.device,
        )
        worker_group.init_model()
        self.attach_speco_worker_group(worker_group)

    def _speco_v1_batch_data(self, batch):
        import transfer_queue as tq
        import torch
        from verl.protocol import DataProto

        fields = ["prompts", "responses", "input_ids", "response_mask", "position_ids"]
        if bool(self.config.actor_rollout_ref.rollout.get("calculate_log_probs", False)):
            fields.append("rollout_log_probs")
        if bool(
            self.config.actor_rollout_ref.rollout.get(
                "enable_rollout_routing_replay", False
            )
        ):
            fields.append("routed_experts")
        data = tq.kv_batch_get(
            keys=batch.keys, partition_id=batch.partition_id, select_fields=fields
        )
        input_ids = data["input_ids"]
        if "position_ids" not in data.keys():
            # Standard V1 agent loops persist position ids.  This fallback is
            # only for a custom text loop that omits them.
            input_padded = input_ids.to_padded_tensor(padding=0)
            sequence_width = input_padded.shape[1]
            data["position_ids"] = torch.arange(
                sequence_width, device=input_ids.device
            ).expand(input_padded.shape[0], -1)
        if "attention_mask" not in data.keys():
            lengths = input_ids.offsets().diff()
            width = int(lengths.max().item()) if lengths.numel() else 0
            positions = torch.arange(width, device=input_ids.device).unsqueeze(0)
            data["attention_mask"] = (positions < lengths.unsqueeze(1)).to(torch.int64)
        return data, DataProto(batch=data.to_padded_tensor())

    def _compute_old_log_prob(self, batch, metrics):
        if not self._speco_oldlogprob_collection_enabled():
            return super()._compute_old_log_prob(batch, metrics)

        import transfer_queue as tq
        import torch
        from verl.utils import tensordict_utils as tu
        from verl.workers.utils.padding import (
            left_right_2_no_padding,
            no_padding_2_padding,
            response_to_nested,
        )
        from verl_speco.integration.oldlogprob_runtime import (
            OLD_LOGPROB_AUX_LAYER_IDS_KEY,
            OLD_LOGPROB_COLLECT_MASK_KEY,
            OLD_LOGPROB_HIDDEN_CAPTURE_IMPL_KEY,
            OLD_LOGPROB_HIDDEN_LAYOUT_KEY,
            OLD_LOGPROB_HIDDEN_OBJECT_REF_KEY,
            OLD_LOGPROB_HIDDEN_POSITION_MASK_KEY,
            OLD_LOGPROB_HIDDEN_POSITIONS_KEY,
            OLD_LOGPROB_OWNER_RANK_KEY,
        )

        nested_data, data_proto = self._speco_v1_batch_data(batch)
        collect_plan = self._speco_build_oldlogprob_collect_plan(data_proto)
        if collect_plan is None:
            return super()._compute_old_log_prob(batch, metrics)

        control = data_proto.to_tensordict()
        control = left_right_2_no_padding(control)
        control[OLD_LOGPROB_COLLECT_MASK_KEY] = collect_plan["collect_mask"]
        control[OLD_LOGPROB_HIDDEN_POSITIONS_KEY] = collect_plan["hidden_positions"]
        control[OLD_LOGPROB_HIDDEN_POSITION_MASK_KEY] = collect_plan["hidden_position_mask"]
        control[OLD_LOGPROB_OWNER_RANK_KEY] = collect_plan["owner_rank"]
        tu.assign_non_tensor_data(
            control,
            OLD_LOGPROB_AUX_LAYER_IDS_KEY,
            self._speco_oldlogprob_aux_layer_ids(),
        )
        tu.assign_non_tensor_data(
            control,
            OLD_LOGPROB_HIDDEN_CAPTURE_IMPL_KEY,
            self._speco_oldlogprob_hidden_capture_impl(),
        )
        tu.assign_non_tensor_data(
            control,
            OLD_LOGPROB_HIDDEN_LAYOUT_KEY,
            self._speco_oldlogprob_hidden_layout(),
        )
        tu.assign_non_tensor_data(control, OLD_LOGPROB_HIDDEN_OBJECT_REF_KEY, True)
        actor_megatron_cfg = self.config.actor_rollout_ref.actor.get("megatron", {})
        tu.assign_non_tensor_data(
            control,
            "speco_oldlogprob_sp_disabled",
            not bool(actor_megatron_cfg.get("sequence_parallel", True)),
        )
        tu.assign_non_tensor(
            control,
            # V1 always computes entropy in this stage.  Preserve that contract
            # even though the legacy adapter permits disabling it.
            calculate_entropy=True,
            compute_loss=False,
            temperature=self.config.actor_rollout_ref.rollout.temperature,
        )
        output = self.actor_rollout_wg.compute_log_prob(control)
        output_data = output
        collected = self._speco_collect_oldlogprob_features(
            data_proto, collect_plan, output_data
        )
        self._speco_v1_state["features_collected"] = int(
            self._speco_v1_state.get("features_collected", 0) + collected
        )
        collection_plan_data = collect_plan["collection_plan"]
        collection_outcome = getattr(self, "_speco_last_collection_outcome", None)
        metrics.update(
            {
                "drafter/oldlogprob_candidate_samples": int(
                    getattr(self, "_speco_last_oldlogprob_candidate_samples", 0)
                ),
                "drafter/oldlogprob_short_response_skipped": int(
                    getattr(self, "_speco_last_oldlogprob_short_response_skipped", 0)
                ),
                "drafter/oldlogprob_planned_samples": int(
                    getattr(self, "_speco_last_oldlogprob_planned_samples", 0)
                ),
                "drafter/oldlogprob_collected_samples": int(
                    getattr(self, "_speco_last_oldlogprob_collected_samples", 0)
                ),
                "drafter/oldlogprob_collected_rows": int(
                    getattr(self, "_speco_last_oldlogprob_collected_rows", 0)
                ),
            }
        )
        metrics.update(collection_plan_data.metrics())
        if collection_outcome is not None:
            metrics.update(collection_outcome.metrics())
            worker_versions = [
                (
                    result.worker_id,
                    result.data_version,
                    result.buffer_version_before,
                    result.buffer_version_after,
                )
                for result in (collection_outcome.worker_results or [])
            ]
            logger.info(
                "SPECO V1 old-logprob collection step=%s collection_id=%s "
                "candidates=%s planned=%s collected=%s rows=%s reason=%s "
                "worker_versions=%s",
                collection_plan_data.source_global_step,
                collection_plan_data.collection_id,
                self._speco_last_oldlogprob_candidate_samples,
                self._speco_last_oldlogprob_planned_samples,
                self._speco_last_oldlogprob_collected_samples,
                self._speco_last_oldlogprob_collected_rows,
                collection_outcome.reason,
                worker_versions,
            )

        response_mask = data_proto.batch.get("response_mask")
        log_probs = no_padding_2_padding(output_data["log_probs"], control)
        entropy = output_data.get("entropy")
        entropy = (
            no_padding_2_padding(entropy, control)
            if entropy is not None
            else torch.zeros_like(log_probs)
        )
        nested_data["old_log_probs"] = response_to_nested(
            log_probs.float(), nested_data["response_mask"]
        )
        nested_data["entropy"] = response_to_nested(
            entropy.float(), nested_data["response_mask"]
        )
        updated_batch = tq.kv_batch_put(
            keys=batch.keys,
            partition_id=batch.partition_id,
            fields=nested_data.select("old_log_probs", "entropy"),
        )
        if bool(self.config.actor_rollout_ref.rollout.get("calculate_log_probs", False)):
            from verl.utils.debug.metrics import calculate_debug_metrics

            # The V1 batch is normally kept in TransferQueue, so construct the
            # same padded view expected by the upstream debug helper after the
            # new old-logprobs have been produced.
            data_proto.batch["old_log_probs"] = log_probs.float()
            metrics.update(calculate_debug_metrics(data_proto))
        from verl.trainer.ppo.core_algos import agg_loss

        actor_config = self.config.actor_rollout_ref.actor
        entropy_agg = agg_loss(
            loss_mat=entropy,
            loss_mask=response_mask,
            loss_agg_mode=actor_config.loss_agg_mode,
            loss_scale_factor=actor_config.loss_scale_factor,
        )
        metrics["actor/entropy"] = entropy_agg.detach().item()
        return updated_batch

    def _update_actor(self, batch, metrics):
        if not self._speco_online_enabled_from_config(self.config):
            return super()._update_actor(batch, metrics)
        event = self._speco_on_before_actor_update()
        plan = event.training_plan
        metrics.update(event.metrics or {})
        result = super()._update_actor(batch, metrics)
        pending_sync = getattr(self, "_pending_target_lm_head_sync", None)
        if pending_sync is not None:
            metrics.update(self._speco_finish_target_lm_head_weight_sync(pending_sync))
            self._pending_target_lm_head_sync = None
        if plan is not None:
            if plan.launch:
                trained, train_metrics = self._speco_train_drafter(plan)
            else:
                trained = False
                train_metrics = {
                    "drafter/trained": 0,
                    "drafter/train_successful_steps_max": 0,
                    "drafter/train_no_trainable_batch": int(
                        plan.reason == "no_trainable_batch"
                    ),
                    "drafter/train_activation_failed": 0,
                }
            metrics.update(train_metrics)
            self._speco_v1_pending_training = bool(trained)
            self._speco_v1_training_plan = plan
        return result

    def fit(self, agent_loop_manager):
        try:
            if self._speco_online_enabled_from_config(self.config):
                self._speco_activate_drafter_training_model_before_fit()
            return super().fit(agent_loop_manager)
        finally:
            if self._speco_online_enabled_from_config(self.config):
                self._speco_wait_pending_drafter_publish()
                self._speco_wait_pending_drafter_checkpoint()

    def speco_v1_status(self) -> dict[str, Any]:
        """Return diagnostics used by smoke tests and startup logging."""

        return {
            "trainer_mode": str(self.trainer_mode),
            "adapter": type(self).__name__,
            "features_collected": int(
                getattr(self, "_speco_v1_state", {}).get("features_collected", 0)
            ),
        }
