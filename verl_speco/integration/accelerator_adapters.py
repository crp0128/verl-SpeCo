# Licensed under the Apache License, Version 2.0. See LICENSE.
"""Consolidated multi-node capabilities. Hardware adapters load only when explicitly selected."""
from __future__ import annotations
import copy
import os
from types import MethodType
from vllm_ascend.worker.worker import NPUWorker
from verl_speco.integration.accelerator_speculation import install_dspark_k_query_eager
from verl_speco.integration.accelerator_speculation import install_speco_92b_eagle3_target as install_speco_92b_eagle3_target

class SpecoMultiNodeDSparkWorker(NPUWorker):

    def load_model(self):
        import vllm_ascend.worker.model_runner_v1 as runner_module
        from vllm.model_executor.models.registry import ModelRegistry
        runner = self.model_runner
        spec = runner.speculative_config
        if spec is None or spec.method != 'dflash' or (not spec.enforce_eager):
            raise ValueError('dedicated target model DSpark worker requires an eager DFlash carrier')
        config = spec.draft_model_config.hf_config
        layers = tuple(config.target_layer_ids)
        if tuple(config.eagle_aux_hidden_state_layer_ids) != tuple((index + 1 for index in layers)):
            raise ValueError('target model DSpark checkpoint auxiliary indexing differs from decoder output identity')
        if config.block_size != spec.num_speculative_tokens:
            raise ValueError('target model DSpark trained block and verified K differ')
        if spec.max_model_len is None or spec.max_model_len > config.sliding_window:
            raise ValueError('initial target model DSpark functional rollout requires context within trained SWA window')
        architecture = config.architectures[0]
        previous_entry = ModelRegistry.models.get(architecture)
        ModelRegistry.register_model(architecture, MultiNodeDSparkForCausalLM)
        original_loader = runner_module.get_model
        runner.use_aux_hidden_state_outputs = True

        def keep_trainable_embedding(proposer, _target_language_model):
            if not proposer.model.has_own_embed_tokens or 'embed_tokens.weight' not in proposer.model._speco_loaded_parameter_names:
                raise RuntimeError('DSpark requires its loaded independent trainable embedding')

        def keep_dense_draft_topk(proposer, _target_language_model):
            if hasattr(proposer.model.model, 'topk_indices_buffer'):
                raise RuntimeError('unexpected MoE routing buffer in dense DSpark draft')
        runner.drafter._maybe_share_embeddings = MethodType(keep_trainable_embedding, runner.drafter)
        runner.drafter._maybe_share_topk_indices = MethodType(keep_dense_draft_topk, runner.drafter)

        def loaded_target(*args, **kwargs):
            model = original_loader(*args, **kwargs)
            if hasattr(self, '_speco_aux_target'):
                raise RuntimeError('unexpected target loader reentry')
            adapter, restore = install_speco_92b_eagle3_target(model, layer_ids=layers)
            from verl_speco.integration.accelerator_speculation import install_mome_speculation_history
            convolutions = [module for module in model.model.modules() if type(module).__name__ == 'MomeConv']
            if not model.model.cfg.use_mome:
                raise RuntimeError('92B DSpark target requires its trained MoME state')
            self._speco_native_decode_math = os.environ.get('SPECO_DSPARK_NATIVE_DECODE_MATH') == '1'
            if self._speco_native_decode_math:
                from verl_speco.integration.accelerator_speculation import install_mome_native_decode_math
                self._speco_restore_native_decode_math = install_mome_native_decode_math(convolutions)
            self._speco_mome_histories, self._speco_restore_mome_histories = install_mome_speculation_history(convolutions, max_rejected_tokens=spec.num_speculative_tokens)
            native_layers = tuple((index + 1 for index in layers))
            select_output_layers = model.set_aux_hidden_state_layers

            def select(_model, requested):
                if tuple(requested) != native_layers:
                    raise ValueError('native runner selected foreign auxiliary layers')
                select_output_layers(layers)
            model.set_aux_hidden_state_layers = MethodType(select, model)
            model.get_eagle3_default_aux_hidden_state_layers = MethodType(lambda _model: native_layers, model)
            if not hasattr(model.config, 'image_token_index'):
                model.config.image_token_index = model.config.image_token_id
            self._speco_aux_target = adapter
            self._speco_restore_aux_target = restore
            return model
        runner_module.get_model = loaded_target
        try:
            super().load_model()
        finally:
            runner_module.get_model = original_loader
            if previous_entry is None:
                ModelRegistry.models.pop(architecture, None)
            else:
                ModelRegistry.models[architecture] = previous_entry
        if runner.drafter.model.lm_head is not runner.model.lm_head:
            raise RuntimeError('DSpark must bind the actual target TP-local LM head')
        if runner.drafter.model.model.embed_tokens is runner.model.model.embed_tokens:
            raise RuntimeError('DSpark trainable embedding aliased target storage')
        from vllm_ascend.utils import lmhead_tp_enable
        original_merged = runner.drafter._run_merged_draft
        original_dummy = runner.drafter.dummy_run
        self._speco_dspark_adapter, self._speco_restore_dspark_proposer = install_dspark_k_query_eager(runner.drafter, block_size=config.block_size, loaded_parameter_names=runner.drafter.model._speco_loaded_parameter_names, lmhead_tp_enabled=lmhead_tp_enable())
        actual_proposal = runner.drafter._runnable
        self._speco_proposal_calls = 0
        self._speco_proposed_tokens = 0

        def propose(_proposer, **kwargs):
            result = actual_proposal(**kwargs)
            self._speco_proposal_calls += 1
            self._speco_proposed_tokens += result.numel()
            return result
        runner.drafter._runnable = MethodType(propose, runner.drafter)

        def dummy(_proposer, *args, **kwargs):
            proxy = copy.copy(_proposer)
            proxy._runnable = MethodType(original_merged.__func__, proxy)
            return original_dummy.__func__(proxy, *args, **kwargs)
        runner.drafter.dummy_run = MethodType(dummy, runner.drafter)
        from verl_speco.integration.multinode_publication import initialize_worker_publication
        initialize_worker_publication(self)

    def speco_dspark_runtime_status(self):
        runner = self.model_runner
        return dict(rank=self.rank, worker_incarnation=self._speco_serving_incarnation, draft_revision=self._speco_draft_revision, K=self._speco_dspark_adapter.block_size, auxiliary_layers=list(self._speco_aux_target.layer_ids), own_draft_embedding=runner.drafter.model.has_own_embed_tokens, draft_embedding_not_shared=runner.drafter.model.model.embed_tokens is not runner.model.model.embed_tokens, target_lm_head_bound=runner.drafter.model.lm_head is runner.model.lm_head, loaded_parameters=sorted(runner.drafter.model._speco_loaded_parameter_names), actual_proposal_calls=self._speco_proposal_calls, actual_proposed_tokens=self._speco_proposed_tokens, mome_convolutions=len(self._speco_mome_histories), mome_rollback_calls=sum((history.rollback_calls for history in self._speco_mome_histories)), mome_rolled_back_tokens=sum((history.rolled_back_tokens for history in self._speco_mome_histories)), native_decode_math_diagnostic=self._speco_native_decode_math, native_decode_math_rows=sum((getattr(history.conv, '_speco_native_decode_rows', 0) for history in self._speco_mome_histories)), eager_only=True, full_training_or_publish_not_claimed=True)

    def speco_accepted_query_arm(self, *, request_id, prompt_token_ids):
        if self._speco_draft_update_phase != 'idle':
            raise RuntimeError('accepted rollout is blocked by pending draft publication')
        from vllm.forward_context import get_forward_context
        from verl_speco.integration.multinode_online import AcceptedTargetQueryProbe
        if not hasattr(self, '_speco_accepted_probe'):
            self._speco_accepted_probe = AcceptedTargetQueryProbe(self.model_runner, metadata_supplier=lambda: get_forward_context().attn_metadata)
        self._speco_accepted_probe.arm(request_id=request_id, prompt_token_ids=prompt_token_ids, layer_ids=self._speco_aux_target.layer_ids, hidden_size=self._speco_aux_target.hidden_size, max_capture_tokens=1024 + 8, max_query_tail=self._speco_dspark_adapter.block_size)
        return dict(rank=self.rank, armed=True, request_id=request_id, single_request_eager_only=True)

    def speco_accepted_query_finish(self, *, generated_token_ids):
        if not hasattr(self, '_speco_accepted_probe'):
            raise RuntimeError('accepted query probe was not initialized')
        return dict(rank=self.rank, **self._speco_accepted_probe.finish(generated_token_ids))

    def speco_accepted_query_release(self):
        if not hasattr(self, '_speco_accepted_probe'):
            raise RuntimeError('accepted query probe was not initialized')
        self._speco_accepted_probe.release()
        return dict(rank=self.rank, released=True)

    def speco_accepted_query_publish(self, **settings):
        from verl_speco.integration.accepted_publication import publish_accepted_query
        return publish_accepted_query(self, **settings)

    def speco_admit_online_collection(self, *, sample_count, payload_bytes_per_sample, producer_endpoint, **settings):
        if self._speco_draft_update_phase != 'idle' or getattr(self, '_speco_target_update_phase', 'idle') != 'idle':
            raise RuntimeError('online admission is blocked by pending weight publication')
        if self.rank != 0:
            return dict(rank=self.rank, admitted=True, source_owner=False)
        from verl_speco.integration.multinode_online import online_publisher_context
        from verl_speco.integration.multinode_online import admit_pool_collection
        context = online_publisher_context(self)
        context._speco_get_tq_publisher(producer_endpoint=producer_endpoint, **settings)
        context.speco_native_reap_sources()
        pool = context._speco_native_store.transport
        if pool.leases:
            raise RuntimeError('online admission requires the previous collection to be drained')
        receipt = admit_pool_collection(pool, sample_count=sample_count, payload_bytes_per_sample=payload_bytes_per_sample, capacity_bytes=settings['capacity_bytes'])
        return dict(receipt, rank=self.rank, admitted=True, source_owner=True)

    def speco_accepted_query_abort(self):
        probe = getattr(self, '_speco_accepted_probe', None)
        if probe is not None:
            probe.close()
        return dict(rank=self.rank, capture_aborted=True)

    def speco_online_reap_sources(self):
        context = getattr(self, '_speco_online_publisher_context', None)
        if context is None:
            return dict(rank=self.rank, native_retained_sources=0)
        return dict(rank=self.rank, **context.speco_native_reap_sources())

    def speco_prepare_draft_update(self, **request):
        from verl_speco.integration.multinode_publication import prepare_worker_draft_update
        return prepare_worker_draft_update(self, **request)

    def speco_commit_draft_update(self, **request):
        from verl_speco.integration.multinode_publication import commit_worker_draft_update
        return commit_worker_draft_update(self, **request)

    def speco_finish_draft_update(self, **request):
        from verl_speco.integration.multinode_publication import finish_worker_draft_update
        return finish_worker_draft_update(self, **request)

    def speco_close_online_publisher(self):
        context = getattr(self, '_speco_online_publisher_context', None)
        if context is None:
            return dict(rank=self.rank, client_closed=False)
        return dict(rank=self.rank, **context.speco_close_tq_client())
import asyncio
import torch
from verl_speco.integration.multinode_publication import TargetWireDigest, load_received_target_buckets
from verl_speco.integration.multinode_publication import reset_native_weight_coverage, verify_native_weight_coverage

class SpecoMultiNodeTargetUpdateWorker(SpecoMultiNodeDSparkWorker):

    def speco_colocated_native_placement(self):
        from verl_speco.integration.multinode_recovery import observe_native_placement
        return observe_native_placement(self)

    def speco_install_joint_native_restore(self, *, config, binding, owners):
        """Restore RPCs use metadata-only names over vLLM's msgpack channel."""
        from verl_speco.integration.multinode_recovery import install_joint_native_restore
        return install_joint_native_restore(self, config=config, binding=binding, owners=owners)

    def speco_audit_joint_native_draft(self):
        from verl_speco.integration.multinode_recovery import audit_joint_native_draft
        return audit_joint_native_draft(self)

    def speco_release_joint_native_restore(self, *, proof):
        from verl_speco.integration.multinode_recovery import release_joint_native_restore
        return release_joint_native_restore(self, proof=proof)

    def speco_install_checkpoint_channel(self, *, identity):
        """Named RPC keeps the engine transport free of Python callables."""
        from verl_speco.integration.multinode_publication import install_native_checkpoint_channel
        return install_native_checkpoint_channel(self, identity=identity)

    def speco_connect_checkpoint_channel(self, *, identity, master_metadata):
        from verl_speco.integration.multinode_publication import connect_native_checkpoint_channel
        return connect_native_checkpoint_channel(self, identity=identity, master_metadata=master_metadata)

    def speco_close_checkpoint_channel(self, *, identity):
        from verl_speco.integration.multinode_publication import close_native_checkpoint_channel
        return close_native_checkpoint_channel(self, identity=identity)

    def load_model(self):
        super().load_model()
        self._speco_target_revision = 0
        self._speco_target_update_phase = 'idle'
        self._speco_target_update = None

    def execute_model(self, *args, **kwargs):
        if self._speco_target_update_phase != 'idle':
            raise RuntimeError('native target generation is fenced during full weight publication')
        return super().execute_model(*args, **kwargs)

    def speco_prepare_target_transfer(self, *, manifest, bucket_bytes):
        from verl.checkpoint_engine.hccl_checkpoint_engine import HCCLCheckpointEngine
        from verl_speco.integration.multinode_online import install_hccl_checkpoint_compat
        install_hccl_checkpoint_compat()
        probe = getattr(self, '_speco_accepted_probe', None)
        if manifest['run_id'] != os.environ.get('SPECO_RUN_ID') or manifest['generation'] != int(os.environ.get('SPECO_GENERATION', '-1')) or manifest['target_revision'] != self._speco_target_revision + 1 or (self._speco_target_update_phase != 'idle') or (self._speco_draft_update_phase != 'idle') or (probe is not None and (probe.capture is not None or probe.finished_features is not None)):
            raise ValueError('target publication requires an owned released collection safe point')
        digest = TargetWireDigest(manifest)
        if len(manifest['tensors']) != 35869 or type(bucket_bytes) is not int or bucket_bytes < max((row['nbytes'] for row in manifest['tensors'])):
            raise ValueError('target wire does not cover the complete audited 92B state')
        if os.environ.get('PANGU_92B_EXPERT_SLOTS_EAGER', '') in ('1', 'true', 'True'):
            raise ValueError('native bucketed target loading cannot use one-shot expert coverage')
        self._speco_target_update_phase = 'preparing'
        try:
            engine = HCCLCheckpointEngine(bucket_size=bucket_bytes, group_name=manifest['operation_id'], rebuild_group=True, is_master=False)
            engine.prepare()
            self._speco_target_update = dict(manifest=manifest, digest=digest, engine=engine, receipt=None)
            self._speco_target_update_phase = 'prepared'
            return dict(rank=self.rank, worker_incarnation=self._speco_serving_incarnation, header_sha256=manifest['header_sha256'], prepared=True)
        except BaseException:
            self._speco_target_update_phase = 'failed'
            raise

    def speco_init_target_transfer(self, *, operation_id, master_metadata, rollout_world_size):
        from verl.checkpoint_engine.hccl_checkpoint_engine import MasterMetadata
        pending = self._speco_target_update
        if self._speco_target_update_phase != 'prepared' or pending['manifest']['operation_id'] != operation_id:
            raise RuntimeError('native HCCL target topology is foreign or uncertain')
        pending['engine'].init_process_group(rank=self.rank + 1, world_size=rollout_world_size + 1, master_metadata=MasterMetadata(**master_metadata))
        self._speco_target_update_phase = 'connected'
        return dict(rank=self.rank, worker_incarnation=self._speco_serving_incarnation, connected=True)

    def speco_receive_target_transfer(self, *, operation_id):
        pending = self._speco_target_update
        if self._speco_target_update_phase != 'connected' or pending['manifest']['operation_id'] != operation_id:
            raise RuntimeError('native target loading lacks a connected owned stream')
        self._speco_target_update_phase = 'applying'
        model = self.model_runner.model
        try:
            from verl_speco.integration.accelerator_speculation import prepare_native_target_reload, finish_native_target_reload
            if not self.model_runner.model_config.enforce_eager:
                raise RuntimeError('native target layout reload currently needs the explicit eager functional recipe')
            layout = prepare_native_target_reload(model)
            reset_native_weight_coverage(model)
            stream = pending['engine'].receive_weights(global_steps=pending['manifest']['rl_step'])
            receipt = asyncio.run(load_received_target_buckets(stream, model, pending['digest']))
            receipt.update(verify_native_weight_coverage(model))
            from vllm.model_executor.model_loader.utils import process_weights_after_loading
            process_weights_after_loading(model, self.model_runner.model_config, self.model_runner.device)
            model.post_weight_load()
            finish_native_target_reload(model, layout)
            receipt['native_expert_loader_layout_restored_and_reprocessed'] = True
            torch.npu.synchronize()
            pending['receipt'] = receipt
            self._speco_target_update_phase = 'loaded'
            return dict(receipt, rank=self.rank, worker_incarnation=self._speco_serving_incarnation, loaded=True)
        except BaseException:
            self._speco_target_update_phase = 'failed'
            raise

    def speco_commit_target_transfer(self, *, operation_id, actor_weight_manifest_sha256):
        """Called only after real scheduler prefix reset and all-stream consensus."""
        from verl_speco.integration.multinode_publication import invalidate_draft_caches
        pending = self._speco_target_update
        if self._speco_target_update_phase != 'loaded' or pending['manifest']['operation_id'] != operation_id or pending['receipt']['actor_weight_manifest_sha256'] != actor_weight_manifest_sha256:
            raise RuntimeError('native target commit lacks matching actual full weight bytes')
        self._speco_target_update_phase = 'committing'
        try:
            cleared = invalidate_draft_caches(self)
            sinks = [module for module in self.model_runner.model.modules() if getattr(module, 'sink_len', 0) > 0 and hasattr(module, 'sink_populated')]
            if len(sinks) != 46 or any((module.sink_populated or module.sink_key is None or module.sink_value is None for module in sinks)):
                raise RuntimeError('updated target did not refresh all actual parameter sinks')
            runner = self.model_runner
            if runner.drafter.model.lm_head is not runner.model.lm_head:
                raise RuntimeError('native draft is not bound to the updated target LM head')
            torch.npu.synchronize()
            manifest = pending['manifest']
            receipt = dict(run_id=manifest['run_id'], generation=manifest['generation'], rl_step=manifest['rl_step'], target_revision=manifest['target_revision'], worker_id=str(self.rank), worker_incarnation=self._speco_serving_incarnation, actor_weight_manifest_sha256=actor_weight_manifest_sha256, kv_cache_cleared=True, draft_target_head_synced=True)
            pending['ack'] = receipt
            pending['engine'].finalize()
            self._speco_target_revision = manifest['target_revision']
            self._speco_target_update_phase = 'ack_pending'
            return dict(receipt, rank=self.rank, operation_id=operation_id, applied=True, cleared_draft_cache_tensors=cleared, refreshed_parameter_sink_layers=len(sinks))
        except BaseException:
            self._speco_target_update_phase = 'failed'
            raise

    def speco_finish_target_transfer(self, *, operation_id, actor_weight_manifest_sha256):
        pending = self._speco_target_update
        if self._speco_target_update_phase != 'ack_pending' or pending['manifest']['operation_id'] != operation_id or pending['ack']['actor_weight_manifest_sha256'] != actor_weight_manifest_sha256:
            raise RuntimeError('native target resume lacks the acknowledged full-stream identity')
        self._speco_target_update_phase = 'idle'
        self._speco_target_update = None
        return dict(rank=self.rank, worker_incarnation=self._speco_serving_incarnation, released=True)
from types import MethodType, SimpleNamespace

class SpecoMultiNodeDSparkGraphWorker(SpecoMultiNodeTargetUpdateWorker):

    def load_model(self):
        from vllm.config.compilation import CUDAGraphMode
        runner = self.model_runner
        if runner.model_config.enforce_eager or runner.compilation_config.cudagraph_mode != CUDAGraphMode.FULL_DECODE_ONLY:
            raise ValueError('DSpark graph worker requires target FULL_DECODE_ONLY')
        super().load_model()
        self._speco_graph_forward_calls = 0
        self._speco_graph_live_forward_calls = 0
        self._speco_graph_replay_calls = 0
        self._install_graph_history()
        from verl_speco.integration.accelerator_speculation import install_sequence_graph_attention
        previous = os.environ.get('SPECO_SEQUENCE_FULL_DECODE')
        os.environ['SPECO_SEQUENCE_FULL_DECODE'] = '1'
        try:
            install_sequence_graph_attention()
        finally:
            if previous is None:
                os.environ.pop('SPECO_SEQUENCE_FULL_DECODE', None)
            else:
                os.environ['SPECO_SEQUENCE_FULL_DECODE'] = previous

    def _install_graph_history(self):
        import torch
        from vllm.config.compilation import CUDAGraphMode
        from vllm.forward_context import get_forward_context
        from pangu_infer.models._92b_vl import openpangu_v2_vl_text as module
        from verl_speco.integration.accelerator_speculation import speculative_graph_conv
        from verl_speco.integration.accelerator_speculation import select_target_graph_metadata
        from verl_speco.integration.accelerator_speculation import accepted_graph_contexts
        from verl_speco.integration.accelerator_speculation import mome_speculation_Window as _Window
        runner = self.model_runner
        capacity = int(runner.vllm_config.scheduler_config.max_num_seqs)
        K = self._speco_dspark_adapter.block_size
        query = K + 1
        active = {'plan': None}
        original_plan = module._build_mome_plan
        original_forward = runner._model_forward
        buffers = {}
        histories = self._speco_mome_histories
        target_layers = self._speco_aux_target.model.model.layers
        names = tuple((layer.self_attn.attn.layer_name for layer in target_layers))
        sink_lengths = {layer.self_attn.attn.sink_len for layer in target_layers}
        if sink_lengths != {128}:
            raise RuntimeError('DSpark graph requires the audited uniform target sinks')
        static = runner.compilation_config.static_forward_context
        if len(set(names)) != len(target_layers) or any((static.get(name) is not layer.self_attn.attn for name, layer in zip(names, target_layers))):
            raise RuntimeError('DSpark graph target attention ownership is not bound to loaded layers')
        self._speco_target_attention_layer_names = names
        from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
        original_graph_call = ACLGraphWrapper.__call__

        def graph_call(wrapper, *args, **kwargs):
            context = get_forward_context()
            entry = wrapper.concrete_aclgraph_entries.get(context.batch_descriptor)
            replay = active['plan'] is not None and context.cudagraph_runtime_mode == CUDAGraphMode.FULL and (wrapper.runtime_mode == CUDAGraphMode.FULL) and (entry is not None) and (entry.aclgraph is not None)
            result = original_graph_call(wrapper, *args, **kwargs)
            if replay:
                self._speco_graph_replay_calls += 1
            return result
        ACLGraphWrapper.__call__ = graph_call
        for history in histories:
            conv = history.conv
            conv._speco_graph_ring = conv.weight.new_zeros((2 * capacity, K + conv.left_pad, conv.channels))
            native = conv._causal_stateful

            def causal(conv, x, plan, *, native=native):
                if not getattr(plan, '_speco_speculative_graph', False):
                    return native(x, plan)
                return speculative_graph_conv(conv.weight, conv._speco_graph_ring, x.reshape(-1, query, conv.channels), plan.slots, plan.contexts, mask_start=conv.left_pad).reshape_as(x)
            conv._causal_stateful = MethodType(causal, conv)

        def build_plan(*args, **kwargs):
            return active['plan'] if active['plan'] is not None else original_plan(*args, **kwargs)

        def forward(_runner, num_tokens_padded, *args, **kwargs):
            forward_context = get_forward_context()
            if forward_context.cudagraph_runtime_mode != CUDAGraphMode.FULL:
                return original_forward(num_tokens_padded, *args, **kwargs)
            if num_tokens_padded % query or num_tokens_padded > capacity * query:
                raise RuntimeError('DSpark FULL graph requires uniform K+1 target queries')
            batch = num_tokens_padded // query
            requests = module._current_req_ids()
            contexts = []
            slots = []
            if requests:
                lengths = runner.query_lens[:len(requests)].tolist()
                contexts = accepted_graph_contexts(forward_context.attn_metadata, names, request_count=len(requests), query=query, sink_len=128)
                if len(requests) > batch or len(set(requests)) != len(requests) or any((type(q) is not int or q != query for q in lengths)) or any((type(c) is not int or c < 0 for c in contexts)):
                    raise RuntimeError('DSpark graph routing differs from uniform CPU scheduler metadata')
                slots, clear, _ = module._mome_slots_for(requests, [c == 0 for c in contexts], len(requests))
                module._mome_clear_slots(clear, capacity)
                if any((s < 0 or s >= capacity for s in slots)):
                    raise RuntimeError('DSpark graph request exceeds reserved history capacity')
                for history_index, history in enumerate(histories):
                    for request, slot, context in zip(requests, slots, contexts):
                        prior = history.windows.get(slot)
                        if context and (prior is None or context > prior.end or prior.end - context > K or (max(context - history.conv.left_pad, 0) < prior.start)):
                            raise RuntimeError(f"DSpark graph lacks retained MoME rejection boundary: history={history_index}, request={request!r}, slot={slot}, context={context}, prior_start={getattr(prior, 'start', None)}, prior_end={getattr(prior, 'end', None)}, left_pad={history.conv.left_pad}, K={K}, target_revision={getattr(self, '_speco_target_revision', None)}, live_FULL_calls={self._speco_graph_live_forward_calls}")
                for history in histories:
                    for slot, context in zip(slots, contexts):
                        if not context:
                            continue
                        prior = history.windows[slot]
                        positions = torch.arange(prior.start, prior.end, device=history.conv.weight.device)
                        history.conv._speco_graph_ring[slot, positions.remainder(K + history.conv.left_pad)] = prior.inputs
                        if prior.end > context:
                            history.rollback_calls += 1
                            history.rolled_back_tokens += prior.end - context
            all_slots = slots + list(range(capacity, capacity + batch - len(slots)))
            all_contexts = contexts + [0] * (batch - len(contexts))
            if batch not in buffers:
                device = histories[0].conv.weight.device
                buffers[batch] = SimpleNamespace(slots=torch.empty(batch, device=device, dtype=torch.int64), contexts=torch.empty(batch, device=device, dtype=torch.int64), stateful=True, decode_only=False, _speco_speculative_graph=True)
            plan = buffers[batch]
            plan.slots.copy_(torch.tensor(all_slots, device=plan.slots.device, dtype=torch.int64))
            plan.contexts.copy_(torch.tensor(all_contexts, device=plan.contexts.device, dtype=torch.int64))
            if active['plan'] is not None:
                raise RuntimeError('DSpark graph history cannot be reentered')
            original_metadata = forward_context.attn_metadata
            selected_metadata = select_target_graph_metadata(original_metadata, names)
            active['plan'] = plan
            forward_context.attn_metadata = selected_metadata
            try:
                result = original_forward(num_tokens_padded, *args, **kwargs)
            except BaseException:
                for history in histories:
                    for slot in slots:
                        history.windows.pop(slot, None)
                raise
            finally:
                forward_context.attn_metadata = original_metadata
                active['plan'] = None
            for history in histories:
                conv = history.conv
                for slot, context in zip(slots, contexts):
                    end = context + query
                    start = max(0, end - (K + conv.left_pad))
                    positions = torch.arange(start, end, device=conv.weight.device)
                    values = conv._speco_graph_ring[slot, positions.remainder(K + conv.left_pad)].detach().clone()
                    history.windows[slot] = _Window(start, end, values)
            self._speco_graph_forward_calls += 1
            self._speco_graph_live_forward_calls += bool(requests)
            return result
        module._build_mome_plan = build_plan
        runner._model_forward = MethodType(forward, runner)

    def speco_dspark_runtime_status(self):
        status = super().speco_dspark_runtime_status()
        status.update(eager_only=False, draft_eager=True, target_graph_mode=str(self.model_runner.compilation_config.cudagraph_mode), target_FULL_forward_calls=self._speco_graph_forward_calls, target_live_FULL_forward_calls=self._speco_graph_live_forward_calls, target_actual_aclgraph_replay_calls=self._speco_graph_replay_calls, target_attention_layer_names=list(self._speco_target_attention_layer_names), graph_full_model_acceptance_not_claimed=True)
        return status

    def speco_native_expert_layout_status(self):
        """Read actual expert metadata for the synchronous reload investigation."""
        rows = []
        for name, layer in self._speco_aux_target.model.named_modules():
            if not hasattr(layer, 'w13_weight') and (not hasattr(layer, 'w2_weight')):
                continue
            rows.append(dict(name=name, method=type(layer.quant_method).__name__, local_num_experts=layer.local_num_experts, hidden_size=layer.hidden_size, intermediate_size_per_partition=layer.intermediate_size_per_partition, parameters={key: dict(shape=list(getattr(layer, key).shape), dtype=str(getattr(layer, key).dtype), requires_grad=getattr(layer, key).requires_grad, is_transposed=getattr(getattr(layer, key), 'is_transposed', False)) for key in ('w13_weight', 'w2_weight')}))
        return dict(rank=self.rank, layers=rows, read_only=True)
import hashlib
from pathlib import Path

class dspark_conv_trace_worker_Trace:

    def _install_conv_trace(self):
        import torch
        if os.environ.get('SPECO_DSPARK_CONV_DIAGNOSTIC') != '1':
            raise RuntimeError('convolution trace requires explicit diagnostic ownership')
        histories = self._speco_mome_histories
        indices = sorted({0, len(histories) // 2, len(histories) - 1})
        buffers = {}
        for index in indices:
            conv = histories[index].conv
            buffers[index] = dict(x=conv.weight.new_empty(9, conv.channels), y=conv.weight.new_empty(9, conv.channels), weight=conv.weight.detach().cpu().clone(), context=conv.weight.new_empty(1, dtype=torch.int64))
            original = conv._causal_stateful

            def causal(_conv, x, plan, *, original=original, buffer=buffers[index]):
                result = original(x, plan)
                if x.shape == buffer['x'].shape:
                    actual_context = plan.contexts if getattr(plan, '_speco_speculative_graph', False) else torch.tensor(plan.ctx_lens, device=x.device, dtype=torch.int64)
                    buffer['context'].copy_(actual_context)
                    buffer['x'].copy_(x)
                    buffer['y'].copy_(result)
                return result
            conv._causal_stateful = MethodType(causal, conv)
        runner = self.model_runner
        original_forward = runner._model_forward
        self._speco_conv_trace = []

        def forward(_runner, num_tokens_padded, *args, **kwargs):
            result = original_forward(num_tokens_padded, *args, **kwargs)
            requests = list(runner.input_batch.req_ids)
            if len(self._speco_conv_trace) < 3 and len(requests) == 1:
                lengths = runner.query_lens[:1].tolist()
                contexts = runner.input_batch.num_computed_tokens_cpu[:1].tolist()
                if lengths == [9] and contexts[0] > 0:
                    self._speco_conv_trace.append(dict(requests=requests, contexts=contexts, query_lengths=lengths, layers={index: {key: value.detach().cpu().clone() for key, value in buffer.items()} for index, buffer in buffers.items()}))
            return result
        runner._model_forward = MethodType(forward, runner)

    def speco_save_conv_trace(self):
        import torch
        if len(self._speco_conv_trace) != 3:
            raise RuntimeError('three actual target verification forwards were not observed')
        packet = Path(os.environ['SPECO_DSPARK_GRAPH_PROBE_RECEIPT']).parent
        mode = 'eager' if self.model_runner.model_config.enforce_eager else 'FULL_DECODE_ONLY'
        path = packet / f'conv-trace-{mode}-rank{self.rank}.pt'
        torch.save(self._speco_conv_trace, path)
        return dict(rank=self.rank, mode=mode, path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), forwards=3, observed_layers=3, model_values_not_modified=True)

class EagerConvTraceWorker(dspark_conv_trace_worker_Trace, SpecoMultiNodeTargetUpdateWorker):

    def load_model(self):
        super().load_model()
        self._install_conv_trace()

class GraphConvTraceWorker(dspark_conv_trace_worker_Trace, SpecoMultiNodeDSparkGraphWorker):

    def _install_graph_history(self):
        super()._install_graph_history()
        self._install_conv_trace()
from torch import nn
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3ForCausalLM
from verl_speco.models.dspark.modeling_dspark import DSparkVanillaMarkovHead, DSparkConfidenceHead

class MultiNodeDSparkForCausalLM(DFlashQwen3ForCausalLM):
    has_own_embed_tokens = True
    has_own_lm_head = False

    def __init__(self, *, vllm_config, prefix=''):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        if self.config.markov_head_type != 'vanilla':
            raise ValueError('target model DSpark serving currently requires vanilla Markov heads')
        self.model.markov_head = DSparkVanillaMarkovHead(vocab_size=self.config.vocab_size, markov_rank=self.config.markov_rank).float()
        self.model.confidence_head = DSparkConfidenceHead(self.config.hidden_size + self.config.markov_rank).float()
        self._speco_loaded_parameter_names = frozenset()

    def load_weights(self, weights):
        source_names = set()

        def translated():
            for name, tensor in weights:
                if not name.startswith('draft_model.'):
                    raise ValueError('target model DSpark checkpoint requires draft_model snapshot namespace')
                name = name[len('draft_model.'):]
                if name in source_names:
                    raise ValueError('duplicate target model DSpark weight')
                source_names.add(name)
                yield (name, tensor)
        super().load_weights(translated())
        required = {'embed_tokens.weight', 'fc.weight', 'hidden_norm.weight', 'norm.weight', 'markov_head.markov_w1.weight', 'markov_head.markov_w2.weight', 'confidence_head.proj.weight', 'confidence_head.proj.bias'}
        if not required.issubset(source_names):
            raise ValueError('target model DSpark checkpoint is missing required trainable weights')
        self._speco_loaded_parameter_names = frozenset(source_names)
from verl_speco.integration.multinode_publication import GraphTargetPublicationMixin
from verl_speco.integration.multinode_recovery import record_colocated_memory

class SpecoMultiNodeDSparkGraphPublicationWorker(GraphTargetPublicationMixin, SpecoMultiNodeDSparkGraphWorker):

    def wake_up(self, tags=None):
        details = dict(operation='wake_up', tags=tags)
        record_colocated_memory(role='native', rank=self.rank, stage='before', details=details)
        try:
            result = super().wake_up(tags=tags)
        except BaseException as error:
            record_colocated_memory(role='native', rank=self.rank, stage='failed', details=dict(details, error=str(error)[:1000]))
            raise
        record_colocated_memory(role='native', rank=self.rank, stage='after', details=details)
        return result
