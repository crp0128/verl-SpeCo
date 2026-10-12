# Licensed under the Apache License, Version 2.0. See LICENSE.
"""Consolidated multi-node capabilities. Hardware adapters load only when explicitly selected."""
from __future__ import annotations
import logging
import os
from copy import deepcopy
from typing import Any, Optional, cast
import torch
import torch.nn.functional as F
from verl_speco.backends.dflash_trainer_backend import DFlashTrainerBackend, DFlashTrainingModel, _block_acceptance_counts, _resolve_sliding_windows, _sliding_window_config, build_dflash_attention_masks, _document_boundary_validity
from verl_speco.models.dflash import resolve_rope_theta
from verl_speco.models.dspark import DSparkConfig, DSparkDraftModel
from verl_speco.ops.dspark_fused_loss import fused_label_cross_entropy, fused_loss_capability, fused_total_variation
from verl_speco.trainer.checkpoint import log_drafter_checkpoint_step
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv('VERL_LOGGING_LEVEL', 'INFO'))
from verl_speco.backends.dspark_trainer_backend import DSparkTrainingModel as DefaultDSparkTrainingModel

class MultiNodeDSparkTrainingModel(DefaultDSparkTrainingModel):
    """Training wrapper around DSparkDraftModel.

    It reuses the DFlash block drafter backbone and changes the target alignment
    to DSpark semantics:

    - anchor token `x[a]` seeds position 0,
    - labels are `x[a + 1 : a + 1 + block_size]`,
    - previous tokens are `[x[a], labels[:-1]]`,
    - every supervised position, including position 0, contributes to CE.
    """
    draft_model: DSparkDraftModel

    def __init__(self, draft_model: DSparkDraftModel, block_size: int=7, num_anchors: int=512, loss_decay_gamma: float=7.0, loss_mode: str='full_vocab', sampled_ce_negatives: int=0, ce_loss_alpha: float=0.1, l1_loss_alpha: float=0.9, confidence_head_alpha: float=0.0, l1_chunk_size: int=0, distribution_loss_impl: str='auto', anchor_sampling_policy: str='random', debug_log: bool=False, debug_log_first_n: int=2, debug_log_interval: int=100):
        DFlashTrainingModel.__init__(self, draft_model=draft_model, block_size=block_size, num_anchors=num_anchors, loss_decay_gamma=loss_decay_gamma, front_position_weight=1.0, front_position_count=0, loss_mode=loss_mode, sampled_ce_negatives=sampled_ce_negatives)
        if anchor_sampling_policy not in {'random', 'dmtp_cpu'}:
            raise ValueError('unknown DSpark anchor sampling policy')
        configured_policy = getattr(draft_model.config, 'anchor_sampling_policy', 'random')
        if configured_policy not in {'random', 'dmtp_cpu'}:
            raise ValueError('unknown configured DSpark anchor sampling policy')
        if configured_policy != 'random' and configured_policy != anchor_sampling_policy:
            raise ValueError('DSpark anchor sampling policy conflicts with loaded model config')
        self.anchor_sampling_policy = anchor_sampling_policy
        if anchor_sampling_policy != 'random':
            self.draft_model.config.anchor_sampling_policy = anchor_sampling_policy
        self.ce_loss_alpha = float(ce_loss_alpha)
        self.l1_loss_alpha = float(l1_loss_alpha)
        self.confidence_head_alpha = float(confidence_head_alpha)
        self.l1_chunk_size = int(l1_chunk_size or 0)
        self.distribution_loss_impl = str(distribution_loss_impl).lower()
        if self.distribution_loss_impl not in {'auto', 'fused', 'eager'}:
            raise ValueError('distribution_loss_impl must be one of: auto, fused, eager')
        self._logged_distribution_loss_backend: Optional[str] = None
        self._distribution_loss_by_device: dict[str, bool] = {}
        if not 0 <= self.confidence_head_alpha < float('inf'):
            raise ValueError('DSpark confidence loss weight must be finite and nonnegative')
        confidence_head = getattr(self.draft_model, 'confidence_head', None)
        if self.confidence_head_alpha > 0 and confidence_head is None:
            raise ValueError('DSpark confidence loss requires an enabled confidence head')
        if confidence_head is not None:
            confidence_head.requires_grad_(self.confidence_head_alpha > 0)
            if self.confidence_head_alpha == 0:
                logger.info('[dspark-trainer] confidence head is loaded but frozen because confidence loss is disabled')
        self.debug_log = bool(debug_log)
        self.debug_log_first_n = max(int(debug_log_first_n), 0)
        self.debug_log_interval = max(int(debug_log_interval), 1)
        self._debug_forward_count = 0

    def _validated_planned_anchors(self, *, loss_mask, document_ids, positions, keep):
        if positions is None and keep is None and (loss_mask.device.type == 'cpu'):
            from verl_speco.trainer.distributed_batches import plan_dspark_response_anchors
            positions, keep = plan_dspark_response_anchors(loss_mask=loss_mask, document_ids=document_ids, num_anchors=self.num_anchors, block_size=self.block_size)
        shape = (loss_mask.shape[0], self.num_anchors)
        if positions is None or keep is None or positions.shape != shape or (keep.shape != shape) or (positions.dtype != torch.int64) or (keep.dtype != torch.bool) or (positions.device != loss_mask.device) or (keep.device != loss_mask.device):
            raise ValueError('DMTP anchors must be planned on CPU before accelerator transfer')
        if ((positions < 0) | (positions + self.block_size >= loss_mask.shape[1]))[keep].any():
            raise ValueError('planned anchor label block is outside its window')
        if ((positions != 0) & ~keep).any():
            raise ValueError('unused planned anchor slots must be zero')
        offsets = torch.arange(1, self.block_size + 1, device=positions.device)
        labels = (positions[:, :, None] + offsets).clamp(max=max(loss_mask.shape[1] - 1, 0))
        supervised = torch.gather(loss_mask[:, None, :].expand(-1, self.num_anchors, -1), 2, labels) > 0.5
        if (keep[:, :, None] & ~supervised).any():
            raise ValueError('planned anchor crosses a supervision boundary')
        if document_ids is not None:
            seed_docs = torch.gather(document_ids, 1, positions.clamp(max=max(loss_mask.shape[1] - 1, 0)))
            label_docs = torch.gather(document_ids[:, None, :].expand(-1, self.num_anchors, -1), 2, labels)
            if (keep[:, :, None] & ((label_docs != seed_docs[:, :, None]) | (seed_docs[:, :, None] < 0))).any():
                raise ValueError('planned anchor crosses a document boundary')
        for row, valid in zip(positions, keep):
            if row[valid].unique().numel() != int(valid.sum()):
                raise ValueError('planned anchors contain duplicate starts')
        return (positions, keep)

    def _compute_confidence_distribution_loss_for_active(self, *, active_hidden, active_prev_tokens, active_target_hidden, active_weights, lm_head_weight, active_draft_log_probs=None, active_draft_logits=None):
        """DMTP overlap target with live draft/Markov confidence features.

        The target is detached: BCE never differentiates through overlap or the
        teacher. Full-vocabulary probabilities are required even for sampled CE.
        Token chunks bound temporary teacher logits and reuse full CE log-probs.
        """
        zero = active_weights.new_zeros((), dtype=torch.float32)
        if active_hidden.numel() == 0:
            return (zero, zero, zero, zero)
        count = active_hidden.size(0)
        if active_draft_log_probs is not None and tuple(active_draft_log_probs.shape) != (count, lm_head_weight.size(0)):
            raise ValueError('confidence loss needs full-vocabulary draft probabilities')
        if active_draft_logits is not None and tuple(active_draft_logits.shape) != (count, lm_head_weight.size(0)):
            raise ValueError('confidence loss needs full-vocabulary draft logits')
        use_fused = self._use_fused_distribution_loss(active_hidden.device)
        l1_sum, confidence_sum = (zero, zero)
        denominator = active_weights.float().sum()
        chunk_size = self.l1_chunk_size if self.l1_chunk_size > 0 else count
        for start in range(0, count, chunk_size):
            end = min(start + chunk_size, count)
            hidden = active_hidden[start:end]
            previous = active_prev_tokens[start:end]
            weights = active_weights[start:end].float()
            if use_fused or active_draft_log_probs is None:
                if active_draft_logits is not None:
                    logits = active_draft_logits[start:end]
                else:
                    logits = F.linear(hidden, lm_head_weight)
                    bias = self._markov_bias_for_active(active_hidden=hidden, active_prev_tokens=previous, restricted_vocab=None)
                    if bias is not None:
                        logits = logits + bias
                if not use_fused:
                    draft_probs = torch.softmax(logits.float(), dim=-1)
            else:
                draft_probs = active_draft_log_probs[start:end].exp()
            with torch.no_grad():
                target_logits = F.linear(active_target_hidden[start:end], lm_head_weight)
                if not use_fused:
                    target_probs = torch.softmax(target_logits.float(), dim=-1)
            l1 = 2.0 * fused_total_variation(logits, target_logits) if use_fused else (draft_probs - target_probs).abs().sum(dim=-1)
            confidence_logits = self.draft_model.predict_confidence(hidden, prev_token_ids=previous)
            if confidence_logits is None:
                raise ValueError('confidence head cannot produce logits')
            overlap = (1.0 - 0.5 * l1).detach().clamp(0.0, 1.0)
            if not (torch.isfinite(overlap).all() and torch.isfinite(confidence_logits).all()):
                raise ValueError('nonfinite DSpark confidence logits or overlap target')
            confidence_loss = F.binary_cross_entropy_with_logits(confidence_logits.float(), overlap, reduction='none')
            l1_sum = l1_sum + (l1 * weights).sum()
            confidence_sum = confidence_sum + (confidence_loss * weights).sum()
        return (l1_sum, denominator, confidence_sum, denominator)

    def forward(self, input_ids: torch.Tensor, hidden_states_list: list[torch.Tensor], loss_mask: torch.Tensor, lm_head_weight: torch.Tensor, document_ids: Optional[torch.Tensor]=None, target_last_hidden_states: Optional[torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None, preplanned_anchor_positions: Optional[torch.Tensor]=None, preplanned_anchor_keep_mask: Optional[torch.Tensor]=None):
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        self._debug_forward_count += 1
        if self.anchor_sampling_policy == 'dmtp_cpu':
            anchor_positions, block_keep_mask = self._validated_planned_anchors(loss_mask=loss_mask, document_ids=document_ids, positions=preplanned_anchor_positions, keep=preplanned_anchor_keep_mask)
        else:
            if preplanned_anchor_positions is not None or preplanned_anchor_keep_mask is not None:
                raise ValueError('preplanned anchors require an explicit DMTP sampling profile')
            anchor_positions, block_keep_mask = self._sample_anchor_positions(seq_len, loss_mask, device, document_ids=document_ids)
        context_feature = self.draft_model.extract_context_feature(hidden_states_list)
        n_blocks = anchor_positions.shape[1]
        noise_embedding = self._create_noise_embed(input_ids, anchor_positions, block_keep_mask)
        context_position_ids, draft_position_ids = self._create_position_ids(anchor_positions, seq_len, position_ids=position_ids)
        block_mask, dense_attention_mask = build_dflash_attention_masks(anchor_positions=anchor_positions, block_keep_mask=block_keep_mask, ctx_len=seq_len, block_size=self.block_size, device=device, windows=_resolve_sliding_windows(self.draft_model.config), document_ids=document_ids)
        draft_hidden = self.draft_model(draft_input_ids=None, context_feature=context_feature, draft_position_ids=draft_position_ids, context_position_ids=context_position_ids, block_mask=block_mask, dense_attention_mask=dense_attention_mask, noise_embedding=noise_embedding).view(bsz, n_blocks, self.block_size, -1)
        target_ids, prev_token_ids, eval_mask, label_indices = self._build_label_tensors(input_ids=input_ids, loss_mask=loss_mask, anchor_positions=anchor_positions, block_keep_mask=block_keep_mask)
        aligned_target_hidden = self._gather_aligned_target_hidden(target_last_hidden_states=target_last_hidden_states, label_indices=label_indices, block_keep_mask=block_keep_mask)
        weight_mask = eval_mask.float()
        if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
            positions = self._cached_arange('dspark_decay_positions', self.block_size, device, view_shape=(1, 1, -1))
            weight_mask = weight_mask * torch.exp(-positions.float() / float(self.loss_decay_gamma))
        flat_targets = target_ids.reshape(-1)
        flat_prev_tokens = prev_token_ids.reshape(-1)
        flat_weights = weight_mask.reshape(-1)
        active_mask = flat_weights > 0
        flat_hidden = draft_hidden.reshape(-1, draft_hidden.size(-1))
        active_hidden = flat_hidden[active_mask]
        active_targets = flat_targets[active_mask]
        active_prev_tokens = flat_prev_tokens[active_mask]
        active_weights = flat_weights[active_mask]
        active_target_hidden = aligned_target_hidden.reshape(-1, aligned_target_hidden.size(-1))[active_mask] if aligned_target_hidden is not None else None
        loss_per_token = torch.zeros_like(flat_weights)
        sanitized_rows = torch.zeros((), dtype=torch.float32, device=device)
        local_ploss_sum = torch.zeros((), dtype=torch.float32, device=device)
        local_ce_den = torch.zeros((), dtype=torch.float32, device=device)
        local_l1_sum = torch.zeros((), dtype=torch.float32, device=device)
        local_l1_den = torch.zeros((), dtype=torch.float32, device=device)
        local_confidence_sum = torch.zeros((), dtype=torch.float32, device=device)
        local_confidence_den = torch.zeros((), dtype=torch.float32, device=device)
        active_logits = None
        active_log_probs = None
        restricted_vocab = None
        use_fused_loss = self._use_fused_distribution_loss(device)
        if active_targets.numel() == 0:
            loss = flat_weights.sum() * 0.0
            if self.anchor_sampling_policy == 'dmtp_cpu':
                loss = loss + draft_hidden.float().sum() * 0.0
                previous = input_ids[:, :1].reshape(-1)
                markov_head = getattr(self.draft_model, 'markov_head', None)
                if markov_head is not None:
                    bias = markov_head.compute_step_bias(previous)
                    loss = loss + bias.float().sum() * 0.0
                if self.confidence_head_alpha > 0:
                    confidence = self.draft_model.predict_confidence(draft_hidden[:, 0, 0], prev_token_ids=previous)
                    loss = loss + confidence.float().sum() * 0.0
        else:
            if self.loss_mode in {'restricted_ce', 'sampled_ce'}:
                restricted_vocab = self._build_restricted_vocab(input_ids, active_targets, int(lm_head_weight.shape[0]))
                restricted_weight = lm_head_weight.index_select(0, restricted_vocab)
                active_logits = F.linear(active_hidden, restricted_weight)
                markov_bias = self._markov_bias_for_active(active_hidden=active_hidden, active_prev_tokens=active_prev_tokens, restricted_vocab=restricted_vocab)
                if markov_bias is not None:
                    active_logits = active_logits + markov_bias
                active_ce_targets = torch.searchsorted(restricted_vocab, active_targets)
                active_loss = fused_label_cross_entropy(active_logits, active_ce_targets) if use_fused_loss else F.cross_entropy(active_logits, active_ce_targets, reduction='none')
            else:
                active_logits = F.linear(active_hidden, lm_head_weight)
                markov_bias = self._markov_bias_for_active(active_hidden=active_hidden, active_prev_tokens=active_prev_tokens, restricted_vocab=None)
                if markov_bias is not None:
                    active_logits = active_logits + markov_bias
                if use_fused_loss:
                    active_loss = fused_label_cross_entropy(active_logits, active_targets)
                else:
                    active_log_probs = F.log_softmax(active_logits.float(), dim=-1)
                    active_loss = F.nll_loss(active_log_probs, active_targets, reduction='none')
            finite_loss = torch.isfinite(active_loss)
            sanitized_rows = (~finite_loss).sum().to(dtype=torch.float32)
            active_loss = torch.where(finite_loss, active_loss, torch.zeros_like(active_loss))
            active_loss_weights = active_weights * finite_loss.to(dtype=active_weights.dtype)
            loss_per_token[active_mask] = active_loss
            local_ce_den = active_loss_weights.sum()
            valid_token_count = local_ce_den.clamp(min=1e-06)
            local_ploss_sum = (active_loss * active_loss_weights).sum()
            ce_loss = local_ploss_sum / valid_token_count
            if self.l1_loss_alpha > 0 or self.confidence_head_alpha > 0:
                if active_target_hidden is None:
                    raise ValueError('DSpark L1 loss requires target_last_hidden_states. Confidence loss requires it too. Enable dflash_aux_plus_last collection or disable both L1 and confidence loss.')
                finite_target_hidden = torch.isfinite(active_target_hidden).all(dim=-1)
                l1_mask = finite_loss & finite_target_hidden
                if l1_mask.any():
                    all_l1_rows = bool(l1_mask.all())
                    l1_hidden = active_hidden if all_l1_rows else active_hidden[l1_mask]
                    l1_prev = active_prev_tokens if all_l1_rows else active_prev_tokens[l1_mask]
                    l1_target_hidden = active_target_hidden if all_l1_rows else active_target_hidden[l1_mask]
                    l1_weights = active_loss_weights if all_l1_rows else active_loss_weights[l1_mask]
                    if self.confidence_head_alpha > 0:
                        reusable_logits = None
                        if restricted_vocab is None and use_fused_loss:
                            reusable_logits = active_logits if all_l1_rows else active_logits[l1_mask]
                        reusable_log_probs = None
                        if restricted_vocab is None and active_log_probs is not None:
                            reusable_log_probs = active_log_probs if all_l1_rows else active_log_probs[l1_mask]
                        local_l1_sum, local_l1_den, local_confidence_sum, local_confidence_den = self._compute_confidence_distribution_loss_for_active(active_hidden=l1_hidden, active_prev_tokens=l1_prev, active_target_hidden=l1_target_hidden, active_weights=l1_weights, lm_head_weight=lm_head_weight, active_draft_log_probs=reusable_log_probs, active_draft_logits=reusable_logits)
                    elif use_fused_loss:
                        reusable_draft_logits = None
                        if restricted_vocab is None:
                            reusable_draft_logits = active_logits if all_l1_rows else active_logits[l1_mask]
                        local_l1_sum, local_l1_den = self._compute_fused_l1_loss_for_active(active_hidden=l1_hidden, active_prev_tokens=l1_prev, active_target_hidden=l1_target_hidden, active_weights=l1_weights, lm_head_weight=lm_head_weight, active_draft_logits=reusable_draft_logits)
                    else:
                        reusable_draft_log_probs = None
                        if restricted_vocab is None:
                            if active_log_probs is None:
                                raise ValueError('DSpark L1 loss requires active_log_probs')
                            reusable_draft_log_probs = active_log_probs if all_l1_rows else active_log_probs[l1_mask]
                        local_l1_sum, local_l1_den = self._compute_l1_loss_for_active(active_hidden=l1_hidden, active_prev_tokens=l1_prev, active_target_hidden=l1_target_hidden, active_weights=l1_weights, lm_head_weight=lm_head_weight, active_draft_log_probs=reusable_draft_log_probs)
                l1_loss = local_l1_sum / local_l1_den.clamp(min=1e-06)
            else:
                l1_loss = local_ploss_sum.new_zeros(())
            loss = ce_loss * self.ce_loss_alpha + l1_loss * self.l1_loss_alpha
            if self.confidence_head_alpha > 0:
                confidence_loss = local_confidence_sum / local_confidence_den.clamp(min=1e-06)
                loss = loss + confidence_loss * self.confidence_head_alpha
        with torch.no_grad():
            flat_eval_mask = eval_mask.reshape(-1)
            binary_eval_mask = flat_eval_mask & (flat_weights > 0)
            correct = torch.zeros_like(flat_weights, dtype=torch.bool)
            top1_correct = torch.zeros((), dtype=torch.float32, device=device)
            top5_correct = torch.zeros((), dtype=torch.float32, device=device)
            quality_token_count = torch.zeros((), dtype=torch.float32, device=device)
            sampled_vocab_size = torch.tensor(float(restricted_vocab.numel()) if restricted_vocab is not None else float(lm_head_weight.shape[0]), dtype=torch.float32, device=device)
            if active_logits is not None and active_targets.numel() > 0:
                if restricted_vocab is not None:
                    pred_local = active_logits.argmax(dim=-1)
                    pred_tokens = restricted_vocab[pred_local]
                    topk = min(5, int(restricted_vocab.numel()))
                    top_tokens = restricted_vocab[active_logits.topk(topk, dim=-1).indices]
                else:
                    pred_tokens = active_logits.argmax(dim=-1)
                    topk = min(5, active_logits.shape[-1])
                    top_tokens = active_logits.topk(topk, dim=-1).indices
                active_correct = pred_tokens.eq(active_targets)
                correct[active_mask] = active_correct
                top1_correct = active_correct.float().sum()
                top5_correct = top_tokens.eq(active_targets.unsqueeze(-1)).any(dim=-1).float().sum()
                quality_token_count = active_targets.new_tensor(float(active_targets.numel()), dtype=torch.float32)
            binary_weights = binary_eval_mask.view(bsz, n_blocks, self.block_size).float()
            loss_3d = loss_per_token.view(bsz, n_blocks, self.block_size)
            correct_3d = correct.view(bsz, n_blocks, self.block_size).float()
            count_per_position = binary_weights.sum(dim=(0, 1)).to(torch.float32)
            loss_sum_per_position = (loss_3d * binary_weights).sum(dim=(0, 1))
            correct_per_position = correct_3d.sum(dim=(0, 1))
            loss_per_position = loss_sum_per_position / count_per_position.clamp(min=1.0)
            acc_per_position = correct_per_position / count_per_position.clamp(min=1.0)
            accepted_length_sum, scored_block_count = _block_acceptance_counts(correct.view(bsz, n_blocks, self.block_size), binary_weights > 0)
            valid_token_count = active_weights.sum().to(dtype=torch.float32)
            weighted_token_count = flat_weights.sum().to(dtype=torch.float32)
            accuracy = correct.float().sum() / binary_eval_mask.float().sum().clamp(min=1.0)
        self._log_debug_preview(input_ids=input_ids, anchor_positions=anchor_positions, block_keep_mask=block_keep_mask, target_ids=target_ids, prev_token_ids=prev_token_ids, eval_mask=eval_mask, active_logits=active_logits, loss=loss)
        diagnostics = {'correct_count': correct.float().sum().detach(), 'eval_token_count': binary_eval_mask.float().sum().detach(), 'top1_correct_count': top1_correct.detach(), 'top5_correct_count': top5_correct.detach(), 'quality_token_count': quality_token_count.detach(), 'valid_token_count': valid_token_count.detach(), 'weighted_token_count': weighted_token_count.detach(), 'ce_loss_sum': local_ploss_sum.detach(), 'ce_weighted_token_count': local_ce_den.detach(), 'l1_loss_sum': local_l1_sum.detach(), 'l1_weighted_token_count': local_l1_den.detach(), 'confidence_loss_sum': local_confidence_sum.detach(), 'confidence_weighted_token_count': local_confidence_den.detach(), 'sanitized_rows': sanitized_rows.detach(), 'masked_rows': (~binary_eval_mask & flat_eval_mask).float().sum().detach(), 'sampled_vocab_size': sampled_vocab_size.detach(), 'loss_mode_id': torch.tensor({'full_vocab': 0.0, 'restricted_ce': 1.0, 'sampled_ce': 2.0}.get(self.loss_mode, 0.0), dtype=torch.float32, device=device), 'distribution_loss_impl_id': torch.tensor(1.0 if use_fused_loss else 0.0, dtype=torch.float32, device=device), 'loss_sum_per_position': loss_sum_per_position.detach(), 'correct_per_position': correct_per_position.detach(), 'count_per_position': count_per_position.detach(), 'accepted_length_sum': accepted_length_sum.detach(), 'scored_block_count': scored_block_count.detach(), 'local_ploss_sum': local_ploss_sum.detach()}
        return (loss, accuracy, loss_per_position, acc_per_position, count_per_position, diagnostics)
