# Licensed under the Apache License, Version 2.0. See LICENSE.
from __future__ import annotations
'A standalone draft artifact must preserve frozen weights and reject tampering.'
from pathlib import Path
import pytest
import torch
from torch import nn
from verl_speco.trainer.distributed_checkpoint import save_trainable_snapshot, load_trainable_snapshot

class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.draft = nn.Linear(3, 2)
        self.frozen = nn.Parameter(torch.ones(2), requires_grad=False)

def save(model, directory):
    return save_trainable_snapshot(model, directory, draft_revision=1, generation=1, source_step_id='committed-step', target_model_fingerprint='target-v1', tokenizer_fingerprint='tokens-v1', model_config={'hidden_size': 3})

def load(model, snapshot, **overrides):
    values = dict(target_model_fingerprint='target-v1', tokenizer_fingerprint='tokens-v1', model_config={'hidden_size': 3})
    values.update(overrides)
    return load_trainable_snapshot(model, snapshot, **values)

def test_round_trip_only_restores_trainable_parameters(tmp_path):
    source = Model()
    destination = Model()
    with torch.no_grad():
        destination.frozen.fill_(7)
    snapshot = save(source, tmp_path / 'revision-1')
    manifest = load(destination, snapshot)
    assert set(manifest['parameters']) == {'draft.weight', 'draft.bias'}
    for name, param in source.named_parameters():
        if param.requires_grad:
            torch.testing.assert_close(param, dict(destination.named_parameters())[name])
    torch.testing.assert_close(destination.frozen, torch.full((2,), 7.0))
    with pytest.raises(FileExistsError):
        save(source, tmp_path / 'revision-1')

@pytest.mark.parametrize('override', [{'target_model_fingerprint': 'target-v2'}, {'tokenizer_fingerprint': 'tokens-v2'}, {'model_config': {'hidden_size': 4}}])
def test_incompatible_artifact_rejected_before_any_parameter_changes(tmp_path, override):
    model = Model()
    snapshot = save(model, tmp_path / 'revision')
    with torch.no_grad():
        model.draft.weight.fill_(11)
    original = model.draft.weight.detach().clone()
    with pytest.raises(ValueError, match='geometry or namespace'):
        load(model, snapshot, **override)
    torch.testing.assert_close(model.draft.weight, original)

def test_changed_weights_are_rejected_before_loading(tmp_path):
    model = Model()
    snapshot = save(model, tmp_path / 'revision')
    path = Path(snapshot.uri) / 'trainable.safetensors'
    with path.open('ab') as handle:
        handle.write(b'tampered')
    with pytest.raises(ValueError, match='weight checksum'):
        load(model, snapshot)

def test_parameter_mapping_must_cover_the_entire_trainable_namespace(tmp_path):
    model = Model()
    model.extra = nn.Parameter(torch.zeros(1))
    with pytest.raises(ValueError, match='namespace'):
        save_trainable_snapshot(model, tmp_path / 'bad', draft_revision=1, generation=1, source_step_id='step', target_model_fingerprint='target', tokenizer_fingerprint='tokens', model_config={}, namespace_prefix='draft.')
    assert not (tmp_path / 'bad').exists()
    model = Model()
    snapshot = save_trainable_snapshot(model, tmp_path / 'good', draft_revision=1, generation=1, source_step_id='step', target_model_fingerprint='target-v1', tokenizer_fingerprint='tokens-v1', model_config={'hidden_size': 3}, namespace_prefix='draft.')
    manifest = load(Model(), snapshot, namespace_prefix='draft.')
    assert set(manifest['parameters']) == {'weight', 'bias'}

def test_partial_shard_cannot_be_exported_as_a_full_snapshot(tmp_path):

    class PartialParameter:
        requires_grad = True
        shape = (4,)

        def to_local(self):
            return torch.ones(2)

    class PartialModule:

        def named_parameters(self):
            return [('weight', PartialParameter())]
    with pytest.raises(ValueError, match='full replicated'):
        save_trainable_snapshot(PartialModule(), tmp_path / 'partial', draft_revision=1, generation=1, source_step_id='step', target_model_fingerprint='target', tokenizer_fingerprint='tokens', model_config={})
    assert not (tmp_path / 'partial').exists()

@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_parameter_digest_preserves_legacy_bytes_for_noncontiguous_weights(dtype):
    import hashlib
    from verl_speco.trainer.distributed_checkpoint import trainable_snapshot_parameter_checksum as _parameter_checksum, trainable_snapshot_parameters as _parameters
    model = nn.Module()
    model.register_parameter('weights', nn.Parameter(torch.arange(12, dtype=dtype).reshape(3, 4).t()))
    model.register_parameter('frozen', nn.Parameter(torch.ones(2), requires_grad=False))
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        if parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    assert _parameter_checksum(_parameters(model, '')) == digest.hexdigest()
