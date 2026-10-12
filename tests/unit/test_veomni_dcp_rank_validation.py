# Licensed under the Apache License, Version 2.0. See LICENSE.
from __future__ import annotations
'A rank-local checkpoint rejection must stop every rank before model mutation.'
import copy
from datetime import timedelta
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def _worker(rank, rendezvous, directory, fault):
    import verl_speco.trainer.distributed_checkpoint as bridge_module
    torch.npu = SimpleNamespace(is_available=lambda: False, get_rng_state=lambda: torch.zeros(4, dtype=torch.uint8), set_rng_state=lambda _: None)
    store = dist.TCPStore('127.0.0.1', rendezvous, 2, False, timeout=timedelta(seconds=15), use_libuv=False)
    dist.init_process_group('gloo', store=store, rank=rank, world_size=2, timeout=timedelta(seconds=15))
    try:
        torch.manual_seed(42)
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        model(torch.ones(1, 3)).sum().backward()
        optimizer.step()
        optimizer.zero_grad()
        source = bridge_module.VeOmniDCPBridge(module=model, optimizer=optimizer, scheduler=None, topology_fingerprint='mesh', model_fingerprint='model')
        candidate = source.save_candidate(Path(directory) / 'candidate', generation=1)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(100)
        optimizer.state.clear()
        before = copy.deepcopy(model.state_dict())
        original_hash = bridge_module.veomni_dcp_bridge_sha256_file
        if rank == 0:
            if fault == 'checksum':
                bridge_module.veomni_dcp_bridge_sha256_file = lambda p: '0' * 64 if p.name == 'extra-rank-0001.pt' else original_hash(p)
            elif fault == 'model':
                source.model_fingerprint = 'foreign-model'
        error = None
        try:
            source.restore_candidate(candidate)
        except BaseException as exc:
            error = dict(type=type(exc).__name__, message=str(exc))
        result = dict(rank=rank, error=error, model_unchanged=all((torch.equal(v, before[k]) for k, v in model.state_dict().items())), optimizer_unchanged=not bool(optimizer.state))
        (Path(directory) / f'rank-{rank}.json').write_text(json.dumps(result))
    finally:
        dist.destroy_process_group()

@pytest.mark.parametrize('fault', ['checksum', 'model'])
def test_rank_local_verification_rejects_every_rank_before_dcp_load(tmp_path, fault):
    store = dist.TCPStore('127.0.0.1', 0, 2, True, wait_for_workers=False, use_libuv=False)
    mp.spawn(_worker, args=(store.port, str(tmp_path), fault), nprocs=2, join=True)
    results = [json.loads((tmp_path / f'rank-{rank}.json').read_text()) for rank in range(2)]
    assert all((r['error'] and r['error']['type'] == 'ValueError' for r in results)), results
    assert all((r['model_unchanged'] and r['optimizer_unchanged'] for r in results)), results
