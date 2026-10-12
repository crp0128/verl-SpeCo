# Licensed under the Apache License, Version 2.0. See LICENSE.
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import hashlib
import importlib.util
import json
import pytest
from verl_speco.runtime.artifacts import publish_json, read_published_json

def fixture(tmp_path):
    root = tmp_path / 'outputs'
    run = 'epoch-retention-test'
    work = root / f'speco-native-wave-{run}-g1'
    (work / 'controls').mkdir(parents=True)
    records = []
    for step in range(3):
        directory = root / f'speco-three-node-{run}' / ('step-' + f'{step:020x}')
        directory.mkdir(parents=True)
        files = {}
        for name in ['__0_0.distcp', '.metadata']:
            payload = (name + str(step)).encode()
            (directory / name).write_bytes(payload)
            files[name] = hashlib.sha256(payload).hexdigest()
        manifest = json.dumps(dict(schema_version=1, generation=1, world_size=16, files=files)).encode()
        (directory / 'manifest.json').write_bytes(manifest)
        control = work / 'controls' / f'step-{step:04d}-control-00000000000000000000000000000000.json'
        control.write_bytes(b'owned-control-' + str(step).encode())
        record = dict(schema_version=1, state='STEP_COMMITTED', run_id=run, generation=1, step_index=step, start=480 * step, end=480 * (step + 1), receipt_sha256='a' * 64, candidate=dict(uri=str(directory), checksum=hashlib.sha256(manifest).hexdigest(), checkpoint_generation=1), control_snapshot=dict(storage='shared-filesystem-v1', path=str(control), nbytes=control.stat().st_size, sha256=hashlib.sha256(control.read_bytes()).hexdigest()))
        publish_json(work / f'committed-step-{step:04d}.json', record)
        records.append(record)
    original = root / 'sequences.jsonl'
    original.write_bytes(b'original-saved-sequences-must-survive')
    journal = work / 'optimizer-journal.json'
    journal.write_bytes(b'full-history-must-survive')
    return (root, work, records, original, journal)
