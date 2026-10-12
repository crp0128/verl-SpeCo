# Licensed under the Apache License, Version 2.0. See LICENSE.
from __future__ import annotations
'Grouped old-policy collection: no partial commits or premature safe points.'
import asyncio
from dataclasses import replace
from types import SimpleNamespace
import pytest
from verl_speco.integration.multinode_online import AsyncCollectionRequest, OnlineCollectionCoordinator as AsyncMultiNodeCollectionCoordinator
from verl_speco.integration.multinode_online import OnlineVersions
from verl_speco.transport.streaming import SampleMetadataV3, make_v3_sample_key

def metadata(request, sequence):
    return SampleMetadataV3(3, 'group-run', 1, 'p0', 'producer-inc', request, 'sample-' + request, sequence, '9', 11, 'tokenizer', 'target', (1, 12, 22, 32, 43), 'dflash_aux_plus_last', 'bfloat16', 4, 4 * (6 * 4 * 2 + 20), rl_step=6, draft_revision='4', prompt_token_count=2, supervised_token_count=2)

class Ports:

    def __init__(self):
        self.events = []
        self.active = None
        self.fail = None
        self.pause = None

    async def rpc(self, method, kwargs):
        if method.endswith('_arm'):
            self.active = kwargs['request_id']
            self.events.append(('arm', self.active))
            return [dict(rank=i, armed=True, request_id=self.active) for i in (0, 1)]
        if method.endswith('_finish'):
            return [dict(rank=i, rows=4, request_id=self.active) for i in (0, 1)]
        if method.endswith('_publish'):
            v3 = SampleMetadataV3.from_dict(kwargs['metadata'])
            if self.fail == 'publish-second' and self.active == 'r1':
                raise RuntimeError('owned publication lost')
            self.events.append(('publish', self.active))
            common = dict(key=make_v3_sample_key(v3), request_id=v3.request_id, run_id=v3.run_id, generation=v3.generation, payload_nbytes=v3.payload_nbytes, payload_checksum='a' * 64, data_version=v3.data_version, target_revision=v3.target_revision, draft_revision=v3.draft_revision, producer_incarnation=v3.producer_incarnation)
            return [dict(common, rank=0, status='published', native_descriptor=dict(run_id=v3.run_id, generation=v3.generation, lease_id=common['key'])), dict(common, rank=1, status='rank_skipped')]
        if method.endswith('_release'):
            self.events.append(('release', self.active))
            return [dict(rank=i, released=True) for i in (0, 1)]
        raise AssertionError(method)

    async def save(self, v3, prompt, response):
        self.events.append(('save', v3.request_id))

    async def abort(self, request, reason):
        self.events.append(('abort', request))
        if self.fail == 'cleanup-first' and request == 'r1':
            raise RuntimeError('first cleanup failed')

    def request(self, index, *, dynamic=False):
        request = 'r' + str(index)

        async def stream():
            self.events.append(('generate', request))
            if request == 'r1':
                if self.pause is not None:
                    await self.pause.wait()
                if self.fail in ('cancel-second', 'cleanup-first'):
                    raise asyncio.CancelledError()
            yield SimpleNamespace(finished=True, prompt_token_ids=[1, 2], outputs=[SimpleNamespace(token_ids=[3, index + 4, 5], finish_reason='length')])

        def build(versions, prompt, response):
            assert versions == OnlineVersions('group-run', 1, 6, 9, 11, 4)
            return metadata(request, index).to_dict()
        return AsyncCollectionRequest(request, (1, 2), stream, build if dynamic else metadata(request, index).to_dict(), {})

def coordinator(p):
    return AsyncMultiNodeCollectionCoordinator(versions=OnlineVersions('group-run', 1, 5, 9, 10, 4), collective_rpc=p.rpc, expected_ranks=(0, 1), producer_id='p0', save_complete_sequence=p.save, abort_collection=p.abort)

@pytest.mark.parametrize('fail', [False, True])
def test_admission_is_awaited_before_any_arm_or_generation(fail):

    async def run():
        p = Ports()
        c = coordinator(p)

        async def admit(versions, requests):
            assert versions == OnlineVersions('group-run', 1, 6, 9, 11, 4)
            assert len(requests) == 2 and (not p.events)
            p.events.append(('admit', 'group'))
            if fail:
                raise RuntimeError('insufficient retained source capacity')
        c.admit_collection = admit
        if fail:
            with pytest.raises(RuntimeError, match='capacity'):
                await c.collect_group(collection_id='group', requests=(p.request(0), p.request(1)))
            assert p.events == [('admit', 'group')]
            assert c.failed and c.pending_collection is None
        else:
            await c.collect_group(collection_id='group', requests=(p.request(0), p.request(1)))
            assert p.events[0] == ('admit', 'group')
            assert p.events[1] == ('arm', 'r0')
    asyncio.run(run())

@pytest.mark.parametrize('dynamic', [False, True])
def test_group_commits_all_keys_at_one_old_policy_boundary(dynamic):

    async def run():
        p = Ports()
        c = coordinator(p)
        commit = await c.collect_group(collection_id='group', requests=tuple((p.request(i, dynamic=dynamic) for i in (0, 1))))
        assert commit.tq_keys == tuple((make_v3_sample_key(metadata('r' + str(i), i)) for i in (0, 1)))
        assert commit.versions == OnlineVersions('group-run', 1, 6, 9, 11, 4)
        assert p.events == [(event, 'r' + str(i)) for i in (0, 1) for event in ('arm', 'generate', 'save', 'publish', 'release')]
        assert c.versions == OnlineVersions('group-run', 1, 5, 9, 10, 4)
        with pytest.raises(RuntimeError):
            await c.collect_group(collection_id='next', requests=(p.request(2),))

        async def publish():
            p.events.append(('all_rank_ack', 'group'))
        await c.commit_safe_point(versions=replace(commit.versions, target_revision=10, draft_revision=5), publish=publish)
        assert c.pending_collection is None and c.versions.target_revision == 10
    asyncio.run(run())

def test_group_lock_prevents_safe_point_between_responses():

    async def run():
        p = Ports()
        c = coordinator(p)
        p.pause = asyncio.Event()
        task = asyncio.create_task(c.collect_group(collection_id='group', requests=(p.request(0), p.request(1))))
        while ('generate', 'r1') not in p.events:
            await asyncio.sleep(0)
        assert c.pending_collection is None
        published = []

        async def publish():
            published.append('all_rank_ack')
        safe = asyncio.create_task(c.commit_safe_point(versions=OnlineVersions('group-run', 1, 6, 10, 11, 5), publish=publish))
        await asyncio.sleep(0)
        assert not safe.done() and (not published)
        p.pause.set()
        await task
        await safe
        assert published == ['all_rank_ack']
    asyncio.run(run())

@pytest.mark.parametrize('damage', ['duplicate-request', 'foreign-version', 'foreign-prompt', 'explicit-ID'])
def test_group_preflight_rejects_all_static_damage_before_any_generation(damage):

    async def run():
        p = Ports()
        c = coordinator(p)
        requests = (p.request(0), p.request(1))
        if damage == 'duplicate-request':
            requests = (requests[0], replace(requests[1], request_id='r0'))
        elif damage == 'foreign-version':
            requests = (requests[0], replace(requests[1], metadata=replace(metadata('r1', 1), target_revision='10').to_dict()))
        elif damage == 'foreign-prompt':
            requests = (requests[0], replace(requests[1], prompt_token_ids=()))
        else:
            requests = (requests[0], replace(requests[1], request_id='foreign'))
        with pytest.raises(ValueError):
            await c.collect_group(collection_id='group', requests=requests)
        assert not p.events and c.pending_collection is None and (not c.failed)
    asyncio.run(run())

@pytest.mark.parametrize('failure', ['publish-second', 'cancel-second', 'cleanup-first'])
def test_partial_group_aborts_every_owned_request_and_preserves_durable_tokens(failure):

    async def run():
        p = Ports()
        c = coordinator(p)
        p.fail = failure
        with pytest.raises((RuntimeError, asyncio.CancelledError)):
            await c.collect_group(collection_id='group', requests=(p.request(0), p.request(1)))
        assert p.events[-2:] == [('abort', 'r1'), ('abort', 'r0')]
        assert ('save', 'r0') in p.events
        if failure == 'publish-second':
            assert ('save', 'r1') in p.events
        assert c.failed and c.pending_collection is None
        with pytest.raises(RuntimeError):
            await c.collect_group(collection_id='next', requests=(p.request(2),))
    asyncio.run(run())
