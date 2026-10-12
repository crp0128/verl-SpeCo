# Licensed under the Apache License, Version 2.0. See LICENSE.
"""Public multi-node planning, standalone, sequence generation and online entry."""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
from pathlib import Path
from typing import Sequence

from verl_speco.runtime.planning import resolve_launch_plan


def read_config(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        result = json.loads(text)
    else:
        import yaml
        result = yaml.safe_load(text)
    if not isinstance(result, dict):
        raise ValueError("configuration must be a mapping")
    return result


def resolve(config: dict, rank_table=None) -> dict:
    plan = resolve_launch_plan(config, rank_table)
    payload = plan.to_dict()
    runtime = config['speco'].get('multinode_runtime', config['speco'].get('speco_runtime'))
    if plan.mode == 'standalone' and runtime.get('execution', {}).get('backend', 'external_ray') == 'external_ray':
        from verl_speco.runtime.external_standalone import external_tq_binding
        payload['external_transfer_queue'] = external_tq_binding(config, plan)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("dry-run", "standalone", "sequence-only", "online"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--rank-table", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    config = read_config(args.config)
    if args.command == "sequence-only":
        from verl_speco.standalone_tq_producer import run_producer
        config = copy.deepcopy(config)
        config.setdefault("speco", {}).setdefault("standalone_tq_producer", {})["mode"] = "sequence_only"
        asyncio.run(run_producer(config))
        return 0
    if args.command == "online":
        # Explicit functional MultiNode profile retains its real actor/draft
        # receipts, graph publication and joint recovery constraints.
        from verl_speco.runtime.training_runtime import online_data_contract
        online_data_contract(config)
        from verl_speco.runtime.training_runtime import run
        asyncio.run(run(config))
        return 0
    table = None if args.rank_table is None else read_config(args.rank_table)
    plan = resolve_launch_plan(config, table)
    if args.command == "dry-run":
        payload = json.dumps(resolve(config, table), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        if args.output:
            args.output.write_text(payload, encoding="utf-8")
        else:
            print(payload, end="")
        return 0
    if plan.mode != 'standalone':
        raise ValueError('standalone command requires runtime.mode=standalone')
    runtime = config["speco"].get("multinode_runtime", config["speco"].get("speco_runtime"))
    if runtime.get("external_services") is not True:
        raise ValueError("standalone currently requires explicit external_services=true")
    if not isinstance(runtime.get("execution"), dict):
        raise ValueError("standalone requires an explicit execution configuration")
    if runtime["execution"].get("backend", "external_ray") != "external_ray":
        raise ValueError("select external_ray; native MultiNode service deployment requires its separately validated profile")
    from verl_speco.runtime.external_standalone import run_external_standalone
    return run_external_standalone(config, plan)


if __name__ == "__main__":
    raise SystemExit(main())
