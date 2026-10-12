# Licensed under the Apache License, Version 2.0. See LICENSE.
"""Consolidated multi-node capabilities. Hardware adapters load only when explicitly selected."""
from __future__ import annotations
from verl_speco.runtime.planning import CapabilityReport, probe_runtime_capabilities
from verl_speco.runtime.planning import ResolvedLaunchPlan, resolve_launch_plan, validate_launch_plan
from verl_speco.runtime.planning import ResourceOwner
