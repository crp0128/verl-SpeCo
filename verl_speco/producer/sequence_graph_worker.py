# Licensed under the Apache License, Version 2.0. See LICENSE.
"""Consolidated multi-node capabilities. Hardware adapters load only when explicitly selected."""
from __future__ import annotations
from verl_speco.integration.accelerator_speculation import install_sequence_graph_mome
from verl_speco.integration.accelerator_speculation import install_sequence_graph_attention
from verl_speco.producer.capture import SpecoMultiNodeCaptureWorkerExtension as SpecoMultiNodeCaptureWorkerExtension
install_sequence_graph_mome()
install_sequence_graph_attention()
import os
if os.environ.get('SPECO_MHC_SP_PREPROFILE') == '1':
    from verl_speco.integration.accelerator_speculation import install_preprofile_sp
    install_preprofile_sp()

class SpecoFullDecodeSequenceWorkerExtension(SpecoMultiNodeCaptureWorkerExtension):
    pass
