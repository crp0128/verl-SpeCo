# CI layers

The repository uses three workflow layers:

- `cpu_unit_tests.yml`: required PR checks without installing this repository or using accelerator runtimes.
- `gpu_example_tests.yml`: scheduled/manual vLLM and SGLang example-script runs on GPU.
- `npu_example_tests.yml`: scheduled/manual vLLM and SGLang example-script runs on NPU.

The hardware workflows require self-hosted runner labels:

- GPU: `self-hosted`, `linux`, `x64`, `gpu`
- NPU: `self-hosted`, `linux`, `x64`, `npu`

Like verl's CI, the hardware workflows assume the runner image has the runtime
stack and a small default model/data cache. By default they look under:

- `/home/runner/models`
- `/home/runner/models/hf_data`

The default paths are intentionally tiny-runner friendly and can be overridden
with GitHub environment variables or manual workflow inputs.

The CPU layer uses `PYTHONPATH=$PWD` and checks out the upstream verl commit
from `REQUIRED_VERL.txt`. It runs:

```bash
python -m compileall verl_speco
bash -n examples/*.sh
python -m pytest tests/compat tests/config tests/examples tests/integration -q
```

The hardware layers call `ci/run_example_test.sh`, which selects one of the
repository-owned scripts in `examples/` and passes CI variables as Hydra
overrides. Configure these variables in the `speco-gpu-ci` and `speco-npu-ci`
GitHub environments, or pass them as manual workflow inputs where available:

- `SPECO_MODEL_ROOT`
- `SPECO_DATA_ROOT`
- `SPECO_TARGET_MODEL`
- `SPECO_EAGLE3_DRAFT_MODEL`
- `SPECO_DFLASH_DRAFT_MODEL`
- `SPECO_TRAIN_FILE`
- `SPECO_TEST_FILE`
- `SPECO_CKPT_DIR`
- `SPECO_ACCELERATOR_COUNT`
- `SPECO_TENSOR_PARALLEL_SIZE`
- `SPECO_SEQUENCE_PARALLEL_SIZE`
- `SPECO_ENABLE_TRAINING`
- `SPECO_SPEC_STEPS`
- `SPECO_SPEC_TOPK`
- `SPECO_SPEC_VERIFY_TOKENS`
- `SPECO_DFLASH_NUM_ANCHORS`
- `SPECO_DFLASH_MAX_WINDOW`
- `SPECO_EXTRA_HYDRA_ARGS`

The runner image is responsible for providing the matching verl, vLLM/SGLang,
PyTorch accelerator runtime, and model files. Hardware workflows deliberately
fail closed when required models or datasets are absent.

## Testing CI locally

Run the CPU layer checks from the repository root:

```bash
python -m compileall verl_speco
bash -n examples/*.sh
bash -n ci/run_example_test.sh
python -m pytest tests/compat tests/config tests/examples tests/integration -q
```

On Windows, prefer Git for Windows bash when WSL is not installed:

```powershell
D:\git\bin\bash.exe -n examples/*.sh
D:\git\bin\bash.exe -n ci/run_example_test.sh
python -m pytest tests/compat tests/config tests/examples tests/integration -q
```

To test the hardware workflows on GitHub, open Actions, choose
`gpu_example_tests` or `npu_example_tests`, and run the workflow without inputs
after preparing the default paths above. Fill the manual inputs only when you
want to override the defaults for one run.
