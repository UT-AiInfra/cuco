# cuda_evolve

## Layout

- **`cuco/`** – evolution core (runner, LLM, database, edit, prompts, launch).
- **`examples/`** – evolution tasks; `split_gemm_allgather/` is the split-K GIN matmul example.
- **`pyproject.toml`** – package and dependencies.
- **`uv.lock`** – locked dependency versions (from OS_Evolve).
- **`.env`** – API keys (Bedrock, etc.); loaded by `cuco/llm/client.py` from this root.

## Setup with uv (recommended)

From this directory:

```bash
cd /path/to/cufuse/cuda_evolve

# Create venv and install project + deps (editable)
uv venv
source .venv/bin/activate   # Linux/macOS
uv sync                     # installs cuco in editable mode and all deps from uv.lock
```

If you add or change dependencies later:

```bash
uv lock
uv sync
```

## Run split-K GIN matmul evolution

With the venv activated:

```bash
cd examples/split_gemm_allgather
python run_evo.py --num_generations=5
```

Or from cuda_evolve root:

```bash
uv run python examples/split_gemm_allgather/run_evo.py --num_generations=5
```

Ensure `.env` in `cuda_evolve/` contains your Bedrock (or other LLM) credentials.

## Alternative: pip (no uv)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Then run evolution from `examples/split_gemm_allgather` as above.
