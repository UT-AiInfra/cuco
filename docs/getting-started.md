# Getting Started

This guide walks you through installing CUCo and running the included DeepSeek-V3 MoE example end-to-end.

## Prerequisites

### Hardware

- **GPUs**: NVIDIA A100 (or later) with NVLink (intra-node) or RoCE/InfiniBand (inter-node)
- **Multi-GPU**: At least 2 GPUs, potentially across nodes depending on the workload

### Software

<table>
<tr><th>Dependency</th><th>Version</th><th>Notes</th></tr>
<tr><td>Python</td><td>&gt;= 3.10</td><td>3.10, 3.11, or 3.12</td></tr>
<tr><td>CUDA</td><td>&gt;= 13.1</td><td>nvcc compiler</td></tr>
<tr><td>NCCL</td><td>&gt;= 2.28.9</td><td>Must include device-side API headers (gin.h, etc.)</td></tr>
<tr><td>MPI</td><td>OpenMPI</td><td>For mpirun-based multi-GPU execution</td></tr>
<tr><td>Git</td><td>Any</td><td>For cloning the repository</td></tr>
</table>

### LLM API Access

CUCo requires access to at least one LLM provider. The default configuration uses **Anthropic Claude via AWS Bedrock**. See [LLM Backends](llm-backends.md) for all supported providers.

## Installation

### Clone and install

```bash
git clone https://github.com/UT-AiInfra/cuco.git
cd cuco

python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Or with [uv](https://docs.astral.sh/uv/) (faster):

```bash
uv venv
source .venv/bin/activate
uv sync
```

### Configure LLM credentials

Create a `.env` file in the repository root:

```bash
# AWS Bedrock (default provider)
AWS_DEFAULT_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION_NAME=us-east-1
```

CUCo loads this file automatically via `python-dotenv`. See [LLM Backends](llm-backends.md) for other providers (OpenAI, Gemini, DeepSeek, etc.).

### Verify CUDA/NCCL paths

The example `evaluate.py` expects these paths (edit if your installation differs):

```python
NVCC = "/usr/local/cuda-13.1/bin/nvcc"
NCCL_INCLUDE = "/usr/local/nccl_2.28.9-1+cuda13.0_x86_64/include"
NCCL_STATIC_LIB = "/usr/local/nccl_2.28.9-1+cuda13.0_x86_64/lib/libnccl_static.a"
```

### Configure multi-node (if applicable)

For inter-node experiments, create a hostfile at `examples/ds_v3_moe/build/hostfile`:

```
node1 slots=1
node2 slots=1
```

## Running the Example

The included example is a DeepSeek-V3 MoE (Mixture of Experts) dispatch-compute-combine workload running across 2 GPUs over inter-node RoCE links.

### Step 1: Fast-Path Transformation (optional)

If starting from a host-driven NCCL program, the fast-path agent converts it to device-initiated GIN:

```bash
cd examples/ds_v3_moe
python run_transform.py
```

This runs the three-step pipeline (analyze, host-to-device, evolve-block annotation) and writes the transformed kernel to `_transform_host_output/`. The included `ds_v3_moe.cu` seed already has EVOLVE-BLOCK markers, so this step is optional for the provided example.

**Options:**

<table>
<tr><th>Flag</th><th>Default</th><th>Description</th></tr>
<tr><td><code>--source</code></td><td><code>ds_v3_moe.cu</code></td><td>Input CUDA source file</td></tr>
<tr><td><code>--no-agent</code></td><td>off</td><td>Use structured LLM loop instead of Claude Code agent</td></tr>
<tr><td><code>--model</code></td><td><code>sonnet</code></td><td>Claude model for agent mode</td></tr>
<tr><td><code>--max_iterations</code></td><td><code>5</code></td><td>Max iterations in no-agent mode</td></tr>
</table>

### Step 2: Evolutionary Search

Run the slow-path agent to optimize the kernel:

```bash
cd examples/ds_v3_moe
python run_evo.py --num_generations=18 --api=gin
```

This runs a two-phase evolution:
- **Phase 1 (explore)**: First 40% of generations. High-temperature full rewrites to discover diverse architectures.
- **Phase 2 (exploit)**: Remaining 60%. Low-temperature diff patches to refine the best designs.

**Options:**

<table>
<tr><th>Flag</th><th>Default</th><th>Description</th></tr>
<tr><td><code>--num_generations</code></td><td><code>60</code></td><td>Total generation budget</td></tr>
<tr><td><code>--results_dir</code></td><td><code>results_ds_v3_moe</code></td><td>Output directory for results</td></tr>
<tr><td><code>--api</code></td><td><code>gin</code></td><td>Communication API: <code>gin</code> or <code>lsa</code></td></tr>
<tr><td><code>--explore_fraction</code></td><td><code>0.4</code></td><td>Fraction of budget for explore phase</td></tr>
<tr><td><code>--init_program</code></td><td><code>ds_v3_moe.cu</code></td><td>Seed program path</td></tr>
<tr><td><code>--gin_ref</code></td><td><code>None</code></td><td>Reference GIN example for prompts</td></tr>
<tr><td><code>--lsa_ref</code></td><td><code>None</code></td><td>Reference LSA example for prompts</td></tr>
</table>

### Step 3: Monitor Progress

During evolution, watch the console for per-generation output:

```
Generation 3 — Score: 83.86 (best: 83.86)
  Time: 118.26 ms | Correct: True | Patch: full | Model: claude-opus-4-6
```

Results are saved to `results_ds_v3_moe/` as they complete.

### Step 4: Visualize Results

Launch the interactive web UI:

```bash
cuco_visualize --db examples/ds_v3_moe/results_ds_v3_moe/evolution_db.sqlite --open
```

This opens a browser with:
- **Tree view**: Lineage tree showing parent-child relationships and scores
- **Programs table**: Sortable table of all candidates with metrics
- **Embeddings**: Similarity heatmap and clustering
- **Meta scratchpad**: Cross-generation optimization recommendations

See [Visualization](visualization.md) for details.

## Understanding Results

### Directory structure

After evolution, `results_ds_v3_moe/` contains:

```
results_ds_v3_moe/
├── evolution_db.sqlite         # SQLite database of all candidates
├── experiment_config.yaml      # Configuration snapshot
├── meta_memory.json            # Meta-learning state
├── best/                       # Symlink to best generation
│   ├── ds_v3_moe.cu            # Best evolved kernel
│   └── results/
│       └── metrics.json        # Best score and timing
├── gen_0/                      # Generation 0 (seed)
│   ├── ds_v3_moe.cu            # Evolved program
│   ├── original.cu             # Parent code
│   ├── main.cu                 # Copy used for evaluation
│   ├── edit.diff               # Mutation diff
│   ├── rewrite.txt             # LLM output
│   └── results/
│       ├── metrics.json        # Score, timing, feedback
│       ├── correct.json        # Correctness result
│       ├── build.log           # Compiler output
│       └── run.log             # Runtime output
├── gen_1/
│   └── ...
└── meta_8.txt                  # Meta-summary at generation 8
```

### Interpreting metrics.json

```json
{
  "combined_score": 83.85,
  "public": {
    "time_ms": 118.26,
    "rank0_time_ms": 141.83,
    "rank0_tokens": 6144,
    "rank1_time_ms": 47.54,
    "rank1_tokens": 2048,
    "all_run_times_ms": [118.58, 118.26]
  },
  "text_feedback": "LLM suggestions: ..."
}
```

- **combined_score**: `10000 / (1 + time_ms)` — higher is better
- **time_ms**: Token-weighted average time across ranks
- **text_feedback**: LLM-generated optimization suggestions

## Next Steps

- [Adding a New Workload](adding-a-workload.md) — Adapt CUCo for your own kernels
- [Configuration Reference](configuration.md) — Full parameter documentation
- [Fast-Path Agent](fast-path-agent.md) — Deep dive into the transformation pipeline
- [Slow-Path Agent](slow-path-agent.md) — Deep dive into the evolutionary search
