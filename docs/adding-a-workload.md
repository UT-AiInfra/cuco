# Adding a New Workload

This guide walks through setting up a new CUDA kernel for optimization with CUCo.

## Quick Start

Three commands to go from seed kernel to running evolution:

```bash
# 1. One-time cluster setup (auto-detects CUDA, NCCL, MPI, GPU)
cuco_init --setup

# 2. Scaffold a new workload from your seed kernel
cuco_init /path/to/my_kernel.cu

# 3. Run evolution
cuco_run my_kernel --generations 50
```

`cuco_init` creates a ready-to-run workload directory with all required files, auto-detects your cluster environment, validates the seed file, and prints the exact run command. See `cuco_init --help` for all options.

### What `cuco_init` does

- Creates `workloads/<name>/` with your seed file
- Generates `evaluate.py` pre-configured with your cluster's CUDA/NCCL/MPI paths (from `~/.cuco/site.yaml`)
- Copies `run_evo.py`, `run_transform.py`, `nccl_api_docs.py` with correct defaults
- Generates `.gitignore`
- Validates that your seed prints `Time: X.XXXX ms` and `Verification: PASS`
- Prints a review checklist and the exact run command

### Manual setup (alternative)

If you prefer not to use `cuco_init`, you can copy and customize directly:

```bash
cp -r workloads/ds_v3_moe workloads/my_workload
cp /path/to/my_kernel.cu workloads/my_workload/
cd workloads/my_workload
python run_evo.py --init_program my_kernel.cu --results_dir results_my_workload --num_generations 18
```

## Overview

A CUCo workload directory contains:

```
workloads/my_workload/
├── my_kernel.cu          # Seed CUDA kernel (host-driven or device-initiated)
├── evaluate.py           # Build, run, correctness check, fitness scoring
├── run_evo.py            # Evolution launcher with prompt customization
├── run_transform.py      # Fast-path transformation launcher (optional)
├── nccl_api_docs.py      # API documentation / reference material for LLM context
└── .gitignore            # Excludes build artifacts, results, __pycache__
```

When using `cuco_init`, all of these files are generated automatically. When copying from `ds_v3_moe`, they come with the directory.

## Step 1: Prepare the Seed Kernel

The seed kernel is the starting point for evolution. It can be either:

- **Host-driven** (standard NCCL collectives called from CPU) — CUCo's fast-path agent will convert it to device-initiated code before evolution begins.
- **Device-initiated** (already using GIN or LSA) — Evolution starts directly.

### Requirements

1. The kernel must be a single `.cu` file that compiles with `nvcc`.
2. It must run via `mpirun -np N` across the target GPU configuration.
3. The kernel must print a timing line that `evaluate.py` can parse. The default regex expects:
   ```
   Time: X.XXXX ms
   ```
   If your kernel prints timing in a different format, either add a `printf("Time: %.4f ms\n", elapsed);` line, or update `TIME_PATTERN` in `evaluate.py` (see Step 3).
4. The kernel must print `"Verification: PASS"` when results are correct.

Requirements 3 and 4 are not hard constraints — the evaluation logic is entirely defined in `evaluate.py`, so you can implement any scoring scheme (timing, accuracy, throughput, memory usage, or a combination). The kernel just needs to produce output that your `evaluate.py` knows how to parse.

### EVOLVE-BLOCK Markers

If your kernel is already device-initiated or you want to skip the fast-path, add EVOLVE-BLOCK markers manually:

```cuda
// EVOLVE-BLOCK-START
__global__ void myCommKernel(...) {
    // Communication logic — CUCo can modify this
}

__global__ void myComputeKernel(...) {
    // Compute logic — CUCo can modify this
}
// EVOLVE-BLOCK-END

// FROZEN: MPI/NCCL initialization, main(), verification
int main(int argc, char** argv) {
    // ... MPI_Init, ncclCommInitRank, etc.

    // EVOLVE-BLOCK-START
    // Pipeline: stream creation, memory allocation, kernel launches, timing
    cudaEventRecord(start, stream);
    myCommKernel<<<...>>>();
    myComputeKernel<<<...>>>();
    cudaEventRecord(stop, stream);
    // EVOLVE-BLOCK-END

    // FROZEN: verification, cleanup
    printf("Verification: PASS\n");
    printf("Time: %.4f ms\n", elapsed);
}
```

**Rules followed by Fast-Path Agent for EVOLVE-BLOCK placement:**

<table>
  <tr>
    <th>Region</th>
    <th>Mutable?</th>
    <th>Why</th>
  </tr>
  <tr>
    <td>Kernel definitions (<code>__global__</code>, <code>__device__</code>)</td>
    <td>Yes</td>
    <td>Core optimization target</td>
  </tr>
  <tr>
    <td>Pipeline logic (streams, launches, events)</td>
    <td>Yes</td>
    <td>Architecture exploration</td>
  </tr>
  <tr>
    <td>MPI/NCCL initialization</td>
    <td>No</td>
    <td>Breaking this breaks everything</td>
  </tr>
  <tr>
    <td><code>main()</code> signature and MPI setup</td>
    <td>No</td>
    <td>Fixed infrastructure</td>
  </tr>
  <tr>
    <td>Verification and output formatting</td>
    <td>No</td>
    <td>Evaluation depends on exact output format</td>
  </tr>
  <tr>
    <td>Warmup section</td>
    <td>No</td>
    <td>Removing it inflates timing by 10–50 ms</td>
  </tr>
</table>

## Step 2: Write an API Documentation File

The purpose of this file is to supply the LLM with all the reference material it needs — API documentation, usage examples, header snippets, or any other context — so it can generate correct code for APIs or libraries that may be poorly represented in its training data. The file name is arbitrary (e.g., `nccl_api_docs.py`, `cuda_graph_docs.py`, `my_lib_reference.py`); what matters is that your `run_evo.py` imports the variables and injects them into the prompt.

The included `workloads/ds_v3_moe/nccl_api_docs.py` is one such file, providing NCCL device-initiated API documentation. It exports these variables:

<table>
  <tr>
    <th>Variable</th>
    <th>Content</th>
  </tr>
  <tr>
    <td><code>NCCL_DEVICE_API_REFERENCE</code></td>
    <td>Overview of device-initiated communication (GIN, LSA, teams, thread groups)</td>
  </tr>
  <tr>
    <td><code>NCCL_GIN_API_DOC</code></td>
    <td>GIN-specific API: <code>ncclGin</code>, <code>put</code>, <code>flush</code>, signals, barriers</td>
  </tr>
  <tr>
    <td><code>NCCL_LSA_API_DOC</code></td>
    <td>LSA-specific API: barriers, <code>ncclGetLsaPointer</code>, peer access</td>
  </tr>
  <tr>
    <td><code>NCCL_GIN_PURE_EXAMPLE</code></td>
    <td>Complete working GIN AlltoAll example</td>
  </tr>
  <tr>
    <td><code>NCCL_THREAD_GROUPS_DOC</code></td>
    <td><code>ncclCoopThread</code>, <code>ncclCoopWarp</code>, <code>ncclCoopCta</code></td>
  </tr>
  <tr>
    <td><code>NCCL_TEAMS_DOC</code></td>
    <td><code>ncclTeamWorld</code>, <code>ncclTeamLsa</code>, <code>ncclTeamRail</code></td>
  </tr>
  <tr>
    <td><code>NCCL_HOST_TO_DEVICE_COOKBOOK</code></td>
    <td>GIN kernel signature and host launch pattern</td>
  </tr>
  <tr>
    <td><code>NCCL_HEADER_*</code></td>
    <td>Raw C++ header snippets (gin.h, core.h, coop.h, etc.)</td>
  </tr>
</table>

**For a new workload** that also uses NCCL device APIs, you can typically reuse this file unchanged (it's included when you copy the directory). For workloads that use entirely different libraries or custom APIs, create your own docs file exporting string variables with the relevant documentation, then import and inject them into the prompt in `run_evo.py`.

## Step 3: Review and Adapt evaluate.py

`evaluate.py` is the bridge between CUCo and your hardware. The framework calls it automatically for every candidate program:

```bash
python evaluate.py --program_path gen_5/main.cu --results_dir gen_5/results
```

It handles the full pipeline: **build** the `.cu` file, **run** the binary, **check correctness**, **parse timing**, and **write** `metrics.json` + `correct.json`.

The source and binary filenames are derived automatically from `--program_path` — you do not need to change them per workload. However, you may need to review and adapt the following sections for your environment.

### Build Toolchain Paths

When using `cuco_init`, the generated `evaluate.py` imports all cluster-specific paths from `cuco.site_config`, which reads from `~/.cuco/site.yaml` (auto-detected during `cuco_init --setup`). You do not need to edit any paths manually.

To change cluster settings (e.g., when moving to a different machine), either:
- Re-run `cuco_init --setup`, or
- Edit `~/.cuco/site.yaml` directly

If using the manual setup (copied from `ds_v3_moe`), the original `evaluate.py` has hardcoded paths that you'll need to update.

Key build flags:
- `-rdc=true` — Required for device-side NCCL (relocatable device code)
- Static NCCL linking (`libnccl_static.a`) is required for device-initiated APIs (GIN, LSA)

### Run Configuration

`MPI_NP` controls how many ranks `mpirun` launches. It must match the `NUM_RANKS` in your seed kernel:

```python
MPI_NP = 2
```

### Network and Topology

When using `cuco_init`, the generated `evaluate.py` uses `build_mpirun_cmd()` from `cuco.site_config`, which constructs the `mpirun` command dynamically based on `~/.cuco/site.yaml`:

- **Single-node** (default): Simple `mpirun -np N` with no hostfile or IB flags.
- **Multi-node**: Includes `--hostfile`, NCCL network flags, and MCA TCP settings.

To switch between single-node and multi-node, run `cuco_init --setup` and answer the multi-node prompt, or edit `~/.cuco/site.yaml` directly:

```yaml
multi_node: true
hostfile: /path/to/hostfile
network:
  nccl_socket_ifname: enp75s0f1np1
  nccl_ib_hca: mlx5_1
  nccl_ib_gid_index: 3
  btl_tcp_if_include: enp75s0f1np1
  oob_tcp_if_include: enp75s0f1np1
```

All workloads on the same cluster share this config — change it once, and all workloads pick up the new settings.

### Timing and Verification Patterns

`evaluate.py` parses program output using these regexes:

```python
TIME_PATTERN = re.compile(r"^Time:\s*([\d.]+)\s*ms", re.MULTILINE)
VERIFICATION_PASS_STR = "Verification: PASS"
```

**Your kernel must print output that matches these patterns.** The simplest approach is to add a matching `printf` to your kernel:

```c
printf("Time: %.4f ms\n", elapsed_ms);
printf("Verification: PASS\n");
```

If your kernel prints timing in a different format, update the regex instead. For example, if your kernel prints `"Elapsed: 5.23 ms"`:

```python
TIME_PATTERN = re.compile(r"^Elapsed:\s*([\d.]+)\s*ms", re.MULTILINE)
```

For multi-rank workloads with unequal token counts, the evaluator also supports token-weighted averaging via:

```python
RANK_HEADER_PATTERN = re.compile(
    r"RESULTS \(Rank (\d+).*?receives (\d+) tokens\).*?\nTime:\s*([\d.]+)\s*ms",
    re.DOTALL,
)
```

### Scoring Function

The default fitness function maps lower time to higher score:

```python
def score_from_time_ms(time_ms: float) -> float:
    return 10000.0 / (1.0 + time_ms)
```

This works for most latency-focused workloads. For multi-objective optimization (e.g., balancing latency and memory), modify this function and add sub-metrics to the `public` dict in `metrics.json` so the LLM can see the breakdown:

```python
def compute_score(time_ms, memory_mb, accuracy):
    latency_score = 10000.0 / (1.0 + time_ms)
    memory_score = 1000.0 / (1.0 + memory_mb)
    return 0.7 * latency_score + 0.2 * memory_score + 0.1 * accuracy
```

### Output Files

`evaluate.py` writes exactly two files:

**metrics.json:**
```json
{
  "combined_score": 83.85,
  "public": {
    "time_ms": 118.26,
    "throughput_gb_s": 42.5,
    "build_duration_sec": 6.72
  },
  "private": {
    "peak_memory_mb": 1024,
    "nvcc_warnings": 2
  },
  "text_feedback": "Optional LLM-generated suggestions"
}
```

**correct.json:**
```json
{
  "correct": true
}
```

Or on failure:
```json
{
  "correct": false,
  "error": "Build failed: undefined reference to ncclGin"
}
```

See [Evaluation](evaluation.md) for the full reference.

## Step 4: Write run_transform.py (optional)

If your seed kernel uses host-side NCCL collectives, write a transformation launcher:

```python
from cuco.transform import CUDAAnalyzer, HostToDeviceTransformer
from cuco.transform.transformer import TransformConfig

config = TransformConfig(
    rewrite_model="bedrock/us.anthropic.claude-sonnet-4-6",
    nvcc_path="/usr/local/cuda-13.1/bin/nvcc",
    nccl_include="/usr/local/nccl_2.28.9-1+cuda13.0_x86_64/include",
    nccl_static_lib="/usr/local/nccl_2.28.9-1+cuda13.0_x86_64/lib/libnccl_static.a",
    num_mpi_ranks=2,
    hostfile="build/hostfile",
    api_type="gin",
    two_stage=True,
    reference_code=open("reference_gin_example.cu").read(),
    nccl_api_docs=NCCL_API_FULL,
)

transformer = HostToDeviceTransformer(config)
result = transformer.transform("my_kernel.cu", work_dir="_transform_work")
```

When copying from `ds_v3_moe`, the `run_transform.py` is already included. Pass your kernel via CLI:

```bash
python run_transform.py --source my_kernel.cu
```

See [Fast-Path Agent](fast-path-agent.md) for details on the two-stage transformation.

## Step 5: Write run_evo.py

This is the main evolution launcher. When copying from `ds_v3_moe`, pass workload-specific values via CLI:

```bash
python run_evo.py \
    --init_program my_kernel.cu \
    --results_dir results_my_workload \
    --num_generations 18 \
    --api gin
```

Key decisions if you need to customize beyond CLI arguments:

### Prompt Customization

Build a task system message that tells the LLM what to optimize:

```python
_COMMON_CONSTRAINTS = """
## Evolve Block Structure
The file contains EVOLVE-BLOCK regions. You have FULL AUTONOMY over code inside these blocks.

## Hard Constraints
- Only modify code between EVOLVE-BLOCK-START and EVOLVE-BLOCK-END markers.
- Correctness: the program must print "Verification: PASS".
- Do NOT use external libraries (cuBLAS, cuDNN, etc.).
"""

_STRATEGIES = """
## Choosing the Right Fusion Level
### Kernel-Level Fusion
Fuse compute and communication into a single persistent kernel when...
### Stream-Level Overlap
Use separate CUDA streams when...
### Split Communication Kernels
Break into separate PUT and WAIT launches when...
"""
```

Include API-specific knowledge (GIN rules, LSA rules) and hardware context.

### Phase Configuration

```python
_PHASE_CONFIGS = {
    "explore": {
        "patch_type_probs": [0.15, 0.70, 0.15],  # diff, full, cross
        "temperatures": [0.2, 0.5, 0.8],
    },
    "exploit": {
        "patch_type_probs": [0.25, 0.60, 0.15],
        "temperatures": [0.0, 0.2, 0.5],
    },
}
```

### Config Assembly

```python
from cuco.core import EvolutionRunner, EvolutionConfig
from cuco.database import DatabaseConfig
from cuco.launch import LocalJobConfig

evo_config = EvolutionConfig(
    task_sys_msg=task_msg,
    patch_types=["diff", "full", "cross"],
    patch_type_probs=[0.15, 0.70, 0.15],
    num_generations=18,
    max_parallel_jobs=1,
    language="cuda",
    llm_models=["bedrock/us.anthropic.claude-opus-4-6-v1"],
    llm_kwargs=dict(temperatures=[0.2, 0.5, 0.8], max_tokens=32768),
    meta_rec_interval=8,
    meta_max_recommendations=5,
    init_program_path="my_kernel.cu",
    results_dir="results_my_workload",
    embedding_model="bedrock-amazon.titan-embed-text-v1",
    use_text_feedback=True,
)

db_config = DatabaseConfig(
    db_path="evolution_db.sqlite",
    num_islands=2,
    archive_size=60,
    migration_interval=8,
    parent_selection_strategy="weighted",
)

job_config = LocalJobConfig(eval_program_path="evaluate.py")

runner = EvolutionRunner(
    evo_config=evo_config,
    job_config=job_config,
    db_config=db_config,
)
runner.run()
```

See [Configuration Reference](configuration.md) for all parameters.

## Step 6: Set Up Credentials

CUCo requires LLM API credentials for evolution, feedback, and transformation. Create a `.env` file in the CUCo root directory:

```bash
# For AWS Bedrock (default)
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_REGION_NAME=us-east-1
```

The `.env` file is loaded automatically by `evaluate.py`, `run_evo.py`, and `run_transform.py`.

## Step 7: Test the Pipeline

Before running a full evolution, verify each component:

### 1. Verify the seed builds and runs

```bash
cd workloads/my_workload
# Build
nvcc -o my_kernel my_kernel.cu -I... -rdc=true -arch=sm_80 ...
# Run (single-node)
CUDA_VISIBLE_DEVICES=0,1 mpirun -np 2 ./my_kernel
# Run (multi-node)
mpirun --hostfile build/hostfile -np 2 ./my_kernel
# Should print: "Verification: PASS" and "Time: X.XXXX ms"
```

### 2. Verify evaluate.py

```bash
mkdir -p test_results
python evaluate.py --program_path my_kernel.cu --results_dir test_results
cat test_results/metrics.json    # Should show combined_score > 0
cat test_results/correct.json    # Should show "correct": true
```

### 3. Run a short evolution

```bash
python run_evo.py --init_program my_kernel.cu --results_dir results_test --num_generations 3
```

Check `results_test/gen_0/` for the first generation's output.

## Checklist

Before running a full evolution, verify:

- [ ] `cuco_init --setup` has been run (or `~/.cuco/site.yaml` exists with correct paths)
- [ ] Seed kernel compiles with your nvcc command
- [ ] Seed kernel runs correctly via mpirun on your target GPUs
- [ ] Seed kernel prints `Time: X.XXXX ms` (or you've updated `TIME_PATTERN`)
- [ ] Seed kernel prints `Verification: PASS`
- [ ] `evaluate.py` correctly parses timing and writes `metrics.json` / `correct.json`
- [ ] EVOLVE-BLOCK markers are placed around mutable regions only
- [ ] Frozen regions (init, main, verification) are outside EVOLVE-BLOCK
- [ ] `.env` file has valid LLM API credentials
- [ ] `nccl_api_docs.py` is present and importable

## Common Pitfalls

1. **Timing format mismatch**: The default `evaluate.py` regex expects `^Time:\s*([\d.]+)\s*ms`. If your kernel prints timing differently (e.g., `"Elapsed: 5.23 ms"` or `"Total Pipeline Time: 3.14 ms"`), the evaluator will silently fail to parse it and report score 0. Either add a matching `printf` to your kernel or update `TIME_PATTERN`.

2. **Missing `-rdc=true`**: Device-side NCCL requires relocatable device code. Without it, linking fails silently or produces wrong results.

3. **Wrong NCCL version**: Device-initiated APIs (GIN, LSA) require NCCL >= 2.28.9. Older versions will compile but crash at runtime.

4. **Timing includes warmup**: Always include a warmup section before the timed region. The first GIN/NCCL call triggers lazy RDMA initialization (10-50 ms).

5. **EVOLVE-BLOCK too broad**: If the entire file is mutable, the LLM may break initialization or verification code. Keep EVOLVE-BLOCKs focused on kernels and pipeline logic.

6. **EVOLVE-BLOCK too narrow**: If only a single kernel is mutable, the LLM cannot explore architectural changes (e.g., switching from sequential to pipelined execution).

7. **Network configuration**: Inter-node GIN requires correct `NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA`, and `NCCL_IB_GID_INDEX`. Wrong values cause silent hangs.

8. **Static NCCL linking**: Device APIs require static linking (`libnccl_static.a`), not dynamic (`-lnccl`).

9. **Single-node vs multi-node mismatch**: If running on a single node, make sure `~/.cuco/site.yaml` has `multi_node: false`. When using the manual setup (without `cuco_init`), remove the `--hostfile`, `--map-by node`, and InfiniBand flags from `_run_binary` in `evaluate.py`.
