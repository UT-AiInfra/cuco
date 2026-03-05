# Adding a New Workload

This is the most important guide for external users. It walks through every step needed to adapt CUCo for a new CUDA kernel, using the included DeepSeek-V3 MoE example (`examples/ds_v3_moe/`) as a concrete reference.

## Overview

A CUCo workload consists of five files in a directory:

```
examples/my_workload/
├── my_kernel.cu          # Seed CUDA kernel (host-driven or device-initiated)
├── evaluate.py           # Build, run, correctness check, fitness scoring
├── run_evo.py            # Evolution launcher with prompt customization
├── run_transform.py      # Fast-path transformation launcher (optional)
├── nccl_api_docs.py      # API documentation / reference material for LLM context (any file name works, but should be changed accordingly in run_evo.py)
└── build/
    └── hostfile           # MPI hostfile for multi-node runs
```

## Step 1: Prepare the Seed Kernel

The seed kernel is the starting point for evolution. It can be either:

- **Host-driven** (standard NCCL collectives called from CPU) — CUCo's fast-path agent will convert it to device-initiated code before evolution begins.
- **Device-initiated** (already using GIN or LSA) — Evolution starts directly.

### Requirements

1. The kernel must be a single `.cu` file that compiles with `nvcc`.
2. It must run via `mpirun -np N` across the target GPU configuration.
3. If using timing-based evaluation (the most common case), the kernel should time its critical section using CUDA event timers (`cudaEventRecord` / `cudaEventElapsedTime`) and print the result so `evaluate.py` can parse it (see Step 3).
4. If using adding a correctness-based evaluation, the kernel should print a verification result (e.g., `"Verification: PASS"`) for `evaluate.py` to check.

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

The included `examples/ds_v3_moe/nccl_api_docs.py` is one such file, providing NCCL device-initiated API documentation. It exports these variables:

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

**For a new workload** that also uses NCCL device APIs, you can typically reuse this file unchanged and extend it if needed. For workloads that use entirely different libraries or custom APIs, create your own docs file exporting string variables with the relevant documentation, then import and inject them into the prompt in `run_evo.py`.

## Step 3: Write evaluate.py

The evaluation script is the bridge between CUCo and your hardware. It must:

1. Accept `--program_path` and `--results_dir` CLI arguments
2. Build the candidate `.cu` file
3. Run the compiled binary
4. Check correctness
5. Parse timing
6. Write `metrics.json` and `correct.json`

### CLI Contract

```python
# CUCo calls: python evaluate.py --program_path gen_5/main.cu --results_dir gen_5/results
parser = argparse.ArgumentParser()
parser.add_argument("--program_path", required=True)
parser.add_argument("--results_dir", required=True)
```

### Build Command

Configure for your toolchain:

```python
NVCC = "/usr/local/cuda-13.1/bin/nvcc"
NCCL_INCLUDE = "/usr/local/nccl_2.28.9-1+cuda13.0_x86_64/include"
NCCL_STATIC_LIB = "/usr/local/nccl_2.28.9-1+cuda13.0_x86_64/lib/libnccl_static.a"

def get_build_command(work_dir):
    return [
        NVCC, "-o", str(work_dir / BINARY_NAME), str(work_dir / SOURCE_NAME),
        f"-I{NCCL_INCLUDE}", NCCL_STATIC_LIB,
        "-rdc=true", "-arch=sm_80",
        f"-L{CUDA_LIB64}", "-lcudart", "-lcudadevrt", "-lpthread",
        f"-I{MPI_INCLUDE}", f"-I{MPI_INCLUDE_OPENMPI}",
        f"-L{MPI_LIB}", "-lmpi",
    ]
```

Key flags:
- `-rdc=true` — Required for device-side NCCL (relocatable device code)
- `-arch=sm_80` — Match your GPU architecture (sm_80 for A100)
- Static NCCL linking is required for device-initiated APIs

### Run Command

```python
def get_run_command(work_dir):
    cmd = [
        "mpirun", "--hostfile", str(work_dir / "build" / "hostfile"),
        "-np", str(MPI_NP), "--map-by", "node",
        "-x", "LD_LIBRARY_PATH",
        "-x", "CUDA_VISIBLE_DEVICES=0",
        "-x", "NCCL_GIN_ENABLE=1",          # Enable GIN
        "-x", "NCCL_SOCKET_IFNAME=enp75s0f1np1",  # Network interface
        "-x", "NCCL_IB_HCA=mlx5_1",         # RDMA device
        # ... more environment variables
    ]
    cmd.append(str(work_dir / BINARY_NAME))
    return cmd
```

Adjust `NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA`, and `NCCL_IB_GID_INDEX` for your network configuration.

### Fitness Function

The evolutionary search requires a single scalar `combined_score` (higher is better) for parent selection, archive ranking, and progress tracking. However, you can track **multiple metrics** alongside it:

- **`public`** — a free-form dictionary of metrics that are shown to the LLM in mutation prompts. All key-value pairs are formatted and injected into the context, so the LLM can reason about multiple dimensions (latency, throughput, memory, etc.).
- **`private`** — a free-form dictionary stored in the database but not exposed to the LLM. Use this for diagnostics or internal bookkeeping.

The simplest case maps a single metric to the score:

```python
def score_from_time_ms(time_ms: float) -> float:
    return 10000.0 / (1.0 + time_ms)
```

For multi-objective optimization, you can combine several metrics into `combined_score` using a weighted sum, Pareto ranking, or any custom aggregation — the search only compares the final scalar. Individual sub-metrics should go in `public` so the LLM can see the breakdown:

```python
def compute_score(time_ms, memory_mb, accuracy):
    latency_score = 10000.0 / (1.0 + time_ms)
    memory_score = 1000.0 / (1.0 + memory_mb)
    return 0.7 * latency_score + 0.2 * memory_score + 0.1 * accuracy
```

### Timing Parsing

Match your kernel's output format:

```python
TIME_PATTERN = re.compile(r"^Time:\s*([\d.]+)\s*ms", re.MULTILINE)
VERIFICATION_PASS_STR = "Verification: PASS"
```

For multi-rank workloads with unequal token counts, we suggest using token-weighted averaging:

```python
RANK_HEADER_PATTERN = re.compile(
    r"RESULTS \(Rank (\d+).*?receives (\d+) tokens\).*?\nTime:\s*([\d.]+)\s*ms",
    re.DOTALL,
)
```

### Output Files

Write exactly these two files:

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

See [Fast-Path Agent](fast-path-agent.md) for details on the two-stage transformation.

## Step 5: Write run_evo.py

This is the main evolution launcher. Key decisions:

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

## Step 6: Test the Pipeline

Before running a full evolution, verify each component:

### 1. Verify the seed builds and runs

```bash
cd examples/my_workload
# Build
nvcc -o my_kernel my_kernel.cu -I... -rdc=true -arch=sm_80 ...
# Run
mpirun --hostfile build/hostfile -np 2 ./my_kernel
# Should print: "Verification: PASS" and "Time: X.XXXX ms"
```

### 2. Verify evaluate.py

```bash
mkdir -p test_results
python evaluate.py --program_path my_kernel.cu --results_dir test_results
cat test_results/metrics.json
cat test_results/correct.json
```

### 3. Run a short evolution

```bash
python run_evo.py --num_generations=3
```

Check `results_my_workload/gen_0/` for the first generation's output.

## Checklist

Before running a full evolution, verify:

- [ ] Seed kernel compiles with your nvcc command
- [ ] Seed kernel runs correctly via mpirun on your target GPUs
- [ ] Seed kernel prints timing and verification strings
- [ ] `evaluate.py` correctly parses timing and writes metrics.json / correct.json
- [ ] EVOLVE-BLOCK markers are placed around mutable regions only
- [ ] Frozen regions (init, main, verification) are outside EVOLVE-BLOCK
- [ ] `.env` file has valid LLM API credentials
- [ ] Hostfile is configured for your GPU topology
- [ ] NCCL environment variables match your network (SOCKET_IFNAME, IB_HCA, etc.)
- [ ] `nccl_api_docs.py` is present and importable

## Common Pitfalls

1. **Missing `-rdc=true`**: Device-side NCCL requires relocatable device code. Without it, linking fails silently or produces wrong results.

2. **Wrong NCCL version**: Device-initiated APIs (GIN, LSA) require NCCL >= 2.28.9. Older versions will compile but crash at runtime.

3. **Timing includes warmup**: Always include a warmup section before the timed region. The first GIN/NCCL call triggers lazy RDMA initialization (10-50 ms).

4. **EVOLVE-BLOCK too broad**: If the entire file is mutable, the LLM may break initialization or verification code. Keep EVOLVE-BLOCKs focused on kernels and pipeline logic.

5. **EVOLVE-BLOCK too narrow**: If only a single kernel is mutable, the LLM cannot explore architectural changes (e.g., switching from sequential to pipelined execution).

6. **Network configuration**: Inter-node GIN requires correct `NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA`, and `NCCL_IB_GID_INDEX`. Wrong values cause silent hangs.

7. **Static NCCL linking**: Device APIs require static linking (`libnccl_static.a`), not dynamic (`-lnccl`).
