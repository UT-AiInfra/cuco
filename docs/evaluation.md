# Writing Custom Evaluations

The `evaluate.py` script is the bridge between CUCo's evolutionary search and your hardware. It defines how candidate kernels are built, executed, verified, and scored. This document explains the contract and shows how to write one for a new workload.

## Contract

CUCo invokes your evaluation script as:

```bash
python evaluate.py --program_path gen_5/main.cu --results_dir gen_5/results
```

Your script must:
1. Copy the candidate source to a build directory
2. Compile it (e.g., with `nvcc`)
3. Run the binary (e.g., via `mpirun`)
4. Evaluate the result — this is fully customizable (correctness checking, timing, throughput, accuracy, or any combination)
5. Write `metrics.json` and `correct.json` to `--results_dir`

Steps 1-3 are the typical pattern for CUDA workloads, but the only hard contract is that your script accepts `--program_path` and `--results_dir` and writes the two output JSON files. Everything in between is up to you.

## CLI Interface

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--program_path", required=True, help="Path to the candidate .cu file")
parser.add_argument("--results_dir", required=True, help="Directory for output files")
args = parser.parse_args()
```

CUCo expects exactly these two arguments. You can add optional arguments via `JobConfig.extra_cmd_args`.

## Build Configuration

### Toolchain Paths

Define your CUDA/NCCL/MPI installation paths at the top of the file:

```python
NVCC = "/usr/local/cuda-13.1/bin/nvcc"
NCCL_INCLUDE = "/usr/local/nccl_2.28.9-1+cuda13.0_x86_64/include"
NCCL_STATIC_LIB = "/usr/local/nccl_2.28.9-1+cuda13.0_x86_64/lib/libnccl_static.a"
CUDA_LIB64 = "/usr/local/cuda-13.1/lib64"
MPI_INCLUDE = "/usr/lib/x86_64-linux-gnu/openmpi/include"
MPI_INCLUDE_OPENMPI = "/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi"
MPI_LIB = "/usr/lib/x86_64-linux-gnu/openmpi/lib"
```

### Build Command

```python
SOURCE_NAME = "my_kernel.cu"
BINARY_NAME = "my_kernel"

def get_build_command(work_dir):
    return [
        NVCC, "-o", str(work_dir / BINARY_NAME), str(work_dir / SOURCE_NAME),
        f"-I{NCCL_INCLUDE}", NCCL_STATIC_LIB,
        "-rdc=true",           # Required for device-side NCCL
        "-arch=sm_80",         # A100; use sm_90 for H100, etc.
        f"-L{CUDA_LIB64}", "-lcudart", "-lcudadevrt", "-lpthread",
        f"-I{MPI_INCLUDE}", f"-I{MPI_INCLUDE_OPENMPI}",
        f"-L{MPI_LIB}", "-lmpi",
    ]
```

Critical flags:
- **`-arch=sm_XX`**: Match your GPU architecture (sm_80 for A100, sm_90 for H100, etc.).
- **`-rdc=true`**: Required only for device-initiated NCCL (GIN/LSA). Without it, device-side API calls silently fail. Not needed for host-only NCCL or non-NCCL workloads.
- **Static NCCL**: Use `libnccl_static.a`, not `-lnccl`. Device-initiated APIs are only available with static linking. Not needed for host-only NCCL workloads.

### Building

```python
import subprocess
from pathlib import Path

def build(program_path, work_dir, results_dir):
    # Copy source to work directory
    shutil.copy2(program_path, work_dir / SOURCE_NAME)

    cmd = get_build_command(work_dir)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

    # Save build log
    with open(results_dir / "build.log", "w") as f:
        f.write(result.stdout + "\n" + result.stderr)

    if result.returncode != 0:
        return False, result.stderr
    return True, ""
```

## Run Configuration

### MPI Run Command

```python
MPI_NP = 2  # Number of ranks
RUN_TIMEOUT = 120  # Seconds

def get_run_command(work_dir):
    cmd = [
        "mpirun",
        "--hostfile", str(work_dir / "build" / "hostfile"),
        "-np", str(MPI_NP),
        "--map-by", "node",
    ]

    # Environment variables passed to all ranks
    env_vars = {
        "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
        "CUDA_VISIBLE_DEVICES": "0",
        "NCCL_GIN_ENABLE": "1",
        "NCCL_SOCKET_IFNAME": "enp75s0f1np1",  # Your network interface
        "NCCL_IB_HCA": "mlx5_1",                # Your RDMA device
        "NCCL_IB_GID_INDEX": "3",
    }
    for key, val in env_vars.items():
        cmd.extend(["-x", f"{key}={val}"])

    # MCA parameters for network routing
    cmd.extend([
        "--mca", "btl_tcp_if_include", "enp75s0f1np1",
        "--mca", "oob_tcp_if_include", "enp75s0f1np1",
    ])

    cmd.append(str(work_dir / BINARY_NAME))
    return cmd
```

### Network Configuration

These NCCL environment variables are critical for inter-node GIN:

<table>
  <tr>
    <th>Variable</th>
    <th>Purpose</th>
    <th>How to Find</th>
  </tr>
  <tr>
    <td><code>NCCL_SOCKET_IFNAME</code></td>
    <td>Network interface for NCCL</td>
    <td><code>ip addr</code> — look for the RoCE/IB interface</td>
  </tr>
  <tr>
    <td><code>NCCL_IB_HCA</code></td>
    <td>RDMA HCA device</td>
    <td><code>ibstat</code> — look for the active port</td>
  </tr>
  <tr>
    <td><code>NCCL_IB_GID_INDEX</code></td>
    <td>GID index for RoCE</td>
    <td>Usually 3 for RoCEv2; check <code>ibv_devinfo -v</code></td>
  </tr>
  <tr>
    <td><code>NCCL_GIN_ENABLE</code></td>
    <td>Enable GIN API</td>
    <td>Must be <code>1</code> for device-initiated GIN</td>
  </tr>
</table>

For intra-node (NVLink) experiments, most of these are not needed.

### Running

```python
def run_binary(work_dir, results_dir):
    cmd = get_run_command(work_dir)
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=RUN_TIMEOUT,
        cwd=str(work_dir),
    )

    # Save run log
    with open(results_dir / "run.log", "w") as f:
        f.write(result.stdout + "\n" + result.stderr)

    return result.stdout, result.stderr, result.returncode
```

## Correctness Verification

The most common approach is to check for a known pass string in stdout:

```python
VERIFICATION_PASS_STR = "Verification: PASS"

def check_correctness(stdout):
    return VERIFICATION_PASS_STR in stdout
```

The pass string is configurable — just match it between your kernel and `evaluate.py`. Correctness checking itself is optional: if your workload doesn't have a ground-truth reference, you can skip it and set `correct=True` unconditionally, or use alternative checks (e.g., NaN detection, output range validation, convergence criteria).

## Timing Parsing

For accurate GPU timing, your kernel should use CUDA event timers (`cudaEventRecord` / `cudaEventElapsedTime`) around the critical section rather than CPU-side `clock()` or `gettimeofday()`. CPU timers include launch overhead and synchronization latency, which can skew results by 1-5 ms.

### Simple: Single Time Value

```python
import re

TIME_PATTERN = re.compile(r"^Time:\s*([\d.]+)\s*ms", re.MULTILINE)

def parse_time(stdout):
    match = TIME_PATTERN.search(stdout)
    if match:
        return float(match.group(1))
    return None
```

### Advanced: Token-Weighted Multi-Rank

For workloads with unequal work distribution across ranks:

```python
RANK_HEADER_PATTERN = re.compile(
    r"RESULTS \(Rank (\d+).*?receives (\d+) tokens\).*?\nTime:\s*([\d.]+)\s*ms",
    re.DOTALL,
)

def parse_weighted_time(stdout):
    matches = RANK_HEADER_PATTERN.findall(stdout)
    if not matches:
        return parse_time(stdout)  # Fallback to simple

    total_weighted = 0.0
    total_tokens = 0
    for rank, tokens, time_ms in matches:
        t = int(tokens)
        total_weighted += float(time_ms) * t
        total_tokens += t

    return total_weighted / total_tokens if total_tokens > 0 else None
```

### Multiple Runs

Run the binary multiple times and take the best:

```python
NUM_RUNS = 2

def evaluate(program_path, results_dir):
    # Build
    build_ok, build_err = build(program_path, work_dir, results_dir)
    if not build_ok:
        write_failure(results_dir, f"Build failed: {build_err}")
        return

    # Run multiple times
    best_time = float('inf')
    all_times = []
    for _ in range(NUM_RUNS):
        stdout, stderr, rc = run_binary(work_dir, results_dir)
        if rc != 0 or not check_correctness(stdout):
            write_failure(results_dir, f"Run failed or incorrect")
            return
        time_ms = parse_weighted_time(stdout)
        if time_ms and time_ms < best_time:
            best_time = time_ms
        all_times.append(time_ms)

    write_success(results_dir, best_time, all_times)
```

## Fitness Function

The evolutionary search requires a single scalar `combined_score` (float, **higher is better**) for parent selection, archive ranking, and progress tracking. The simplest case maps a single metric:

```python
def score_from_time_ms(time_ms):
    return 10000.0 / (1.0 + time_ms)
```

You can use any monotone mapping — the search only compares relative ordering.

### Multi-Metric Evaluation

You can track multiple metrics alongside the scalar score. Put sub-metrics in the `public` dict — all key-value pairs in `public` are automatically formatted and injected into the LLM's mutation prompts, so the LLM can reason about multiple performance dimensions (latency, throughput, memory, etc.).

For multi-objective optimization, combine sub-metrics into `combined_score` using a weighted sum, Pareto ranking, or any custom aggregation:

```python
def compute_score(time_ms, memory_mb, accuracy):
    latency_score = 10000.0 / (1.0 + time_ms)
    memory_score = 1000.0 / (1.0 + memory_mb)
    return 0.7 * latency_score + 0.2 * memory_score + 0.1 * accuracy
```

The individual sub-metrics go in `public` for the LLM to see; the aggregated scalar goes in `combined_score` for the evolutionary search.

## Output Files

### metrics.json

```python
def write_success(results_dir, time_ms, all_times):
    metrics = {
        "combined_score": score_from_time_ms(time_ms),
        "public": {
            "time_ms": time_ms,
            "throughput_gb_s": compute_throughput(time_ms),
            "all_run_times_ms": all_times,
            "build_duration_sec": build_duration,
        },
        "private": {
            "peak_memory_mb": get_peak_memory(),
            "nvcc_warnings": count_warnings(build_log),
        },
        "text_feedback": "",  # Optional LLM feedback (see below)
    }
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
```

**Required fields:**
- `combined_score` (float) — the single scalar fitness value used by the evolutionary search for parent selection and ranking

**Optional fields:**
- `public` (dict) — a free-form dictionary of metrics that are shown in the web UI **and** automatically injected into LLM mutation prompts. Every key-value pair is formatted and included, so the LLM can reason about latency, throughput, memory, or any other dimension you track. Put anything here that you want the LLM to see when optimizing.
- `private` (dict) — a free-form dictionary stored in the database but **not** shown to the LLM. Use this for diagnostics, internal bookkeeping, or metrics you don't want influencing mutations.
- `text_feedback` (str) — LLM-generated optimization suggestions (see below). Stored with the candidate and injected into prompts when this candidate's lineage is selected as a parent.

### correct.json

```python
def write_failure(results_dir, error_msg):
    with open(results_dir / "correct.json", "w") as f:
        json.dump({"correct": False, "error": error_msg}, f)
    with open(results_dir / "metrics.json", "w") as f:
        json.dump({"combined_score": 0.0}, f)
```

On success:
```json
{"correct": true}
```

On failure:
```json
{"correct": false, "error": "Description of the failure"}
```

## Optional: LLM Feedback

You can add LLM-generated optimization suggestions to `text_feedback`. This feedback is stored with the candidate and injected into prompts when the candidate's lineage is later selected as a parent.

```python
from cuco.llm import LLMClient

def get_llm_feedback(code, stdout, api_type):
    client = LLMClient(models=["bedrock/us.anthropic.claude-opus-4-6-v1"])
    prompt = f"Analyze this CUDA kernel and suggest the single highest-impact optimization:\n\n{code}\n\nRuntime output:\n{stdout}"
    result = client.query(system_msg="You are a CUDA optimization expert.", user_msg=prompt)
    return result.content
```

## Optional: Hardware Context

Provide hardware information to the LLM for better optimization suggestions:

```python
def get_hardware_context():
    return f"""
## Hardware Context
- GPU: NVIDIA A100-SXM4-80GB
- Architecture: sm_80, 108 SMs, 80 GB HBM2e (2039 GB/s)
- Inter-node: RoCE (RDMA over Converged Ethernet)
- NCCL: 2.28.9 with GIN support
- Ranks: {MPI_NP} (1 per node)
"""
```

This is typically called in `run_evo.py` and included in the task system message.

## Reference Implementation

See `workloads/ds_v3_moe/evaluate.py` for a complete, production-quality evaluation script that includes:
- LLM-based code instrumentation (per-phase timers)
- API detection (GIN / LSA / host NCCL)
- LLM feedback with API-specific context
- Token-weighted multi-rank timing
- Build and run error handling with detailed logging
