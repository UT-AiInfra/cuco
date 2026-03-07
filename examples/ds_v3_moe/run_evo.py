#!/usr/bin/env python3
"""Evolution entry point for CUDA kernels with NCCL device-side communication.

Two-phase evolution:
  Phase 1 (explore) — diverse full rewrites to discover winning architectures
  Phase 2 (exploit) — incremental diffs to refine the best architectures found

The pre-transform pipeline (analyze -> host_to_device -> evolve_markers -> warmup)
runs automatically via the cuco core runner before evolution begins.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from cuco.core import EvolutionRunner, EvolutionConfig
from cuco.database import DatabaseConfig
from cuco.launch import LocalJobConfig

logger = logging.getLogger(__name__)

_EXAMPLE_DIR = Path(__file__).resolve().parent
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))

from run_transform import _build_agent_prompt, _AGENT_SYSTEM_PROMPT
from evaluate import get_hardware_context

try:
    from nccl_api_docs import (
        NCCL_DEVICE_API_REFERENCE,
        NCCL_GIN_API_DOC,
        NCCL_GIN_PURE_EXAMPLE,
        NCCL_LSA_API_DOC,
        NCCL_THREAD_GROUPS_DOC,
        NCCL_TEAMS_DOC,
        NCCL_HEADER_GIN_H,
        NCCL_HEADER_CORE_H,
        NCCL_HEADER_COOP_H,
        NCCL_HEADER_PTR_H,
        NCCL_HEADER_GIN_BARRIER_H,
        NCCL_HEADER_BARRIER_H,
    )
    _NCCL_GIN_DOCS = "\n\n".join([
        NCCL_DEVICE_API_REFERENCE,
        NCCL_GIN_API_DOC,
        NCCL_GIN_PURE_EXAMPLE,
        NCCL_THREAD_GROUPS_DOC,
        NCCL_TEAMS_DOC,
    ])
    _NCCL_GIN_HEADERS = "\n\n".join([
        NCCL_HEADER_GIN_H,
        NCCL_HEADER_CORE_H,
        NCCL_HEADER_COOP_H,
        NCCL_HEADER_PTR_H,
        NCCL_HEADER_GIN_BARRIER_H,
    ])
    _NCCL_LSA_DOCS = "\n\n".join([
        NCCL_DEVICE_API_REFERENCE,
        NCCL_LSA_API_DOC,
        NCCL_THREAD_GROUPS_DOC,
        NCCL_TEAMS_DOC,
    ])
    _NCCL_LSA_HEADERS = "\n\n".join([
        NCCL_HEADER_CORE_H,
        NCCL_HEADER_COOP_H,
        NCCL_HEADER_PTR_H,
        NCCL_HEADER_BARRIER_H,
    ])
except ImportError:
    _NCCL_GIN_DOCS = ""
    _NCCL_GIN_HEADERS = ""
    _NCCL_LSA_DOCS = ""
    _NCCL_LSA_HEADERS = ""

# =========================================================================
# EVOLVE BLOCK CONSTRAINTS (generic)
# =========================================================================
_COMMON_CONSTRAINTS = """
## Evolve Block Structure

The file contains EVOLVE-BLOCK regions delimited by EVOLVE-BLOCK-START and
EVOLVE-BLOCK-END markers. You have FULL AUTONOMY over code inside these blocks:
  - **Kernel definitions block**: All __global__ and __device__ functions.
    You may add, remove, fuse, split, or completely rewrite any kernel here.
  - **Infrastructure + Pipeline block**: Stream creation, memory allocation,
    window registration, DevComm setup, warmup, and the entire timed
    execution pipeline (all kernel launches, synchronization, event management).
    You may restructure the pipeline freely: reorder operations, add/remove
    streams, change launch configs, create events, etc.

## Hard Constraints
- Only modify code between EVOLVE-BLOCK-START and EVOLVE-BLOCK-END markers.
- Preserve any variable names that are referenced in FROZEN sections (outside
  the evolve blocks). Read the frozen sections carefully to identify these.
- Do NOT duplicate top-level function definitions that already exist.
- Communication warmup is already included in the seed. Do NOT remove the
  warmup section — it eliminates 10-50ms of lazy RDMA/NIC initialization
  that would otherwise inflate timing.
- Correctness: the program must print "Verification: PASS".
- The evaluator compiles with nvcc and runs the binary via mpirun.
  Lower "Time: X.XXXX ms" gives higher combined_score.
- Do NOT use external libraries (cuBLAS, cuDNN, cuSPARSE, Thrust, CUB, etc.).
  The build links ONLY cuda, nccl, and mpi. Any #include or call to an
  external library will fail at compile time. All compute must use
  hand-written CUDA kernels.
- Do NOT modify the GEMM or compute internals of existing compute kernels (matmul, activation, etc.). Focus on pipeline and overlap improvements.

## What You CAN Change (if not done already)
- Fuse compute and communication into a single persistent kernel
- Create multiple CUDA streams with priorities
- Reorder kernel launches to overlap compute and communication
- Split communication into separate put/wait kernels
- Add or remove cudaEvent synchronization between streams
- Change ncclDevCommRequirements (ginContextCount, ginSignalCount, etc.)
- Change kernel launch configurations (blocks, threads, cooperative launch)
- Add entirely new kernels
- Merge or split existing kernels
- Use vectorized memory access (int4, float4)
"""

# =========================================================================
# OPTIMIZATION STRATEGIES (generic — applicable to any compute+comm workload)
# =========================================================================
_STRATEGIES = """
## Choosing the Right Fusion Level

There is no single best strategy. The right level of compute-communication
fusion depends entirely on the workload's data-dependency structure. You
must analyze the program and decide. Here is the decision framework:

### Kernel-Level Fusion

Fuse compute and communication into a **single persistent kernel** when:
  - The workload is **iterative**: each step produces a small chunk of
    output that another rank consumes before the next step can proceed.
  - Per-chunk data is **small** relative to the compute on that chunk,
    so kernel launch overhead between steps would dominate.
  - The produce → send → receive → consume cycle is the **inner loop**
    itself, not a one-shot bulk transfer.

Implementation patterns:
  - **Warp specialization**: Reserve 1-2 CTAs for communication, remaining
    CTAs compute. Sync via device-side flags or atomics.
  - **Tile-and-send**: After each output tile, immediately issue the
    transfer; overlap the next tile's compute with the previous transfer.
  - **Cooperative launch**: Grid spanning all SMs, block roles assigned
    at launch (compute blocks vs. communication blocks).

### Stream-Level Overlap

Use **separate CUDA streams** for compute and communication when:
  - The workload has **large, self-contained compute phases** (each big
    enough to saturate the GPU for milliseconds) separated by **bulk
    communication phases**.
  - The compute kernels are already highly optimized and fusing them into
    a communication kernel would **sacrifice their tiling, shared-memory
    reuse, or register-level optimizations**.
  - Communication is a **one-shot bulk transfer**, not a fine-grained
    iterative exchange — so there is no benefit to tile-level interleaving.

Implementation: high-priority stream for communication, default stream
for compute, cudaEventRecord + cudaStreamWaitEvent for cross-stream sync.

### Split Communication Kernels

Break a monolithic communication kernel into separate **PUT (initiate)**
and **WAIT (complete)** launches when:
  - You want to **start sending as early as possible** and defer the wait
    until the received data is actually consumed.
  - There is useful compute work between the send and the receive that
    can fill the overlap window.

Split structure:
  - **PUT kernel**: 1 block, 32 threads — issues all transfers, returns
    immediately.
  - (intervening compute overlaps with the network transfer)
  - **WAIT kernel**: 1 block, 32 threads — polls until transfers land.

### Combining Strategies

These approaches are composable. A single program can use:
  - Stream-level overlap for the outer pipeline
  - A fused kernel for a specific phase that has iterative structure
  - Split put/wait for a bulk transfer that bookends a compute phase

The key question: look at the data-dependency graph. Where do you produce
data that must travel to another rank? How much independent local work
exists between that produce and the point where you consume the reply?

### Contention Awareness

GPU resources are shared between compute and communication:
  - **SMs**: Compute-heavy kernels saturate all SMs. Communication kernels
    that occupy many SMs steal compute throughput.
  - **HBM bandwidth**: Compute and device-side memory copies compete for
    the same bandwidth.
  - **NIC DMA** (GIN): Transfers are offloaded to the NIC's DMA engine,
    which operates independently of SMs and HBM — this is why GIN
    communication can overlap with compute without contention.
  - **NVLink** (LSA): Peer stores share NVLink bandwidth with compute
    peer-memory accesses. LSA barrier sync requires dedicated CTAs.

Design suggestion: communication kernels should use minimal GPU resources
(1 block, 32 threads). The fewer SMs communication occupies, the more
compute throughput is preserved.

### Event-Based Synchronization

Use cudaEventRecord + cudaStreamWaitEvent for cross-stream dependencies.
Avoid cudaStreamSynchronize — it blocks the host and serializes all GPU
work. Events let stream B wait for a specific point in stream A without
blocking anything else.
"""

# =========================================================================
# GIN-SPECIFIC KNOWLEDGE
# =========================================================================
_GIN_KNOWLEDGE = """
## GIN API Rules (correctness)

- GIN windows MUST use ncclMemAlloc. Do not switch to cudaMalloc for
  registered buffers.
- ncclDevCommRequirements: ginSignalCount must be >= the number of distinct
  signal indices used in ncclGin_SignalInc and waitSignal calls.
- There is no ncclGin_NoSignal. Use ncclGin_SignalInc{unsigned_index}.
  Put completion increments the RECEIVER's signal at that index.
- waitSignal(Coop, signal_index, count): waits until THIS RANK's signal
  reaches count. Use ncclCoopThread() for Coop.
- flush(Coop): must match the threads that issued puts (ncclCoopThread
  if a single thread issued them).
- __syncthreads() required between buffer writes and GIN puts.

## Synchronization: Barrier vs Signal-Shadow

ncclGinBarrierSession + bar.sync() is a GLOBAL cross-rank barrier: every rank
must reach the barrier before any rank proceeds. This serializes ranks and
prevents compute-communication overlap when ranks have unequal workloads.

The barrier-free alternative is the **signal-shadow pattern**:
  1. gin.increaseSignalShadow(signalIndex, expectedIncoming)
     — declare how many incoming signals this rank expects (typically nranks-1).
     Call this BEFORE issuing puts (thread 0, then __syncthreads).
  2. Each gin.put(..., ncclGin_SignalInc{signalIndex}, ...) increments the
     RECEIVER's signal counter when the transfer completes.
  3. gin.waitSignalMeetShadow(ncclCoopThread{}, signalIndex)
     — blocks until received signal count meets the shadow value.
  4. gin.flush(ncclCoopThread{}) — ensures all outgoing puts complete.

With this pattern each rank issues puts immediately after its local work
finishes, without waiting for other ranks. Faster ranks proceed without
stalling on slower ones. This is the preferred pattern for overlapping
compute with communication.

## Why GIN Enables Overlap

GIN puts offload data movement to the NIC's DMA engine. The NIC operates
INDEPENDENTLY of GPU SMs. A GIN put kernel uses ~1 SM briefly to initiate
the transfer, then the NIC handles data movement while all other SMs compute.

GIN wait kernels poll the NIC signal register. Use minimal threads
(1 warp = 32 threads in 1 block) to minimize SM occupation while polling.
"""

# =========================================================================
# LSA-SPECIFIC KNOWLEDGE
# =========================================================================
_LSA_KNOWLEDGE = """
## LSA API Rules (correctness)

- LSA (Load/Store Accessible) enables direct peer-to-peer memory reads/writes
  via NVLink or PCIe.
- Use ncclGetLsaPointer(window, byte_offset, peer_rank) for peer memory access.
- ncclLsaBarrierSession provides cross-GPU synchronization:
  sync(Coop, memory_order_relaxed) before reads,
  sync(Coop, memory_order_release) after writes.
- ncclDevCommRequirements: lsaBarrierCount must be >= the number of CTA blocks
  that create barrier sessions.
- devComm.lsaRank and devComm.lsaSize give this rank's position.
- Use ncclTeamLsa(devComm) for barrier team selection.
- LSA works best INTRANODE (direct NVLink stores, ~1-2ms overhead).
  For INTERNODE, GIN is typically faster.
"""

# =========================================================================
# PROFILING FEEDBACK (generic)
# =========================================================================
_PROFILING_INFO = """
## Performance Feedback

After each run, you receive:
- Wall-clock time (Time: X.XXXX ms) — the primary metric to minimize.
- Comparison to the best score achieved so far.
- Architecture classification: your program is classified as **fused**
  (compute+comm in one kernel), **overlap** (multi-stream), or **sequential**.
- Factual code metrics: kernel count and names, stream count, sync pattern
  counts, GIN/LSA configuration values, warmup presence, memory patterns.

Key principles:
- With compute-communication overlap, total time ≈ max(compute_path, comm_path),
  NOT compute + comm. Overlap "hides" whichever path is shorter.
- Focus on reducing the CRITICAL PATH — the longest chain of sequential
  dependencies. Improving a kernel that runs concurrently on another stream
  or another hardware unit does not reduce wall-clock time.
- Measure before optimizing: if communication is already hidden, further
  reducing communication time has no benefit. Optimize the bottleneck.
"""


# =========================================================================
# HARDWARE CONTEXT (dynamically extracted from evaluate.py configuration)
# =========================================================================
_HARDWARE_CONTEXT = get_hardware_context()

# =========================================================================
# PROMPT ASSEMBLY
# =========================================================================

def build_task_msg(api_type: str, gin_ref: str = None, lsa_ref: str = None) -> str:
    """Build the complete task system message for the given API type."""
    if api_type == "gin":
        header = (
            "You are an expert in CUDA and NCCL GIN (GPU Integrated Networking) kernels.\n"
            "Improve the given CUDA program to minimize kernel runtime "
            "(Time in ms printed by the program).\n\n"
            "Analyze the current program structure, identify bottlenecks "
            "(sequential execution, missing overlap, cold-start penalties, etc.), "
            "and apply the optimization strategies described below. "
            "You have full autonomy over the approach — choose whichever strategy "
            "or combination best fits this workload.\n"
        )
        api_knowledge = _GIN_KNOWLEDGE
        api_docs = _NCCL_GIN_DOCS
        api_headers = _NCCL_GIN_HEADERS
    elif api_type == "lsa":
        header = (
            "You are an expert in CUDA and NCCL LSA (Load/Store Accessible) device kernels.\n"
            "Improve the given CUDA program to minimize kernel runtime "
            "(Time in ms printed by the program).\n\n"
            "Analyze the current program structure, identify bottlenecks, "
            "and apply the optimization strategies described below. "
            "You have full autonomy over the approach.\n"
        )
        api_knowledge = _LSA_KNOWLEDGE
        api_docs = _NCCL_LSA_DOCS
        api_headers = _NCCL_LSA_HEADERS
    else:
        raise ValueError(f"Unknown api_type: {api_type!r}. Must be 'gin' or 'lsa'.")

    msg = header + _COMMON_CONSTRAINTS + "\n" + _STRATEGIES + "\n" + api_knowledge + "\n" + _HARDWARE_CONTEXT + "\n" + _PROFILING_INFO

    ref_code = None
    if api_type == "gin" and gin_ref:
        ref_path = Path(gin_ref)
        if ref_path.exists():
            ref_code = ref_path.read_text(encoding="utf-8")
            msg += (
                "\n\n---\n\n## Reference: GIN Communication Pattern\n"
                "Below is a working example demonstrating GIN communication patterns. "
                "Study the PATTERN, not the specific dimensions or workload:\n\n```cuda\n"
                + ref_code + "\n```\n"
            )
    elif api_type == "lsa" and lsa_ref:
        ref_path = Path(lsa_ref)
        if ref_path.exists():
            ref_code = ref_path.read_text(encoding="utf-8")
            msg += (
                "\n\n---\n\n## Reference: LSA Communication Pattern\n"
                "Below is a working LSA kernel using barrier-based synchronization. "
                "Study the PATTERN:\n\n```cuda\n"
                + ref_code + "\n```\n"
            )

    if api_docs:
        label = "GIN" if api_type == "gin" else "LSA"
        msg += f"\n\n---\n\n## NCCL {label} API Reference\n\n" + api_docs
    if api_headers:
        label = "GIN" if api_type == "gin" else "LSA"
        msg += f"\n\n---\n\n## NCCL {label} Header Files\n\n" + api_headers

    return msg


# =========================================================================
# EVOLUTION PHASE CONFIGS
# =========================================================================

_PHASE_CONFIGS = {
    "explore": {
        "patch_type_probs": [0.15, 0.70, 0.15],   # diff, full, cross — heavy full rewrites
        "temperatures": [0.2, 0.5, 0.8],           # diverse but compilable
    },
    "exploit": {
        "patch_type_probs": [0.25, 0.60, 0.15],   # diff, full, cross — incremental diffs
        "temperatures": [0.0, 0.2, 0.5],           # focused refinement
    },
}


def build_configs(
    num_generations: int,
    results_dir: str,
    api_type: str,
    phase: str,
    init_program: str,
    source_name: str,
    gin_ref: str = None,
    lsa_ref: str = None,
):
    job_config = LocalJobConfig(
        eval_program_path="evaluate.py",
    )

    db_config = DatabaseConfig(
        db_path="evolution_db.sqlite",
        num_islands=2,
        archive_size=60,
        elite_selection_ratio=0.3,
        num_archive_inspirations=3,
        num_top_k_inspirations=3,
        migration_interval=8,
        migration_rate=0.15,
        island_elitism=True,
        enforce_island_separation=True,
        parent_selection_strategy="weighted",
        parent_selection_lambda=8.0,
    )

    task_msg = build_task_msg(api_type, gin_ref=gin_ref, lsa_ref=lsa_ref)
    phase_cfg = _PHASE_CONFIGS[phase]

    _nccl_transform_docs = ""
    if api_type == "gin" and (_NCCL_GIN_DOCS or _NCCL_GIN_HEADERS):
        _nccl_transform_docs = _NCCL_GIN_DOCS + "\n\n" + _NCCL_GIN_HEADERS
    elif api_type == "lsa" and (_NCCL_LSA_DOCS or _NCCL_LSA_HEADERS):
        _nccl_transform_docs = _NCCL_LSA_DOCS + "\n\n" + _NCCL_LSA_HEADERS

    ref_code_path = gin_ref or lsa_ref or ""

    evo_config = EvolutionConfig(
        task_sys_msg=task_msg,
        patch_types=["diff", "full", "cross"],
        patch_type_probs=phase_cfg["patch_type_probs"],
        num_generations=num_generations,
        max_parallel_jobs=1,
        max_patch_resamples=3,
        max_patch_attempts=4,
        job_type="local",
        language="cuda",
        llm_models=[
            "bedrock/us.anthropic.claude-opus-4-6-v1",
        ],
        llm_kwargs=dict(
            temperatures=phase_cfg["temperatures"],
            max_tokens=32768,
        ),
        meta_rec_interval=8,
        meta_llm_models=["bedrock/us.anthropic.claude-opus-4-6-v1"],
        meta_llm_kwargs=dict(
            temperatures=[0.0],
            max_tokens=8192,
        ),
        meta_max_recommendations=5,
        init_program_path=init_program,
        results_dir=results_dir,
        max_novelty_attempts=5,
        code_embed_sim_threshold=0.995,
        use_text_feedback=True,
        embedding_model="bedrock-amazon.titan-embed-text-v1",
        pre_transform_enabled=True,
        pre_transform_pipeline_steps=["analyze", "host_to_device", "evolve_markers", "warmup"],
        pre_transform_agent=False,
        pre_transform_agent_model="sonnet",
        pre_transform_agent_prompt_builder=_build_agent_prompt,
        pre_transform_agent_system_prompt=_AGENT_SYSTEM_PROMPT,
        pre_transform_nccl_api_docs=_nccl_transform_docs,
        pre_transform_rewrite_model="bedrock/us.anthropic.claude-opus-4-6-v1",
        pre_transform_warmup_model="bedrock/us.anthropic.claude-opus-4-6-v1",
        pre_transform_reference_code_path=ref_code_path,
    )
    return evo_config, job_config, db_config


def main(
    num_generations: int,
    results_dir: str,
    api_type: str,
    init_program: str,
    source_name: str,
    explore_fraction: float,
    gin_ref: str = None,
    lsa_ref: str = None,
):
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    explore_gens = max(1, int(num_generations * explore_fraction))

    # --- Phase 1: Explore ---
    print(f"\n{'='*60}")
    print(f"PHASE 1 — EXPLORE (generations 0..{explore_gens - 1})")
    print(f"  70% full rewrites, high temperature, architectural diversity")
    print(f"{'='*60}\n")

    evo_config, job_config, db_config = build_configs(
        num_generations=explore_gens,
        results_dir=results_dir,
        api_type=api_type,
        phase="explore",
        init_program=init_program,
        source_name=source_name,
        gin_ref=gin_ref,
        lsa_ref=lsa_ref,
    )
    EvolutionRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        verbose=True,
    ).run()

    # --- Phase 2: Exploit ---
    print(f"\n{'='*60}")
    print(f"PHASE 2 — EXPLOIT (generations {explore_gens}..{num_generations - 1})")
    print(f"  60% diffs, lower temperature, refining best architectures")
    print(f"{'='*60}\n")

    evo_config, job_config, db_config = build_configs(
        num_generations=num_generations,
        results_dir=results_dir,
        api_type=api_type,
        phase="exploit",
        init_program=init_program,
        source_name=source_name,
        gin_ref=gin_ref,
        lsa_ref=lsa_ref,
    )
    EvolutionRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        verbose=True,
    ).run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Cufuse evolution to optimize a CUDA kernel with device-side communication."
    )
    parser.add_argument("--num_generations", type=int, default=60)
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results_ds_v3_moe",
        help="Directory to store evolution artifacts.",
    )
    parser.add_argument(
        "--api",
        type=str,
        choices=["gin", "lsa"],
        default="gin",
        help="Communication API: 'gin' (GPU Initiated Networking) or 'lsa' (Load/Store Accessible).",
    )
    parser.add_argument(
        "--explore_fraction",
        type=float,
        default=0.4,
        help="Fraction of generations for Phase 1 (explore). "
             "Remaining generations are Phase 2 (exploit). Default: 0.4 (40%%).",
    )
    parser.add_argument(
        "--init_program",
        type=str,
        default="ds_v3_moe.cu",
        help="Path to the initial seed program (.cu file).",
    )
    parser.add_argument(
        "--source_name",
        type=str,
        default="ds_v3_moe.cu",
        help="Filename the evaluator expects for compilation.",
    )
    parser.add_argument(
        "--gin_ref",
        type=str,
        default=None,
        help="Path to a GIN reference example (.cu file). Included in the "
             "evolution prompt as a pattern to study. Omit to skip.",
    )
    parser.add_argument(
        "--lsa_ref",
        type=str,
        default=None,
        help="Path to an LSA reference example (.cu file). Included in the "
             "evolution prompt as a pattern to study. Omit to skip.",
    )
    args = parser.parse_args()
    main(
        args.num_generations,
        args.results_dir,
        api_type=args.api,
        init_program=args.init_program,
        source_name=args.source_name,
        explore_fraction=args.explore_fraction,
        gin_ref=args.gin_ref,
        lsa_ref=args.lsa_ref,
    )
