"""Host-to-device transformation pipeline (GIN / LSA).

Pipeline: Analyze → LLM Rewrite → Build → Run/Verify → LLM Judge → Loop.

Takes a CUDA source file with host-side NCCL collective calls and iteratively
transforms it into a device-side equivalent (GIN or LSA, controlled by
``TransformConfig.api_type``) through an LLM-driven build/verify feedback loop.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .cuda_analyzer import CUDAAnalyzer, AnalysisReport

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config & result types
# ---------------------------------------------------------------------------

@dataclass
class TransformConfig:
    """Configuration for the host-to-device transformation pipeline."""

    # LLM settings (uses cuco's query interface)
    rewrite_model: str = "bedrock/us.anthropic.claude-sonnet-4-6"
    judge_model: str = ""  # Empty = use same model as rewriter (single LLM)
    rewrite_max_tokens: int = 32768
    judge_max_tokens: int = 2048
    rewrite_temperature: float = 0.0
    judge_temperature: float = 0.0

    # Build settings
    nvcc_path: str = "/usr/local/cuda-13.1/bin/nvcc"
    nccl_include: str = "/usr/local/nccl_2.28.9-1+cuda13.0_x86_64/include"
    nccl_static_lib: str = "/usr/local/nccl_2.28.9-1+cuda13.0_x86_64/lib/libnccl_static.a"
    cuda_lib64: str = "/usr/local/cuda-13.1/lib64"
    mpi_include: str = "/usr/lib/x86_64-linux-gnu/openmpi/include"
    mpi_include_openmpi: str = "/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi"
    mpi_lib: str = "/usr/lib/x86_64-linux-gnu/openmpi/lib"
    binary_name: str = "cuda_program"

    # Run settings
    num_mpi_ranks: int = 4
    run_timeout: int = 120
    cuda_visible_devices: str = "0,1,2,3"

    # Inter-node mpirun settings (leave empty for local-only runs)
    hostfile: str = ""
    mpirun_extra_args: tuple = ()  # e.g. ("--map-by", "node")
    run_env_vars: dict = field(default_factory=dict)  # extra env vars passed via -x

    # Loop settings
    max_iterations: int = 5
    verification_pass_str: str = "Verification: PASS"

    # Target communication API: "gin" or "lsa"
    api_type: str = "gin"

    # Two-stage transformation
    # Stage A: Add device-side infrastructure (ncclMemAlloc, windows, devComm) while keeping host-side collectives
    # Stage B: Replace host-side NCCL collectives with device-side kernel(s)
    two_stage: bool = True
    stage_a_max_iterations: int = 5  # max iterations for infrastructure stage
    stage_b_max_iterations: int = 10  # max iterations for kernel replacement stage

    # Reference code (working device-side example to show the LLM)
    reference_code: str = ""

    # NCCL API docs (injected from nccl_api_docs.py by the caller)
    nccl_api_docs: str = ""


@dataclass
class IterationResult:
    """Result of one rewrite → build → run → judge iteration."""
    iteration: int
    code: str
    build_success: bool
    build_errors: str = ""
    run_success: bool = False
    run_output: str = ""
    verification_passed: bool = False
    time_ms: Optional[float] = None
    has_host_communication: bool = True
    judge_feedback: str = ""
    rewrite_prompt: str = ""      # The full user prompt sent to the rewriter LLM
    system_prompt_name: str = ""  # Which system prompt was used (e.g. "REWRITE", "STAGE_A", "STAGE_B")
    system_prompt: str = ""       # Full system prompt text (saved to iter_N_system_prompt.txt when set)
    duration_sec: float = 0.0


@dataclass
class TransformResult:
    """Final result of the transformation pipeline."""
    success: bool
    final_code: str
    source_filename: str = ""  # original source filename (e.g. "split_k_mat_mul.cu")
    iterations: List[IterationResult] = field(default_factory=list)
    total_duration_sec: float = 0.0
    total_cost: float = 0.0
    error: str = ""

    @property
    def device_filename(self) -> str:
        """Derive the output filename: <stem>_device.cu"""
        if not self.source_filename:
            return "transformed.cu"
        stem = self.source_filename
        if stem.endswith(".cu"):
            stem = stem[:-3]
        return f"{stem}_device.cu"


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_API_LABELS = {
    "gin": ("GIN", "GPU-Initiated Networking (GIN)"),
    "lsa": ("LSA", "Load/Store Agent (LSA)"),
}


def _rewrite_system_prompt(api_type: str = "gin") -> str:
    _, long_name = _API_LABELS.get(api_type, _API_LABELS["gin"])
    return (
        f"You are an expert CUDA and NCCL programmer specializing in {long_name}. "
        f"You transform CUDA programs from host-side NCCL collectives "
        f"to device-side {api_type.upper()} communication.\n\n"
        "Output the COMPLETE transformed CUDA file. Do not omit any sections. "
        "Return the full file contents between ```cuda and ``` markers."
    )



_STAGE_A_COMMON_HEADER = """\
## Task (Stage A — infrastructure only)

Add {api_upper} infrastructure to this program. Do NOT replace any host-side NCCL \
collective calls yet (ncclAllReduce, ncclAllGather, etc.).

You must:
1. Add `#include <nccl_device.h>`
2. Replace `cudaMalloc` for communicated buffers with `ncclMemAlloc` \
({api_upper} requires registered memory)
3. Add remote landing buffer(s) with `ncclMemAlloc` for peer data
4. Add `ncclCommWindowRegister()` for all communicated buffers \
(`NCCL_WIN_COLL_SYMMETRIC`)
5. Add `ncclDevCommCreate()` with correct requirements \
({reqs_hint})
6. Update cleanup: `ncclCommWindowDeregister`, `ncclMemFree`, \
`ncclDevCommDestroy`
7. KEEP all host-side NCCL collective calls exactly as they are — \
do NOT replace them yet.

**IMPORTANT**: Place ALL {api_upper}-specific configuration variables \
(e.g. CTA counts, barrier counts, signal counts) AFTER the `ncclMemAlloc` \
calls, alongside the window registration and `ncclDevCommCreate` section. \
Do NOT place them before the non-communicated `cudaMalloc` calls."""

_STAGE_A_GIN_PATTERN = """
### Infrastructure pattern

```cpp
#include <nccl_device.h>

// GIN-compatible buffers (replace cudaMalloc):
NCCL_CHECK(ncclMemAlloc((void**)&buf, size));
NCCL_CHECK(ncclMemAlloc((void**)&buf_remote, size));  // landing buffer

// Register windows:
ncclWindow_t send_win, recv_win;
NCCL_CHECK(ncclCommWindowRegister(host_comm, buf, size, \
&send_win, NCCL_WIN_COLL_SYMMETRIC));
NCCL_CHECK(ncclCommWindowRegister(host_comm, buf_remote, size, \
&recv_win, NCCL_WIN_COLL_SYMMETRIC));

// Device communicator:
ncclDevComm devComm;
ncclDevCommRequirements reqs;
memset(&reqs, 0, sizeof(reqs));
reqs.ginContextCount = 1;
reqs.ginSignalCount = <appropriate_count>;
NCCL_CHECK(ncclDevCommCreate(host_comm, &reqs, &devComm));
(void)cudaGetLastError();  // MUST clear stale CUDA error
```"""

_STAGE_A_LSA_PATTERN = """
### Infrastructure pattern

```cpp
#include <nccl_device.h>

// LSA-compatible buffers (replace cudaMalloc):
NCCL_CHECK(ncclMemAlloc((void**)&buf, size));
NCCL_CHECK(ncclMemAlloc((void**)&buf_remote, size));  // landing buffer

// Register windows:
ncclWindow_t send_win, recv_win;
NCCL_CHECK(ncclCommWindowRegister(host_comm, buf, size, \
&send_win, NCCL_WIN_COLL_SYMMETRIC));
NCCL_CHECK(ncclCommWindowRegister(host_comm, buf_remote, size, \
&recv_win, NCCL_WIN_COLL_SYMMETRIC));

// LSA config — place AFTER ncclMemAlloc, alongside window registration:
int nCTAs = 4;
int totalBarriers = 2 * nCTAs;  // e.g. dispatch + combine

// Device communicator:
ncclDevComm devComm;
ncclDevCommRequirements reqs;
memset(&reqs, 0, sizeof(reqs));
reqs.lsaBarrierCount = totalBarriers;
NCCL_CHECK(ncclDevCommCreate(host_comm, &reqs, &devComm));
(void)cudaGetLastError();  // MUST clear stale CUDA error
```"""

_STAGE_A_FOOTER = """
**CRITICAL**: You MUST add `(void)cudaGetLastError();` immediately after \
`ncclDevCommCreate()`. It internally triggers CUDA operations that leave a \
stale `cudaErrorNotPermitted` in the error register. If you don't clear it, \
the next `cudaGetLastError()` call will crash with "operation not permitted".

The program must still compile, run, and pass verification with the \
original collectives.
Return the COMPLETE file between ```cuda and ``` markers."""


def _stage_a_task_prompt(api_type: str = "gin") -> str:
    api_upper = api_type.upper()
    if api_type == "lsa":
        reqs_hint = "lsaBarrierCount"
        pattern = _STAGE_A_LSA_PATTERN
    else:
        reqs_hint = "ginContextCount, ginSignalCount, etc."
        pattern = _STAGE_A_GIN_PATTERN
    header = _STAGE_A_COMMON_HEADER.format(
        api_upper=api_upper, reqs_hint=reqs_hint,
    )
    return header + pattern + _STAGE_A_FOOTER



_COLLECTIVE_PATTERNS = """\
### Data-flow patterns by collective type
- **AllReduce**: each rank sends its data to peers, receives peer data, \
accumulates locally
- **AllGather**: each rank sends its chunk, receives all other chunks
- **Broadcast**: root sends to all, others receive
- **ReduceScatter**: all-to-all exchange + local reduction
- **AlltoAll**: each rank sends different data to each peer"""

_GIN_KERNEL_GUIDANCE_TEXT = """\
### For each collective, write a GIN kernel that:
1. Uses `gin.put()` to send local data to the appropriate peer's remote window
2. Uses `gin.flush()` to ensure source buffers are safe to reuse \
(flush does NOT guarantee data has settled at the remote — use barriers/signals for that)
3. Uses `gin.waitSignal()` to wait for expected incoming data
4. Performs any needed local computation (accumulation, copy, etc.)

### Key rules
- Pass `ncclDevComm` **by value** to kernels — do NOT `cudaMalloc`/`cudaMemcpy` it
- Use literal integers for `<<<grid, block>>>` — no NCCL_DEVICE_CTA_COUNT macros
- `__syncthreads()` between: writes→puts, flush→wait, wait→reads
- `ncclCoopThread()` for single-thread put/flush; `ncclCoopCta()` for \
block-wide wait (CAUTION: ncclCoopCta can deadlock if not all CTA threads reach the wait)
- MUST have `(void)cudaGetLastError();` immediately after `ncclDevCommCreate()`

Return the COMPLETE transformed file between ```cuda and ``` markers."""

_LSA_KERNEL_GUIDANCE_TEXT = """\
### For each collective, write an LSA kernel that:
1. Uses `ncclGetLsaPointer(window, byte_offset, peer_rank)` to get a device \
pointer to a peer's registered window for direct load/store
2. Uses `ncclLsaBarrierSession` with `ncclCoopCta()` for cross-GPU synchronization — \
sync with `memory_order_relaxed` before reads and `memory_order_release` after writes
3. Reads directly from peer send buffers and writes into local recv buffers

### Key rules
- Pass `ncclDevComm` **by value** to kernels — do NOT `cudaMalloc`/`cudaMemcpy` it
- Use literal integers for `<<<grid, block>>>` — no NCCL_DEVICE_CTA_COUNT macros
- Use `devComm.lsaRank` and `devComm.lsaSize` for rank/size info
- Use `ncclTeamLsa(devComm)` for barrier team selection
- `ncclDevCommRequirements.lsaBarrierCount` must be >= the number of CTAs \
that create barrier sessions
- MUST have `(void)cudaGetLastError();` immediately after `ncclDevCommCreate()`

Return the COMPLETE transformed file between ```cuda and ``` markers."""


def _kernel_guidance(api_type: str = "gin") -> str:
    text = _LSA_KERNEL_GUIDANCE_TEXT if api_type == "lsa" else _GIN_KERNEL_GUIDANCE_TEXT
    return _COLLECTIVE_PATTERNS + "\n\n" + text



def _stage_b_task_prompt(api_type: str = "gin") -> str:
    api_upper = api_type.upper()
    guidance = _kernel_guidance(api_type)
    return (
        f"## Task (Stage B — replace collectives with {api_upper})\n\n"
        f"The code already has {api_upper} infrastructure (ncclMemAlloc, window registration, "
        f"ncclDevCommCreate). Replace the remaining host-side NCCL collective calls "
        f"with device-side {api_upper} kernel(s).\n\n"
        f"{guidance}"
    )



def _single_stage_task_prompt(api_type: str = "gin") -> str:
    api_upper = api_type.upper()
    guidance = _kernel_guidance(api_type)
    return (
        "## Task\n\n"
        f"Transform this code to replace ALL host-side NCCL collective calls "
        f"(identified above) with device-side {api_upper} equivalents.\n\n"
        f"{guidance}"
    )



JUDGE_SYSTEM_PROMPT_TEMPLATE = """You are a diagnostic reviewer for CUDA code transformations.

You are given:
1. The original CUDA source (with host-side NCCL collectives)
2. The transformed CUDA source (attempting device-side communication)
3. The build/run outcome (compiler errors, runtime errors, or verification output)

Your ONLY job is to DIAGNOSE what went wrong. The same LLM that wrote the code will read your diagnosis 
and decide how to fix it.

Rules:
- Describe the failure: quote the error messages and identify which line/region caused it.
- If it is a build error, state which symbol or syntax is wrong and why.
- If it is a runtime error (e.g. illegal memory access, segfault), identify which kernel or
  host-side call is most likely responsible based on the error output.
- If verification failed, describe the mismatch (expected vs actual).
- Do NOT prescribe specific API call patterns, synchronization placement, or signal values.
  The rewriter has the same API documentation you do and will determine the correct pattern.
- Do NOT invent API functions that don't exist in the reference documentation below.
- Keep your response under 200 words. Focus on one or two most critical issue.

Reference documentation (for verifying API usage):

{nccl_api_docs}
"""


# ---------------------------------------------------------------------------
# EVOLVE-BLOCK marker insertion
# ---------------------------------------------------------------------------

_GIN_DEVICE_TOKENS = re.compile(
    r"\bncclGin\b|\bgin\.\w+|\bncclDevComm\b|\bncclTeamWorld\b"
    r"|\bncclGin_SignalInc\b|\bncclGin_None\b|\bncclCoopThread\b|\bncclCoopCta\b"
)

_GLOBAL_FUNC_RE = re.compile(r"^\s*__global__\s+void\s+\w+\s*\(")
_DEVICE_FUNC_RE = re.compile(r"^\s*__device__\s+")

_TIMING_LINE_RE = re.compile(
    r"cudaEventCreate|cudaEventRecord|cudaEvent_t\s"
    r"|//.*[Tt]iming|//.*DO NOT MODIFY"
)

_CHUNK_LOOP_RE = re.compile(
    r"for\s*\(\s*int\s+\w+\s*=\s*0\s*;\s*\w+\s*<\s*(NUM_CHUNKS|\d+)"
)

_SETUP_START_RE = re.compile(
    r"ncclWindow_t\b|ncclCommWindowRegister\b"
)


def _find_matching_brace(lines: List[str], open_line: int) -> int:
    """Return the line index of the closing '}' that matches the first '{' at or after open_line."""
    depth = 0
    started = False
    for i in range(open_line, len(lines)):
        for ch in lines[i]:
            if ch == '{':
                depth += 1
                started = True
            elif ch == '}':
                depth -= 1
                if started and depth == 0:
                    return i
    return len(lines) - 1


def insert_evolve_markers(
    code: str,
    *,
    llm_model: Optional[str] = None,
) -> str:
    """Insert ``// EVOLVE-BLOCK-START`` / ``// EVOLVE-BLOCK-END`` markers.

    If *llm_model* is provided, an LLM call decides where to place the markers
    (recommended — handles arbitrary code structures).  Otherwise falls back to
    the deterministic regex-based heuristic.

    Returns the code with markers inserted, or unchanged if markers are already
    present.
    """
    if "EVOLVE-BLOCK-START" in code:
        return code

    if llm_model:
        result = _insert_markers_via_llm(code, llm_model)
        if result and "EVOLVE-BLOCK-START" in result:
            if not _validate_markers_only(code, result):
                logger.warning(
                    "insert_evolve_markers: LLM modified code beyond inserting "
                    "markers — rejecting"
                )
            else:
                violations = _validate_marker_placement(result)
                if violations:
                    for v in violations:
                        logger.warning(f"insert_evolve_markers: {v}")
                    logger.warning(
                        "insert_evolve_markers: LLM placed markers around "
                        "frozen code — rejecting"
                    )
                else:
                    n = result.count("EVOLVE-BLOCK-START")
                    logger.info(
                        f"insert_evolve_markers (LLM): inserted {n} "
                        f"EVOLVE-BLOCK region(s)"
                    )
                    return result
        else:
            logger.warning(
                "insert_evolve_markers: LLM failed to produce valid markers"
            )
        logger.warning("insert_evolve_markers: falling back to regex heuristic")

    regex_result = _insert_markers_via_regex(code)
    violations = _validate_marker_placement(regex_result)
    if violations:
        for v in violations:
            logger.warning(f"insert_evolve_markers (regex): {v}")
        logger.warning(
            "insert_evolve_markers (regex): returning code without markers "
            "to avoid evolving frozen regions"
        )
        return code
    return regex_result


def _validate_markers_only(original: str, marked: str) -> bool:
    """Verify that *marked* is identical to *original* except for added marker lines.

    Strips all ``// EVOLVE-BLOCK-START`` and ``// EVOLVE-BLOCK-END`` lines
    from *marked* and checks that the remaining lines match *original*.
    """
    marker_re = re.compile(r"^\s*//\s*EVOLVE-BLOCK-(START|END)\s*$")
    stripped_lines = [
        line for line in marked.split("\n")
        if not marker_re.match(line)
    ]
    original_lines = original.split("\n")

    if len(stripped_lines) != len(original_lines):
        logger.debug(
            f"Marker validation: line count mismatch "
            f"(original={len(original_lines)}, marked-stripped={len(stripped_lines)})"
        )
        return False

    for i, (orig, stripped) in enumerate(zip(original_lines, stripped_lines)):
        if orig != stripped:
            logger.debug(
                f"Marker validation: line {i+1} differs:\n"
                f"  original: {orig!r}\n"
                f"  marked:   {stripped!r}"
            )
            return False

    return True


# Patterns that must NEVER appear inside an EVOLVE-BLOCK.
_FROZEN_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bint\s+main\s*\("), "main() function"),
    (re.compile(r"\bMPI_Init\b"), "MPI_Init"),
    (re.compile(r"\bMPI_Finalize\b"), "MPI_Finalize"),
    (re.compile(r"\bMPI_Comm_rank\b"), "MPI_Comm_rank"),
    (re.compile(r"\bMPI_Comm_size\b"), "MPI_Comm_size"),
    (re.compile(r"\bMPI_Bcast\b"), "MPI_Bcast"),
    (re.compile(r"\bMPI_Barrier\b"), "MPI_Barrier"),
    (re.compile(r"\bncclCommInitRank\b"), "ncclCommInitRank"),
    (re.compile(r"\bncclGetUniqueId\b"), "ncclGetUniqueId"),
    (re.compile(r"\bncclCommDestroy\b(?!.*DevComm)"), "ncclCommDestroy (host)"),
    (re.compile(r"\bcudaEventSynchronize\b"), "cudaEventSynchronize (timing)"),
    (re.compile(r"\bcudaEventElapsedTime\b"), "cudaEventElapsedTime (timing)"),
    (re.compile(r"\bcudaEventDestroy\b"), "cudaEventDestroy (timing)"),
    (re.compile(r"Verification:\s*(PASS|FAIL)"), "verification logic"),
    (re.compile(r"\bcudaMemcpy\s*\([^)]*cudaMemcpyDeviceToHost"), "D2H verification copy"),
    (re.compile(r"#\s*include\b"), "#include directive"),
    (re.compile(r"#\s*define\b"), "#define macro"),
]


def _validate_marker_placement(marked_code: str) -> List[str]:
    """Check that no EVOLVE-BLOCK contains code that must remain frozen.

    Returns a list of violation descriptions (empty = valid).
    """
    violations: List[str] = []
    lines = marked_code.split("\n")
    in_block = False
    block_num = 0
    block_lines: List[str] = []

    for line in lines:
        stripped = line.strip()
        if re.match(r"//\s*EVOLVE-BLOCK-START", stripped):
            in_block = True
            block_num += 1
            block_lines = []
            continue
        if re.match(r"//\s*EVOLVE-BLOCK-END", stripped):
            in_block = False
            continue
        if in_block:
            block_lines.append(line)
            for pattern, desc in _FROZEN_PATTERNS:
                if pattern.search(line):
                    violations.append(
                        f"Block {block_num} contains frozen code: "
                        f"{desc} (line: {stripped[:80]})"
                    )

    # Structural checks
    start_count = marked_code.count("EVOLVE-BLOCK-START")
    end_count = marked_code.count("EVOLVE-BLOCK-END")
    if start_count != end_count:
        violations.append(
            f"Mismatched markers: {start_count} START vs {end_count} END"
        )

    if start_count == 0:
        violations.append("No EVOLVE-BLOCK regions found")
    elif start_count != 2:
        violations.append(
            f"Expected exactly 2 EVOLVE-BLOCK regions, found {start_count}"
        )

    return violations


# ---------------------------------------------------------------------------
# LLM-based marker insertion
# ---------------------------------------------------------------------------

_MARKER_USER_PROMPT = """\
You are an expert CUDA/NCCL programmer. \
Below is a CUDA source file that uses NCCL device-side communication \
(GIN or LSA). I need you to insert exactly TWO pairs of \
``// EVOLVE-BLOCK-START`` and ``// EVOLVE-BLOCK-END`` markers to delineate \
the two regions that an evolutionary optimizer is allowed to modify.

## Rules for placing markers — EXACTLY TWO EVOLVE-BLOCKs

### Block 1 — Kernel / device function DEFINITIONS
Place ``// EVOLVE-BLOCK-START`` BEFORE the first ``__device__`` or \
``__global__`` kernel function definition. Place ``// EVOLVE-BLOCK-END`` \
BEFORE the host orchestration function (``void run_moe_alltoall`` or \
``void run_gpu``). This block contains all kernel definitions and \
device helper functions that the optimizer may rewrite, fuse, or split.

### Block 2 — Communication infrastructure + kernel LAUNCH orchestration
Place ``// EVOLVE-BLOCK-START`` BEFORE the first ``ncclMemAlloc`` call \
(communication buffer allocation) in the host orchestration function. \
Place ``// EVOLVE-BLOCK-END`` AFTER the last kernel launch or \
communication call in the timed region (before \
``cudaEventRecord(stop, ...)``). This block covers communication buffer \
allocation (``ncclMemAlloc``), window registration \
(``ncclCommWindowRegister``), device communicator creation \
(``ncclDevCommCreate`` / ``ncclDevCommRequirements``), launch \
configurations, stream usage, and synchronization ordering.

### Frozen zone between blocks
The code between Block 1's END and Block 2's START must NOT be modified. \
It contains the host orchestration function signature, non-communicated \
buffer allocation (``cudaMalloc``), and config printing.

CRITICAL: You MUST insert exactly TWO ``// EVOLVE-BLOCK-START`` lines and \
exactly TWO ``// EVOLVE-BLOCK-END`` lines. The pairs must not overlap.

## What must NEVER be inside any EVOLVE-BLOCK

The following categories of code must remain OUTSIDE the markers. \
If an EVOLVE-BLOCK would need to include any of these, shrink the \
block boundary so these lines are excluded.

- **Preprocessor directives**: ``#include``, ``#define``, ``#pragma``
- **Error-check macros**: ``CUDA_CHECK``, ``NCCL_CHECK`` macro definitions
- **main()**: The ``int main(...)`` function and everything it contains
- **MPI calls**: ``MPI_Init``, ``MPI_Finalize``, ``MPI_Comm_rank``, \
``MPI_Comm_size``, ``MPI_Bcast``, ``MPI_Barrier``, ``MPI_Reduce``
- **Host-side NCCL init/destroy**: ``ncclGetUniqueId``, ``ncclCommInitRank``, \
``ncclCommDestroy``
- **Non-communicated buffer allocation**: ``cudaMalloc``, ``cudaMemset``
- **Timing measurement**: ``cudaEventSynchronize``, ``cudaEventElapsedTime``, \
``cudaEventDestroy``
- **Verification / correctness checking**: ``cudaMemcpy(...DeviceToHost)`` \
for result checking, ``printf`` with PASS/FAIL, tolerance checks
- **Cleanup**: ``cudaFree``, ``ncclCommDestroy``, ``MPI_Finalize``, \
``ncclCommWindowDeregister``, ``ncclDevCommDestroy``, ``ncclMemFree``
- **Config printing**: ``printf`` showing dimensions, MB sizes, config info

## Output format

Return the COMPLETE source file with markers inserted. Do NOT omit any \
code — return every line of the original. Only add the four marker lines. \
Do NOT modify, delete, or reorder any existing code.

## Source code

```cuda
{code}
```"""


def _insert_markers_via_llm(code: str, model: str) -> str:
    """Call an LLM to insert EVOLVE-BLOCK markers into *code*."""
    user_msg = _MARKER_USER_PROMPT.format(code=code)

    if model.startswith("claude-cli/"):
        return _markers_llm_claude_cli(model, user_msg)
    return _markers_llm_anthropic(model, user_msg)


def _markers_llm_claude_cli(model: str, user_msg: str) -> str:
    """Insert markers via Claude Code CLI."""
    import subprocess as _sp

    model_map = {
        "claude-cli/opus": "opus",
        "claude-cli/sonnet": "sonnet",
        "claude-cli/haiku": "haiku",
    }
    model_alias = model_map.get(model, "opus")

    cmd = [
        "claude", "-p",
        "--model", model_alias,
        "--output-format", "text",
        "--dangerously-skip-permissions",
        "--no-session-persistence",
        "--tools", "",
    ]

    env = os.environ.copy()
    for p in ["/usr/bin", "/usr/local/bin", os.path.expanduser("~/.local/bin")]:
        if p not in env.get("PATH", ""):
            env["PATH"] = p + ":" + env.get("PATH", "")

    try:
        result = _sp.run(
            cmd, input=user_msg, capture_output=True, text=True,
            timeout=180, env=env,
        )
        raw = (result.stdout or "").strip()
    except Exception as exc:
        logger.warning(f"Claude CLI marker insertion failed: {exc}")
        return ""

    return _extract_code_from_response(raw)


def _markers_llm_anthropic(model: str, user_msg: str) -> str:
    """Insert markers via Anthropic/Bedrock API."""
    try:
        import anthropic

        if model.startswith("bedrock/"):
            actual_model = model.split("/", 1)[1]
            client = anthropic.AnthropicBedrock(
                aws_access_key=os.environ.get("AWS_ACCESS_KEY_ID"),
                aws_secret_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
                aws_region=os.environ.get("AWS_REGION_NAME"),
            )
        else:
            actual_model = model
            client = anthropic.Anthropic()

        with client.messages.stream(
            model=actual_model,
            messages=[{"role": "user", "content": user_msg}],
            max_tokens=16384,
            temperature=0.0,
        ) as stream:
            response = stream.get_final_message()

        if response.content and len(response.content) > 0:
            raw = response.content[0].text.strip()
            return _extract_code_from_response(raw)
    except Exception as exc:
        logger.warning(f"Anthropic marker insertion failed: {exc}")
    return ""


def _extract_code_from_response(raw: str) -> str:
    """Pull the CUDA code out of an LLM response (may be in a code fence)."""
    if not raw:
        return ""
    # Try to extract from ```cuda ... ``` or ``` ... ``` fences
    fence_pattern = re.compile(
        r"```(?:cuda|cpp|c\+\+)?\s*\n(.*?)```", re.DOTALL
    )
    m = fence_pattern.search(raw)
    if m:
        return m.group(1).strip()
    # If no fence, check if the raw text itself looks like code
    if "#include" in raw and "EVOLVE-BLOCK" in raw:
        return raw
    return ""


# ---------------------------------------------------------------------------
# Regex-based marker insertion (fallback)
# ---------------------------------------------------------------------------

def _insert_markers_via_regex(code: str) -> str:
    """Deterministic regex-based EVOLVE-BLOCK insertion — two blocks.

    Places TWO ``// EVOLVE-BLOCK-START`` / ``// EVOLVE-BLOCK-END`` pairs:
      Block 1 (definitions): from the first ``__device__``/``__global__``
          kernel through the line before the host orchestration function.
      Block 2 (comm infra + launches): from the first ``ncclMemAlloc``
          (communication infrastructure) through the last kernel launch in
          the timed region.

    The frozen zone between blocks contains host setup: non-communicated
    buffer allocation (cudaMalloc), weight/data init, and timing start.
    """
    lines = code.split("\n")

    # --- Find the first kernel function (global or device) ---
    first_kernel_line: Optional[int] = None
    i = 0
    while i < len(lines):
        if _GLOBAL_FUNC_RE.search(lines[i]) or (
            _DEVICE_FUNC_RE.search(lines[i]) and "{" in lines[i]
        ):
            first_kernel_line = i
            break
        i += 1

    if first_kernel_line is None:
        logger.warning("insert_evolve_markers: no kernel functions found")
        return code

    # --- Find the host orchestration function ---
    run_func_re = re.compile(r"\bvoid\s+(?:run_gpu|run_moe_alltoall)\s*\(")
    run_func_line: Optional[int] = None
    run_func_end: Optional[int] = None
    for idx, line in enumerate(lines):
        if run_func_re.search(line):
            run_func_line = idx
            run_func_end = _find_matching_brace(lines, idx)
            break

    if run_func_line is None or run_func_end is None:
        logger.warning("insert_evolve_markers: no run_gpu/run_moe_alltoall found")
        return code

    # --- Block 1 end: last non-blank line before host orchestration function ---
    block1_end = run_func_line - 1
    while block1_end > first_kernel_line and not lines[block1_end].strip():
        block1_end -= 1

    # --- Find timed region (for Block 2 END) ---
    timed_start: Optional[int] = None
    timed_end: Optional[int] = None
    for idx in range(run_func_line, run_func_end):
        if re.search(r"cudaEventRecord\s*\(\s*(?:ev_)?start", lines[idx]):
            timed_start = idx
        elif timed_start is not None and re.search(
            r"cudaEventRecord\s*\(\s*(?:ev_)?stop", lines[idx]
        ):
            timed_end = idx
            break

    if timed_start is None or timed_end is None:
        logger.warning("insert_evolve_markers: could not find timed region")
        return code

    # --- Block 2 START: before the first ncclMemAlloc (comm infra) ---
    # Fall back to right after cudaEventRecord(start) if no ncclMemAlloc found.
    comm_infra_re = re.compile(r"\bncclMemAlloc\b")
    comm_infra_start: Optional[int] = None
    for idx in range(run_func_line, timed_end):
        if comm_infra_re.search(lines[idx]):
            comm_infra_start = idx
            break

    if comm_infra_start is not None:
        # Walk backwards over preceding comment/blank lines to include section header
        launch_first = comm_infra_start
        while launch_first > run_func_line and (
            not lines[launch_first - 1].strip()
            or lines[launch_first - 1].strip().startswith("//")
        ):
            launch_first -= 1
    else:
        # Fallback: start right after cudaEventRecord(start, ...)
        launch_first = timed_start + 1
        while launch_first < timed_end and not lines[launch_first].strip():
            launch_first += 1

    # --- Block 2 END: right before cudaEventRecord(stop, ...) ---
    launch_last = timed_end - 1
    while launch_last > launch_first and not lines[launch_last].strip():
        launch_last -= 1
    while launch_last > launch_first and lines[launch_last].strip().startswith("//"):
        launch_last -= 1

    if launch_first >= timed_end:
        logger.warning("insert_evolve_markers: empty timed region")
        return code

    # --- Insert markers (bottom-to-top so earlier inserts don't shift later positions) ---
    launch_indent = ""
    leading = lines[launch_first]
    stripped_l = leading.lstrip()
    if stripped_l:
        launch_indent = leading[: len(leading) - len(stripped_l)]

    kernel_indent = ""
    leading = lines[first_kernel_line]
    stripped_k = leading.lstrip()
    if stripped_k:
        kernel_indent = leading[: len(leading) - len(stripped_k)]

    lines.insert(launch_last + 1, f"{launch_indent}// EVOLVE-BLOCK-END")
    lines.insert(launch_first, f"{launch_indent}// EVOLVE-BLOCK-START")
    lines.insert(block1_end + 1, f"{kernel_indent}// EVOLVE-BLOCK-END")
    lines.insert(first_kernel_line, f"{kernel_indent}// EVOLVE-BLOCK-START")

    logger.info("insert_evolve_markers (regex): inserted 2 EVOLVE-BLOCK regions")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------

class HostToDeviceTransformer:
    """Transforms CUDA code from host-side NCCL collectives to device-side GIN/LSA.

    Pipeline:
        1. CUDAAnalyzer extracts structural information (collectives, allocs, kernels, etc.)
        2. LLM receives analysis + original code + reference examples → rewrites code
        3. Build (nvcc) the rewritten code
        4. Run (mpirun) the binary
        5. Verify correctness ("Verification: PASS")
        6. Re-analyze to confirm no host-side collectives remain
        7. If any step fails, LLM Judge generates feedback → loop back to step 2
    """

    def __init__(self, config: TransformConfig):
        self.config = config
        self._accumulated_cost: float = 0.0

    def transform(self, source_path: str | Path, work_dir: Optional[str | Path] = None) -> TransformResult:
        """Run the full transformation pipeline.

        If two_stage is True (default), runs:
          Stage A: Add GIN infrastructure (keep host-side collectives) — must build+pass
          Stage B: Replace host-side collectives with GIN kernel(s) — must build+pass+no host comms

        Args:
            source_path: Path to the original CUDA source file.
            work_dir: Directory for build artifacts. Defaults to source_path's parent.

        Returns:
            TransformResult with the final code and iteration history.
        """
        source_path = Path(source_path).resolve()
        if work_dir is None:
            work_dir = source_path.parent
        else:
            work_dir = Path(work_dir).resolve()
        work_dir.mkdir(parents=True, exist_ok=True)

        self._accumulated_cost = 0.0
        start_time = time.perf_counter()
        original_code = source_path.read_text(encoding="utf-8")

        # Log model configuration
        if not self.config.judge_model:
            logger.info(f"Single-LLM mode: rewriter and judge both use {self.config.rewrite_model}")
        else:
            logger.info(f"Rewriter: {self.config.rewrite_model} | Judge: {self.config.judge_model}")

        # Step 1: Analyze the original code
        analyzer = CUDAAnalyzer(source_path)
        analysis = analyzer.analyze()
        analysis_text = analysis.format_for_llm()

        if not analysis.has_host_communication():
            logger.info("No host-side NCCL collectives found — nothing to transform.")
            return TransformResult(
                success=True,
                final_code=original_code,
                source_filename=source_path.name,
                total_duration_sec=time.perf_counter() - start_time,
                total_cost=self._accumulated_cost,
            )

        logger.info(
            f"Found {len(analysis.nccl_collectives)} host-side NCCL collective(s) to replace: "
            f"{[c.name for c in analysis.nccl_collectives]}"
        )

        all_iterations: List[IterationResult] = []

        if self.config.two_stage:
            # ============================================================
            # STAGE A: Infrastructure (keep host-side collectives, add GIN setup)
            # ============================================================
            logger.info("=" * 60)
            api_label = self.config.api_type.upper()
            logger.info(f"STAGE A: Adding {api_label} infrastructure (keeping host-side collectives)")
            logger.info("=" * 60)

            stage_a_result = self._run_stage(
                original_code=original_code,
                analysis_text=analysis_text,
                work_dir=work_dir,
                system_prompt=_rewrite_system_prompt(self.config.api_type),
                max_iterations=self.config.stage_a_max_iterations,
                stage_name="A",
                require_no_host_comms=False,  # Stage A keeps host-side collectives
                all_iterations=all_iterations,
            )

            if not stage_a_result:
                total_dur = time.perf_counter() - start_time
                return TransformResult(
                    success=False,
                    final_code=all_iterations[-1].code if all_iterations else original_code,
                    source_filename=source_path.name,
                    iterations=all_iterations,
                    total_duration_sec=total_dur,
                    total_cost=self._accumulated_cost,
                    error=f"Stage A (infrastructure) failed after {len(all_iterations)} iterations.",
                )

            infra_code = stage_a_result

            # ============================================================
            # STAGE B: Replace host-side collectives with GIN kernel(s)
            # ============================================================
            logger.info("=" * 60)
            logger.info(f"STAGE B: Replacing host-side collectives with {api_label} kernel(s)")
            logger.info("=" * 60)

            # Re-analyze the infrastructure code for Stage B prompt
            temp_path = work_dir / "_stage_b_input.cu"
            temp_path.write_text(infra_code, encoding="utf-8")
            stage_b_analysis = CUDAAnalyzer(temp_path).analyze()
            stage_b_analysis_text = stage_b_analysis.format_for_llm()
            temp_path.unlink(missing_ok=True)

            stage_b_result = self._run_stage(
                original_code=infra_code,  # Stage B starts from Stage A's output
                analysis_text=stage_b_analysis_text,
                work_dir=work_dir,
                system_prompt=_rewrite_system_prompt(self.config.api_type),
                max_iterations=self.config.stage_b_max_iterations,
                stage_name="B",
                require_no_host_comms=True,  # Stage B must eliminate all host comms
                all_iterations=all_iterations,
            )

            total_dur = time.perf_counter() - start_time
            if stage_b_result:
                marked_code = insert_evolve_markers(
                    stage_b_result, llm_model=self.config.rewrite_model
                )
                return TransformResult(
                    success=True,
                    final_code=marked_code,
                    source_filename=source_path.name,
                    iterations=all_iterations,
                    total_duration_sec=total_dur,
                    total_cost=self._accumulated_cost,
                )
            else:
                return TransformResult(
                    success=False,
                    final_code=all_iterations[-1].code if all_iterations else original_code,
                    source_filename=source_path.name,
                    iterations=all_iterations,
                    total_duration_sec=total_dur,
                    total_cost=self._accumulated_cost,
                    error=f"Stage B (kernel replacement) failed. Total iterations: {len(all_iterations)}.",
                )

        else:
            # Single-stage mode (original behavior)
            result_code = self._run_stage(
                original_code=original_code,
                analysis_text=analysis_text,
                work_dir=work_dir,
                system_prompt=_rewrite_system_prompt(self.config.api_type),
                max_iterations=self.config.max_iterations,
                stage_name="",
                require_no_host_comms=True,
                all_iterations=all_iterations,
            )

            total_dur = time.perf_counter() - start_time
            error = ""
            if not result_code:
                error = (
                    f"Failed to fully transform after {len(all_iterations)} iterations. "
                    f"Last issue: {all_iterations[-1].judge_feedback[:200] if all_iterations else 'unknown'}"
                )

            final = result_code or (all_iterations[-1].code if all_iterations else original_code)
            if result_code:
                final = insert_evolve_markers(
                    final, llm_model=self.config.rewrite_model
                )

            return TransformResult(
                success=bool(result_code),
                final_code=final,
                source_filename=source_path.name,
                iterations=all_iterations,
                total_duration_sec=total_dur,
                total_cost=self._accumulated_cost,
                error=error,
            )

    def _run_stage(
        self,
        original_code: str,
        analysis_text: str,
        work_dir: Path,
        system_prompt: str,
        max_iterations: int,
        stage_name: str,
        require_no_host_comms: bool,
        all_iterations: List[IterationResult],
    ) -> Optional[str]:
        """Run one stage of the transformation (build/verify loop).

        Returns the successful code string, or None if all iterations failed.
        """
        current_code = original_code
        stage_iterations: List[IterationResult] = []
        stage_label = f"Stage {stage_name}: " if stage_name else ""
        iter_offset = len(all_iterations)  # for globally-unique iteration numbering

        for i in range(1, max_iterations + 1):
            iteration = iter_offset + i
            logger.info(f"=== {stage_label}Iteration {i}/{max_iterations} (global #{iteration}) ===")
            iter_start = time.perf_counter()

            # LLM Rewrite
            if i == 1:
                rewrite_prompt = self._build_initial_rewrite_prompt(
                    original_code, analysis_text, stage_name=stage_name
                )
            else:
                prev = stage_iterations[-1]
                rewrite_prompt = self._build_feedback_rewrite_prompt(
                    original_code, current_code, prev, stage_iterations
                )

            rewritten_code = self._llm_rewrite(rewrite_prompt, system_prompt)
            if not rewritten_code:
                logger.error(f"{stage_label}LLM returned empty code. Stopping.")
                empty_result = IterationResult(
                    iteration=iteration,
                    code="",
                    build_success=False,
                    build_errors="LLM returned empty response",
                    rewrite_prompt=rewrite_prompt,
                    system_prompt_name=stage_name,
                    system_prompt=system_prompt,
                    duration_sec=time.perf_counter() - iter_start,
                )
                stage_iterations.append(empty_result)
                all_iterations.append(empty_result)
                self._save_iteration(work_dir, empty_result)
                break

            current_code = rewritten_code

            # Build
            build_ok, build_errors = self._build(current_code, work_dir)

            if not build_ok:
                logger.warning(f"{stage_label}Iteration {i}: Build failed.")
                has_host = self._has_host_comms(current_code)
                judge_fb = self._judge(
                    original_code, current_code,
                    f"BUILD FAILED.\nCompiler errors:\n{build_errors}"
                )
                iter_result = IterationResult(
                    iteration=iteration, code=current_code,
                    build_success=False, build_errors=build_errors,
                    has_host_communication=has_host, judge_feedback=judge_fb,
                    rewrite_prompt=rewrite_prompt,
                    system_prompt_name=stage_name,
                    system_prompt=system_prompt,
                    duration_sec=time.perf_counter() - iter_start,
                )
                stage_iterations.append(iter_result)
                all_iterations.append(iter_result)
                self._save_iteration(work_dir, iter_result)
                continue

            # Run
            run_ok, run_output = self._run(work_dir)

            if not run_ok:
                logger.warning(f"{stage_label}Iteration {i}: Run failed.")

                # If memory error, do a diagnostic re-run with sync after every kernel
                # to pinpoint WHICH kernel crashes (general-purpose, no reference needed)
                diag_output = ""
                if "illegal memory access" in run_output or "misaligned address" in run_output:
                    logger.info(f"{stage_label}Running diagnostic build to isolate faulting kernel...")
                    diag_output = self._diagnostic_rerun(current_code, work_dir)

                has_host = self._has_host_comms(current_code)
                outcome_parts = [f"RUN FAILED.\nOutput:\n{run_output}"]
                if diag_output:
                    outcome_parts.append(
                        f"\n--- Diagnostic re-run (cudaDeviceSynchronize after each kernel) ---\n"
                        f"{diag_output}"
                    )
                judge_fb = self._judge(
                    original_code, current_code,
                    "\n".join(outcome_parts)
                )

                full_run_output = run_output
                if diag_output:
                    full_run_output += f"\n\n--- Diagnostic re-run ---\n{diag_output}"

                iter_result = IterationResult(
                    iteration=iteration, code=current_code,
                    build_success=True, run_success=False, run_output=full_run_output,
                    has_host_communication=has_host, judge_feedback=judge_fb,
                    rewrite_prompt=rewrite_prompt,
                    system_prompt_name=stage_name,
                    system_prompt=system_prompt,
                    duration_sec=time.perf_counter() - iter_start,
                )
                stage_iterations.append(iter_result)
                all_iterations.append(iter_result)
                self._save_iteration(work_dir, iter_result)
                continue

            # Verify
            verification_passed = self.config.verification_pass_str in run_output
            time_ms = self._parse_time(run_output)

            if not verification_passed:
                logger.warning(f"{stage_label}Iteration {i}: Verification failed.")
                has_host = self._has_host_comms(current_code)
                judge_fb = self._judge(
                    original_code, current_code,
                    f"VERIFICATION FAILED. Output:\n{run_output}"
                )
                iter_result = IterationResult(
                    iteration=iteration, code=current_code,
                    build_success=True, run_success=True, run_output=run_output,
                    verification_passed=False, time_ms=time_ms,
                    has_host_communication=has_host, judge_feedback=judge_fb,
                    rewrite_prompt=rewrite_prompt,
                    system_prompt_name=stage_name,
                    system_prompt=system_prompt,
                    duration_sec=time.perf_counter() - iter_start,
                )
                stage_iterations.append(iter_result)
                all_iterations.append(iter_result)
                self._save_iteration(work_dir, iter_result)
                continue

            # Check for remaining host-side collectives
            has_host_comm = self._has_host_comms(current_code)

            if require_no_host_comms and has_host_comm:
                logger.warning(f"{stage_label}Iteration {i}: Still has host-side collectives.")
                judge_fb = self._judge(
                    original_code, current_code,
                    f"PARTIAL SUCCESS. Verification passed (time: {time_ms} ms), but host-side "
                    f"collective calls still remain. They must be replaced with GIN."
                )
                iter_result = IterationResult(
                    iteration=iteration, code=current_code,
                    build_success=True, run_success=True, run_output=run_output,
                    verification_passed=True, time_ms=time_ms,
                    has_host_communication=True, judge_feedback=judge_fb,
                    rewrite_prompt=rewrite_prompt,
                    system_prompt_name=stage_name,
                    system_prompt=system_prompt,
                    duration_sec=time.perf_counter() - iter_start,
                )
                stage_iterations.append(iter_result)
                all_iterations.append(iter_result)
                self._save_iteration(work_dir, iter_result)
                continue

            # SUCCESS for this stage!
            logger.info(
                f"{stage_label}Iteration {i}: SUCCESS! "
                f"Verification passed, time: {time_ms} ms. "
                f"Host comms remaining: {has_host_comm}"
            )
            iter_result = IterationResult(
                iteration=iteration, code=current_code,
                build_success=True, run_success=True, run_output=run_output,
                verification_passed=True, time_ms=time_ms,
                has_host_communication=has_host_comm,
                rewrite_prompt=rewrite_prompt,
                system_prompt_name=stage_name,
                system_prompt=system_prompt,
                duration_sec=time.perf_counter() - iter_start,
            )
            stage_iterations.append(iter_result)
            all_iterations.append(iter_result)
            self._save_iteration(work_dir, iter_result)
            return current_code

        return None  # All iterations exhausted

    @staticmethod
    def _has_host_comms(code: str) -> bool:
        """Check for host-side NCCL collective calls (ignoring comments)."""
        for line in code.split("\n"):
            stripped = line.lstrip()
            if stripped.startswith("//") or stripped.startswith("*"):
                continue
            if re.search(
                r'\bncclAllReduce\s*\(|\bncclAllGather\s*\(|\bncclReduceScatter\s*\('
                r'|\bncclBroadcast\s*\(|\bncclReduce\s*\(|\bncclSend\s*\(|\bncclRecv\s*\('
                r'|\bncclAlltoAll\s*\(|\bncclAllToAll\s*\(',
                line
            ):
                return True
        return False

    # ------------------------------------------------------------------
    # Per-iteration artifact saving
    # ------------------------------------------------------------------

    @staticmethod
    def _save_iteration(work_dir: Path, iter_result: IterationResult) -> None:
        """Save an iteration's artifacts to work_dir for live inspection.

        Creates files like:
          iter_1_code.cu, iter_1_result.txt
          iter_2_code.cu, iter_2_result.txt
          ...
        """
        prefix = f"iter_{iter_result.iteration}"
        try:
            # Save the LLM-generated code
            if iter_result.code:
                (work_dir / f"{prefix}_code.cu").write_text(
                    iter_result.code, encoding="utf-8"
                )

            # Save a combined result report
            lines = [
                f"=== Iteration {iter_result.iteration} ===",
                f"System Prompt: {iter_result.system_prompt_name or 'N/A'}",
                f"Build: {'OK' if iter_result.build_success else 'FAILED'}",
            ]
            if iter_result.build_errors:
                lines.append(f"\n--- Build Errors ---\n{iter_result.build_errors}")
            if iter_result.build_success:
                lines.append(f"Run: {'OK' if iter_result.run_success else 'FAILED'}")
            if iter_result.run_output:
                lines.append(f"\n--- Run Output ---\n{iter_result.run_output}")
            lines.append(
                f"Verification: {'PASS' if iter_result.verification_passed else 'FAIL'}"
            )
            if iter_result.time_ms is not None:
                lines.append(f"Time: {iter_result.time_ms:.4f} ms")
            lines.append(
                f"Host-side comms remaining: {iter_result.has_host_communication}"
            )
            if iter_result.judge_feedback:
                lines.append(f"\n--- Diagnosis ---\n{iter_result.judge_feedback}")
            lines.append(f"\nDuration: {iter_result.duration_sec:.1f}s")

            (work_dir / f"{prefix}_result.txt").write_text(
                "\n".join(lines), encoding="utf-8"
            )

            # Save the full rewrite (user) prompt separately (can be large)
            if iter_result.rewrite_prompt:
                (work_dir / f"{prefix}_prompt.txt").write_text(
                    iter_result.rewrite_prompt, encoding="utf-8"
                )
            # Save system prompt so "Stage A" vs "Stage B" instructions are visible
            if iter_result.system_prompt:
                (work_dir / f"{prefix}_system_prompt.txt").write_text(
                    iter_result.system_prompt, encoding="utf-8"
                )
        except Exception as exc:
            logger.warning(f"Failed to save iteration {iter_result.iteration} artifacts: {exc}")

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _build_initial_rewrite_prompt(
        self, original_code: str, analysis_text: str, stage_name: str = ""
    ) -> str:
        api = self.config.api_type
        task_prompts = {
            "A": _stage_a_task_prompt(api),
            "B": _stage_b_task_prompt(api),
        }
        task = task_prompts.get(stage_name, _single_stage_task_prompt(api))

        parts = [
            "## Original CUDA Source Code\n",
            f"```cuda\n{original_code}\n```\n",
            "## Automated Analysis\n",
            analysis_text,
        ]

        if self.config.reference_code:
            api_label = api.upper()
            parts.append(f"## Reference: Working {api_label} Implementation\n")
            parts.append(f"```cuda\n{self.config.reference_code}\n```\n")

        parts.append(f"\n{task}")

        if self.config.nccl_api_docs:
            parts.append("\n## NCCL Device API Reference\n")
            parts.append(self.config.nccl_api_docs)

        return "\n".join(parts)

    def _build_feedback_rewrite_prompt(
        self, original_code: str, current_code: str,
        prev: IterationResult, all_iterations: List[IterationResult]
    ) -> str:
        parts = [
            "## Original Code (host-side NCCL, for reference)\n",
            f"```cuda\n{original_code}\n```\n",
        ]

        # --- CUMULATIVE FAILURE LOG (all past iterations) ---
        # This prevents the LLM from oscillating between the same two bugs
        if len(all_iterations) > 1:
            parts.append("## History of ALL Previous Attempts (DO NOT repeat these mistakes!)\n")
            for it in all_iterations:
                status = ""
                if not it.build_success:
                    # Extract the first error line
                    first_err = it.build_errors.split('\n')[0][:200] if it.build_errors else "unknown"
                    status = f"BUILD FAIL: {first_err}"
                elif not it.run_success:
                    first_err = it.run_output.split('\n')[0][:200] if it.run_output else "unknown"
                    status = f"RUN FAIL: {first_err}"
                elif not it.verification_passed:
                    status = f"VERIFICATION FAIL (time: {it.time_ms} ms)"
                elif it.has_host_communication:
                    status = f"PARTIAL: passed but still has host-side calls"
                else:
                    status = f"SUCCESS (time: {it.time_ms} ms)"
                parts.append(f"  - Iteration {it.iteration}: {status}\n")
            parts.append("\n")

        # --- Latest attempt + outcome ---
        parts.append("## Your Latest Attempt (iteration {})\n".format(prev.iteration))
        parts.append(f"```cuda\n{current_code}\n```\n")
        parts.append("## Outcome of Latest Attempt\n")

        if not prev.build_success:
            parts.append(f"BUILD FAILED. Compiler errors:\n```\n{prev.build_errors[:3000]}\n```\n")
        elif not prev.run_success:
            parts.append(f"RUN FAILED. Output:\n```\n{prev.run_output[:3000]}\n```\n")
        elif not prev.verification_passed:
            parts.append(
                f"VERIFICATION FAILED (time: {prev.time_ms} ms). Output:\n"
                f"```\n{prev.run_output[:3000]}\n```\n"
            )
        elif prev.has_host_communication:
            parts.append(
                f"PARTIAL: Verification passed (time: {prev.time_ms} ms), but host-side "
                f"NCCL collective calls still remain in the code. They must be fully replaced with GIN.\n"
            )

        parts.append("## Diagnosis of Latest Attempt\n")
        parts.append(prev.judge_feedback + "\n")
        parts.append(
            "\n## Task\n"
            "Fix the issues described in the diagnosis above. Return the COMPLETE corrected file "
            "between ```cuda and ``` markers.\n"
            "Do NOT revert to any host-side NCCL collective. The goal is device-side GIN only.\n"
            "Refer to the NCCL Device API Reference below for correct API usage.\n"
        )

        if self.config.reference_code:
            parts.append("## Reference: Working GIN Implementation\n")
            parts.append(f"```cuda\n{self.config.reference_code}\n```\n")

        if self.config.nccl_api_docs:
            parts.append("\n## NCCL Device API Reference\n")
            parts.append(self.config.nccl_api_docs)

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # LLM calls
    # ------------------------------------------------------------------

    def _llm_rewrite(self, prompt: str, system_prompt: str = "") -> str:
        """Call the LLM to rewrite the CUDA code. Returns extracted code or empty string."""
        try:
            result = self._query_llm(
                model=self.config.rewrite_model,
                system_msg=system_prompt or _rewrite_system_prompt(self.config.api_type),
                user_msg=prompt,
                max_tokens=self.config.rewrite_max_tokens,
                temperature=self.config.rewrite_temperature,
            )
            if not result:
                return ""

            # Extract code from ```cuda ... ``` block
            code = self._extract_code_block(result)
            if not code:
                logger.warning("Could not extract ```cuda code block from LLM response.")
                # Try using the full response as code (last resort)
                if "#include" in result and "__global__" in result:
                    return result
                return ""
            return code
        except Exception as exc:
            logger.error(f"LLM rewrite failed: {exc}")
            return ""

    def _judge(self, original_code: str, current_code: str, outcome: str) -> str:
        """Call the judge LLM to diagnose issues and provide feedback.

        Uses the same model as the rewriter (single LLM) unless judge_model
        is explicitly set to a different model.
        """
        try:
            # Build judge system prompt with full NCCL API docs (same docs the rewriter sees)
            judge_system = JUDGE_SYSTEM_PROMPT_TEMPLATE.format(
                nccl_api_docs=self.config.nccl_api_docs or "(No NCCL API docs provided.)"
            )
            prompt = (
                f"## Original Code\n```cuda\n{original_code}\n```\n\n"
                f"## Transformed Code\n```cuda\n{current_code}\n```\n\n"
                f"## Outcome\n{outcome}\n"
            )
            # Use rewriter model if judge_model is empty (single-LLM mode)
            model = self.config.judge_model or self.config.rewrite_model
            result = self._query_llm(
                model=model,
                system_msg=judge_system,
                user_msg=prompt,
                max_tokens=self.config.judge_max_tokens,
                temperature=self.config.judge_temperature,
            )
            return result or "No feedback generated."
        except Exception as exc:
            logger.warning(f"Judge LLM failed: {exc}")
            return f"Judge LLM error: {exc}"

    def _query_llm(
        self, model: str, system_msg: str, user_msg: str,
        max_tokens: int, temperature: float
    ) -> str:
        """Query LLM. Dispatches to Claude CLI or Anthropic API based on model prefix."""
        if model.startswith("claude-cli/"):
            return self._query_claude_cli(model, system_msg, user_msg)
        return self._query_anthropic(model, system_msg, user_msg, max_tokens, temperature)

    @staticmethod
    def _query_claude_cli(model: str, system_msg: str, user_msg: str) -> str:
        """Query via Claude Code CLI (``claude -p``)."""
        from ..llm.models.claude_cli import CLAUDE_CLI_MODEL_MAP

        model_alias = CLAUDE_CLI_MODEL_MAP.get(model, "opus")
        cmd = [
            "claude", "-p",
            "--model", model_alias,
            "--output-format", "text",
            "--dangerously-skip-permissions",
            "--no-session-persistence",
            "--tools", "",
        ]
        if system_msg:
            cmd.extend(["--system-prompt", system_msg])

        env = os.environ.copy()
        for p in ["/usr/bin", "/usr/local/bin", os.path.expanduser("~/.local/bin")]:
            if p not in env.get("PATH", ""):
                env["PATH"] = p + ":" + env.get("PATH", "")

        logger.info(f"Querying Claude CLI: model={model_alias}")
        result = subprocess.run(
            cmd, input=user_msg,
            capture_output=True, text=True, timeout=600, env=env,
        )
        content = (result.stdout or "").strip()
        if result.returncode != 0 and not content:
            logger.warning(f"Claude CLI exit {result.returncode}: {(result.stderr or '')[:300]}")
        return content

    def _query_anthropic(
        self, model: str, system_msg: str, user_msg: str,
        max_tokens: int, temperature: float,
    ) -> str:
        """Query via Anthropic API (direct or Bedrock). Accumulates cost."""
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package required. Install with: pip install anthropic")

        from ..llm.models.pricing import CLAUDE_MODELS, BEDROCK_MODELS

        model_name = model
        if model_name.startswith("bedrock/"):
            actual_model = model_name.split("/", 1)[1]
            client = anthropic.AnthropicBedrock(
                aws_access_key=os.environ.get("AWS_ACCESS_KEY_ID"),
                aws_secret_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
                aws_region=os.environ.get("AWS_REGION_NAME"),
            )
        else:
            actual_model = model_name
            client = anthropic.Anthropic()

        logger.info(f"Querying LLM: {actual_model} (max_tokens={max_tokens})")

        with client.messages.stream(
            model=actual_model,
            system=system_msg,
            messages=[{"role": "user", "content": user_msg}],
            max_tokens=max_tokens,
            temperature=temperature,
        ) as stream:
            response = stream.get_final_message()

        if response.usage:
            pricing = (
                BEDROCK_MODELS.get(model_name)
                or CLAUDE_MODELS.get(actual_model)
                or {}
            )
            in_cost = pricing.get("input_price", 0) * response.usage.input_tokens
            out_cost = pricing.get("output_price", 0) * response.usage.output_tokens
            call_cost = in_cost + out_cost
            self._accumulated_cost += call_cost
            logger.info(
                f"LLM cost: ${call_cost:.4f} "
                f"(in={response.usage.input_tokens}, out={response.usage.output_tokens}), "
                f"accumulated: ${self._accumulated_cost:.4f}"
            )

        if response.content and len(response.content) > 0:
            return response.content[0].text.strip()
        return ""

    # ------------------------------------------------------------------
    # Build & Run
    # ------------------------------------------------------------------

    def _build(self, code: str, work_dir: Path) -> Tuple[bool, str]:
        """Write code to file and compile with nvcc. Returns (success, error_text)."""
        source_file = work_dir / f"{self.config.binary_name}.cu"
        source_file.write_text(code, encoding="utf-8")

        cmd = [
            self.config.nvcc_path,
            "-o", self.config.binary_name,
            str(source_file),
            f"-I{self.config.nccl_include}",
            self.config.nccl_static_lib,
            "-rdc=true", "-arch=sm_80", "-lineinfo",
            f"-L{self.config.cuda_lib64}", "-lcudart", "-lcudadevrt", "-lpthread",
            f"-I{self.config.mpi_include}", f"-I{self.config.mpi_include_openmpi}",
            f"-L{self.config.mpi_lib}", "-lmpi",
        ]

        # Ensure nvcc can find the host compiler (g++) even in virtual envs,
        # and use the work_dir for temp files (root / may be full).
        env = os.environ.copy()
        essential_paths = ["/usr/bin", "/usr/local/bin", "/usr/sbin"]
        current_path = env.get("PATH", "")
        for p in essential_paths:
            if p not in current_path:
                current_path = p + ":" + current_path
        env["PATH"] = current_path
        env["TMPDIR"] = str(work_dir)

        try:
            result = subprocess.run(
                cmd,
                cwd=str(work_dir),
                capture_output=True,
                text=True,
                timeout=60,
                env=env,
            )
            if result.returncode != 0:
                errors = (result.stderr or result.stdout or "")[:4000]
                return False, errors
            return True, ""
        except subprocess.TimeoutExpired:
            return False, "Build timed out (60s)"
        except Exception as exc:
            return False, str(exc)

    def _build_mpirun_cmd(self, binary_path: str) -> list:
        """Build the full mpirun command from config (hostfile, env vars, extra args)."""
        cmd = ["mpirun"]
        if self.config.hostfile:
            cmd += ["--hostfile", self.config.hostfile]
        cmd += ["-np", str(self.config.num_mpi_ranks)]
        cmd += list(self.config.mpirun_extra_args)
        cmd += ["-x", "LD_LIBRARY_PATH", "-x", "CUDA_VISIBLE_DEVICES"]
        for key, val in self.config.run_env_vars.items():
            cmd += ["-x", f"{key}={val}"]
        cmd.append(binary_path)
        return cmd

    def _build_run_env(self, work_dir: Path) -> dict:
        """Build the environment dict for mpirun execution."""
        env = os.environ.copy()
        essential_paths = ["/usr/bin", "/usr/local/bin", "/usr/sbin"]
        current_path = env.get("PATH", "")
        for p in essential_paths:
            if p not in current_path:
                current_path = p + ":" + current_path
        env["PATH"] = current_path
        env["TMPDIR"] = str(work_dir)
        env["LD_LIBRARY_PATH"] = (
            f"{self.config.cuda_lib64}:{self.config.mpi_lib}:"
            f"{env.get('LD_LIBRARY_PATH', '')}"
        )
        env["CUDA_VISIBLE_DEVICES"] = self.config.cuda_visible_devices
        return env

    def _run(self, work_dir: Path) -> Tuple[bool, str]:
        """Run the binary with mpirun. Returns (success, output_text)."""
        binary = work_dir / self.config.binary_name
        if not binary.exists():
            return False, f"Binary not found: {binary}"

        env = self._build_run_env(work_dir)
        cmd = self._build_mpirun_cmd(str(binary))

        try:
            result = subprocess.run(
                cmd,
                cwd=str(work_dir),
                env=env,
                capture_output=True,
                text=True,
                timeout=self.config.run_timeout,
            )
            output = (result.stdout or "") + "\n" + (result.stderr or "")
            if result.returncode != 0:
                return False, output[:4000]
            return True, output[:4000]
        except subprocess.TimeoutExpired:
            return False, "TIMEOUT: Program did not finish within {}s (likely deadlock or infinite loop).".format(
                self.config.run_timeout
            )
        except Exception as exc:
            return False, str(exc)

    def _diagnostic_rerun(self, code: str, work_dir: Path) -> str:
        """Inject cudaDeviceSynchronize() after every kernel launch, rebuild, re-run.

        This isolates WHICH kernel causes the illegal memory access, turning a
        vague "error detected at cudaEventSynchronize" into a precise
        "error at cudaDeviceSynchronize() after ginKernel<<<...>>>".

        Completely general — works on any CUDA code, no reference needed.
        Returns the diagnostic output string, or empty on failure.
        """
        try:
            # Regex: find <<<...>>> kernel launches followed by their error check,
            # and inject a cudaDeviceSynchronize + check right after.
            # Pattern: matches "kernelName<<<grid, block, ...>>>(args);\n    CHECK(...);"
            # and adds a sync + check after.
            diag_code = re.sub(
                r'(<<<[^>]*>>>(?:\s*\([^;]*\);))\s*\n(\s*)(CUDA_CHECK\(cudaGetLastError\(\)\);)',
                r'\1\n\2\3\n\2CUDA_CHECK(cudaDeviceSynchronize());  // DIAGNOSTIC: sync to isolate faulting kernel',
                code
            )

            # Also add sync after any bare kernel launch without an error check
            # (in case the code doesn't have CUDA_CHECK after every launch)
            if diag_code == code:
                # Fallback: just add cudaDeviceSynchronize after every <<<>>> launch
                diag_code = re.sub(
                    r'(<<<[^>]*>>>\s*\([^;]*\);)',
                    r'\1\n    CUDA_CHECK(cudaDeviceSynchronize());  // DIAGNOSTIC',
                    code
                )

            if diag_code == code:
                logger.info("Diagnostic: could not inject sync points (no kernel launches found).")
                return ""

            # Build diagnostic version
            diag_binary = self.config.binary_name + "_diag"
            source_file = work_dir / f"{diag_binary}.cu"
            source_file.write_text(diag_code, encoding="utf-8")

            build_cmd = [
                self.config.nvcc_path,
                "-o", diag_binary,
                str(source_file),
                f"-I{self.config.nccl_include}",
                self.config.nccl_static_lib,
                "-rdc=true", "-arch=sm_80", "-lineinfo",
                f"-L{self.config.cuda_lib64}", "-lcudart", "-lcudadevrt", "-lpthread",
                f"-I{self.config.mpi_include}", f"-I{self.config.mpi_include_openmpi}",
                f"-L{self.config.mpi_lib}", "-lmpi",
            ]

            env = os.environ.copy()
            for p in ["/usr/bin", "/usr/local/bin", "/usr/sbin"]:
                if p not in env.get("PATH", ""):
                    env["PATH"] = p + ":" + env.get("PATH", "")
            env["TMPDIR"] = str(work_dir)

            build_result = subprocess.run(
                build_cmd, cwd=str(work_dir), capture_output=True,
                text=True, timeout=60, env=env,
            )
            if build_result.returncode != 0:
                logger.warning("Diagnostic build failed.")
                return ""

            # Run diagnostic version
            env = self._build_run_env(work_dir)
            run_cmd = self._build_mpirun_cmd(str(work_dir / diag_binary))

            run_result = subprocess.run(
                run_cmd, cwd=str(work_dir), capture_output=True,
                text=True, timeout=self.config.run_timeout, env=env,
            )
            output = (run_result.stdout or "") + "\n" + (run_result.stderr or "")

            # Clean up diagnostic binary
            (work_dir / diag_binary).unlink(missing_ok=True)
            source_file.unlink(missing_ok=True)

            return output[:4000]
        except subprocess.TimeoutExpired:
            return "Diagnostic re-run timed out (likely deadlock after sync injection)."
        except Exception as exc:
            logger.warning(f"Diagnostic re-run failed: {exc}")
            return ""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_code_block(text: str) -> str:
        """Extract code from ```cuda ... ``` or ```cpp ... ``` or ``` ... ``` blocks."""
        # Try ```cuda first
        pattern = re.compile(r'```(?:cuda|cpp|c\+\+)?\s*\n(.*?)```', re.DOTALL)
        matches = pattern.findall(text)
        if matches:
            # Return the longest match (the full file)
            return max(matches, key=len).strip()
        return ""

    @staticmethod
    def _parse_time(output: str) -> Optional[float]:
        """Parse 'Time: X.XXXX ms' from output."""
        m = re.search(r"Time:\s*([\d.]+)\s*ms", output, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
        return None

    # ------------------------------------------------------------------
    # Convenience: save/load results
    # ------------------------------------------------------------------

    def save_result(self, result: TransformResult, output_dir: str | Path) -> None:
        """Save transformation result to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save final code with descriptive name: <original_stem>_device.cu
        if result.final_code:
            (output_dir / result.device_filename).write_text(result.final_code, encoding="utf-8")

        # Save summary
        summary = {
            "success": result.success,
            "total_duration_sec": result.total_duration_sec,
            "num_iterations": len(result.iterations),
            "error": result.error,
            "iterations": [],
        }
        for it in result.iterations:
            summary["iterations"].append({
                "iteration": it.iteration,
                "build_success": it.build_success,
                "build_errors": it.build_errors[:500] if it.build_errors else "",
                "run_success": it.run_success,
                "verification_passed": it.verification_passed,
                "time_ms": it.time_ms,
                "has_host_communication": it.has_host_communication,
                "judge_feedback": it.judge_feedback[:500] if it.judge_feedback else "",
                "duration_sec": it.duration_sec,
            })

        (output_dir / "transform_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

        # Save each iteration's code
        for it in result.iterations:
            if it.code:
                (output_dir / f"iteration_{it.iteration}.cu").write_text(
                    it.code, encoding="utf-8"
                )
