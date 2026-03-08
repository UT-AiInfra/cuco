#!/usr/bin/env python3
"""Host-to-Device GIN transformation.

Usage:
    # Agent mode (default): Claude Code gets full autonomy.
    python run_transform.py

    # Standard structured loop:
    python run_transform.py --no-agent
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

_EXAMPLE_DIR = Path(__file__).resolve().parent
_CUDA_EVOLVE_DIR = _EXAMPLE_DIR.parent.parent
if str(_CUDA_EVOLVE_DIR) not in sys.path:
    sys.path.insert(0, str(_CUDA_EVOLVE_DIR))
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))

try:
    from dotenv import load_dotenv
    load_dotenv(_CUDA_EVOLVE_DIR / ".env")
except ImportError:
    pass

from cuco.transform import CUDAAnalyzer, HostToDeviceTransformer, insert_evolve_markers
from cuco.transform.transformer import TransformConfig

# ---------------------------------------------------------------------------
# NCCL docs
# ---------------------------------------------------------------------------
try:
    from nccl_api_docs import (
        NCCL_DEVICE_API_REFERENCE, NCCL_GIN_API_DOC, NCCL_GIN_PURE_EXAMPLE,
        NCCL_THREAD_GROUPS_DOC, NCCL_TEAMS_DOC, NCCL_HOST_TO_DEVICE_COOKBOOK,
        NCCL_HEADER_GIN_H, NCCL_HEADER_CORE_H, NCCL_HEADER_COOP_H,
        NCCL_HEADER_PTR_H, NCCL_HEADER_GIN_BARRIER_H, NCCL_HEADER_BARRIER_H,
    )
    NCCL_DOCS = "\n\n".join([
        NCCL_DEVICE_API_REFERENCE, NCCL_GIN_API_DOC, NCCL_GIN_PURE_EXAMPLE,
        NCCL_HOST_TO_DEVICE_COOKBOOK, NCCL_THREAD_GROUPS_DOC, NCCL_TEAMS_DOC,
    ])
    NCCL_HEADERS = "\n\n".join([
        NCCL_HEADER_GIN_H, NCCL_HEADER_CORE_H, NCCL_HEADER_COOP_H,
        NCCL_HEADER_PTR_H, NCCL_HEADER_GIN_BARRIER_H, NCCL_HEADER_BARRIER_H,
    ])
    NCCL_API_FULL = NCCL_DOCS + "\n\n---\n\n## NCCL Headers\n\n" + NCCL_HEADERS
except ImportError:
    NCCL_DOCS = ""
    NCCL_HEADERS = ""
    NCCL_API_FULL = ""
    print("WARNING: nccl_api_docs.py not found.")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("run_transform")

NVCC = "/usr/local/cuda-13.1/bin/nvcc"
NCCL_INCLUDE = "/usr/local/nccl_2.28.9-1+cuda13.0_x86_64/include"
NCCL_STATIC_LIB = "/usr/local/nccl_2.28.9-1+cuda13.0_x86_64/lib/libnccl_static.a"
CUDA_LIB64 = "/usr/local/cuda-13.1/lib64"
MPI_INCLUDE = "/usr/lib/x86_64-linux-gnu/openmpi/include"
MPI_INCLUDE_OPENMPI = "/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi"
MPI_LIB = "/usr/lib/x86_64-linux-gnu/openmpi/lib"


# ===================================================================
# Agent mode
# ===================================================================

_AGENT_SYSTEM_PROMPT = """\
You are an expert CUDA and NCCL programmer specializing in GPU-Initiated \
Networking (GIN).  You transform CUDA programs from host-side NCCL collectives \
to device-side GIN.

Be methodical:
1. Read and understand the source code first
2. Plan the transformation
3. Implement step by step
4. Build, test, and fix any issues
5. Do not give up — keep iterating until verification passes

When you encounter errors:
- Read error messages carefully
- Check for common GIN pitfalls (wrong Coop for flush, missing __syncthreads, \
signal count mismatch)
- Use the provided API reference to verify correct usage
- Add debug prints if needed for runtime diagnosis
"""


def _build_agent_prompt(source_path: Path, work_dir: Path) -> str:
    binary_name = source_path.stem
    output_file = work_dir / f"{binary_name}_device.cu"

    build_cmd = (
        f"{NVCC} -o {binary_name} {binary_name}.cu "
        f"-I{NCCL_INCLUDE} {NCCL_STATIC_LIB} "
        f"-rdc=true -arch=sm_80 -lineinfo "
        f"-L{CUDA_LIB64} -lcudart -lcudadevrt -lpthread "
        f"-I{MPI_INCLUDE} -I{MPI_INCLUDE_OPENMPI} "
        f"-L{MPI_LIB} -lmpi"
    )
    run_cmd = f"CUDA_VISIBLE_DEVICES=0,1 mpirun --allow-run-as-root -np 2 ./{binary_name}"

    # Run CUDAAnalyzer to give the agent structural knowledge of the source
    analysis_text = ""
    try:
        analyzer = CUDAAnalyzer(source_path)
        report = analyzer.analyze()
        analysis_text = report.format_for_llm()
    except Exception as exc:
        logger.warning(f"CUDAAnalyzer failed on {source_path}: {exc}")

    parts: list[str] = []

    parts.append(f"""\
## Task: Transform CUDA Host-Side NCCL → Device-Side GIN

Transform the CUDA source file at `{source_path}` so that all host-side
NCCL collective calls (ncclAllReduce, ncclAllGather, etc.) are replaced with
device-side GIN communication (gin.put / gin.flush / gin.waitSignal).

### Working directory
All intermediate files go in: `{work_dir}`
Write the working copy as: `{work_dir}/{binary_name}.cu`
Write the final verified version to: `{output_file}`

### Build command (run from {work_dir})
```
{build_cmd}
```

### Run command (run from {work_dir})
```
{run_cmd}
```

### Success criteria
1. Code compiles without errors
2. Program runs successfully with mpirun -np 2
3. Program prints "Verification: PASS"
4. **No remaining host-side NCCL collective calls** — all communication uses GIN
""")

    if analysis_text:
        parts.append(f"""\
### Source code analysis

The following is an automated structural analysis of the source file.
It identifies the NCCL collectives to replace, buffer allocations,
existing kernels, and other relevant code structure.

{analysis_text}
""")

    parts.append("""\
### Recommended two-stage approach

**Stage A — add GIN infrastructure (keep host-side collectives):**
1. `#include <nccl_device.h>`
2. Replace `cudaMalloc` for communicated buffers with `ncclMemAlloc`
3. Add remote landing buffer(s) with `ncclMemAlloc`
4. `ncclCommWindowRegister()` for all communicated buffers (`NCCL_WIN_COLL_SYMMETRIC`)
5. `ncclDevCommCreate()` with correct requirements
6. Update cleanup: `ncclCommWindowDeregister`, `ncclMemFree`, `ncclDevCommDestroy`
7. **KEEP** all host-side NCCL collective calls untouched
8. Build, run, verify — must still pass with the original collectives

**Stage B — replace collectives with GIN kernel(s):**
1. Analyze data-flow pattern (reduce / gather / scatter / …)
2. Write GIN kernel(s) using `gin.put` / `gin.flush` / `gin.waitSignal`
3. Replace the host-side collective calls with GIN kernel launches
4. Build, run, verify — must pass with zero host-side collectives
""")

    parts.append("""\
### CRITICAL: clear stale CUDA error after ncclDevCommCreate

```cpp
NCCL_CHECK(ncclDevCommCreate(host_comm, &reqs, &devComm));
(void)cudaGetLastError();  // MUST clear stale CUDA error
```

`ncclDevCommCreate` internally triggers CUDA operations that leave a stale
`cudaErrorNotPermitted` in the error register. If you don't clear it, the
next `cudaGetLastError()` will report "operation not permitted" and crash.
""")

    parts.append(f"""\
### NCCL 2.28.9 GIN API quick reference

Header: `<nccl_device.h>` — the ONLY GIN header needed.

**Types:**
- `ncclGinSignal_t` = `uint32_t`
- `ncclTeam` = `struct {{ int nRanks, rank, stride; }}`
- `ncclGin_SignalInc{{ ncclGinSignal_t signal; }}` — increment remote signal
- `ncclGin_None{{}}` — no-op action

**Host-side setup:**
```cpp
ncclDevCommRequirements reqs;
memset(&reqs, 0, sizeof(reqs));
reqs.ginContextCount = <contexts>;
reqs.ginSignalCount  = <signals>;
ncclDevCommCreate(host_comm, &reqs, &devComm);
(void)cudaGetLastError();
ncclMemAlloc(...)
ncclCommWindowRegister(...)
```

**Device-side API (inside `__global__` kernels):**
```cpp
ncclGin gin(devComm, contextIndex);
ncclTeam world = ncclTeamWorld(devComm);
gin.put(team, peer, dstWin, dstOff, srcWin, srcOff, bytes, remoteAction);
gin.flush(Coop);
gin.waitSignal(Coop, signal, least);
gin.readSignal(signal);
```

**Key rules:**
- Pass `ncclDevComm` **by value** to kernels — do NOT cudaMalloc/cudaMemcpy it.
- Use literal integers for `<<<grid, block>>>` — no `NCCL_DEVICE_CTA_COUNT` macros.
- `__syncthreads()` after writes→puts, after flush→waitSignal, after wait→reads.
- `ncclCoopThread()` for single-thread put/flush; `ncclCoopCta()` for block-wide wait \
(CAUTION: ncclCoopCta can deadlock if not all CTA threads reach the wait).
""")

    if NCCL_DOCS:
        parts.append("### Full NCCL Device API Reference\n\n" + NCCL_DOCS)
    if NCCL_HEADERS:
        parts.append("### NCCL Device Headers (authoritative API)\n\n" + NCCL_HEADERS)

    parts.append(f"""\
### Step-by-step instructions
1. Read the source file at `{source_path}`
2. Copy it to `{work_dir}/{binary_name}.cu` and start transforming
3. After each stage, build and run using the commands above
4. If build or run fails, read the errors, fix, and retry
5. Once "Verification: PASS" appears and no host-side collectives remain,
   write the final code to `{output_file}`
""")

    return "\n".join(parts)


def _run_agent(source_path: Path, work_dir: Path, output_dir: Path,
               model: str, max_budget: float) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    prompt = _build_agent_prompt(source_path, work_dir)
    (work_dir / "_agent_prompt.txt").write_text(prompt, encoding="utf-8")

    cmd = [
        "claude", "-p",
        "--model", model,
        "--output-format", "text",
        "--dangerously-skip-permissions",
        "--no-session-persistence",
        "--allowedTools", "Bash,Read,Write,Edit",
        "--add-dir", str(work_dir),
        "--add-dir", str(source_path.parent),
        "--system-prompt", _AGENT_SYSTEM_PROMPT,
    ]
    if max_budget > 0:
        cmd.extend(["--max-budget-usd", str(max_budget)])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1"
    for p in ["/usr/bin", "/usr/local/bin", "/usr/sbin",
              "/usr/local/cuda-13.1/bin", os.path.expanduser("~/.local/bin")]:
        if p not in env.get("PATH", ""):
            env["PATH"] = p + ":" + env.get("PATH", "")

    logger.info(f"Agent mode: model={model}  budget=${max_budget}")
    start = time.perf_counter()
    try:
        result = subprocess.run(
            cmd, input=prompt, cwd=str(work_dir),
            capture_output=True, text=True, timeout=1800, env=env,
        )
    except subprocess.TimeoutExpired:
        logger.error("Agent timed out (30 min)"); return
    except Exception as exc:
        logger.error(f"Agent failed: {exc}"); return

    duration = time.perf_counter() - start
    agent_output = (result.stdout or "") + "\n" + (result.stderr or "")
    (work_dir / "_agent_output.txt").write_text(agent_output, encoding="utf-8")
    logger.info(f"Agent finished in {duration:.1f}s  exit={result.returncode}")
    print("\n--- Agent output (last 3000 chars) ---")
    print(agent_output[-3000:])
    print("--- end ---\n")

    binary_name = source_path.stem
    output_file = work_dir / f"{binary_name}_device.cu"
    working_file = work_dir / f"{binary_name}.cu"
    final_code = ""
    if output_file.exists():
        final_code = output_file.read_text(encoding="utf-8")
    elif working_file.exists():
        final_code = working_file.read_text(encoding="utf-8")
    if not final_code:
        logger.error("No output file produced by the agent"); return

    marker_model = f"claude-cli/{model}"
    marked_code = insert_evolve_markers(final_code, llm_model=marker_model)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{binary_name}_device.cu"
    out_path.write_text(marked_code, encoding="utf-8")
    print("=" * 70)
    print(f"Saved transformed code to: {out_path}")
    print("=" * 70)


# ===================================================================
# No-agent mode
# ===================================================================

def _run_no_agent(source_path: Path, work_dir: str, output_dir: str,
                  max_iterations: int) -> None:
    analyzer = CUDAAnalyzer(source_path)
    report = analyzer.analyze()
    print("\n" + "=" * 70)
    print(report.format_for_llm())
    print("=" * 70 + "\n")

    if not report.has_host_communication():
        logger.info("No host-side NCCL collectives found. Nothing to transform.")
        return

    hostfile = str(Path(__file__).resolve().parent / "build" / "hostfile")

    config = TransformConfig(
        rewrite_model="bedrock/us.anthropic.claude-opus-4-6-v1",
        judge_model="",
        max_iterations=max_iterations,
        reference_code="",
        nccl_api_docs=NCCL_API_FULL,
        num_mpi_ranks=2,
        cuda_visible_devices="0",
        hostfile=hostfile,
        mpirun_extra_args=(
            "--map-by", "node",
            "--mca", "btl_tcp_if_include", "enp75s0f1np1",
            "--mca", "oob_tcp_if_include", "enp75s0f1np1",
        ),
        run_env_vars={
            "NCCL_GIN_ENABLE": "1",
            "NCCL_SOCKET_IFNAME": "enp75s0f1np1",
            "NCCL_IB_HCA": "mlx5_1",
            "NCCL_IB_GID_INDEX": "3",
        },
    )

    transformer = HostToDeviceTransformer(config)
    result = transformer.transform(source_path, work_dir=work_dir)

    print("\n" + "=" * 70)
    print(f"TRANSFORMATION {'SUCCEEDED' if result.success else 'FAILED'}")
    print(f"Total iterations: {len(result.iterations)}")
    print(f"Total time: {result.total_duration_sec:.1f}s")
    for it in result.iterations:
        status = []
        if not it.build_success: status.append("BUILD_FAIL")
        elif not it.run_success: status.append("RUN_FAIL")
        elif not it.verification_passed: status.append("VERIFY_FAIL")
        elif it.has_host_communication: status.append("PARTIAL")
        else: status.append("SUCCESS")
        if it.time_ms is not None: status.append(f"time={it.time_ms:.2f}ms")
        print(f"  Iteration {it.iteration}: {', '.join(status)} ({it.duration_sec:.1f}s)")
    if result.error: print(f"Error: {result.error}")
    print("=" * 70 + "\n")

    out = Path(output_dir)
    transformer.save_result(result, out)
    logger.info(f"Results saved to: {out}")


# ===================================================================
# CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Transform host-side NCCL → device-side GIN.")
    parser.add_argument("--source", type=str, default=str(_EXAMPLE_DIR / "ds_v3_moe.cu"))
    parser.add_argument("--work_dir", type=str, default=str(_EXAMPLE_DIR / "_transform_work"))
    parser.add_argument("--output_dir", type=str, default=str(_EXAMPLE_DIR / "_transform_output"))
    parser.add_argument("--no-agent", dest="agent", action="store_false",
                        help="Use the structured rewrite→build→judge loop instead of the agent.")
    parser.set_defaults(agent=True)
    parser.add_argument("--model", type=str, default="sonnet", help="Claude model for agent mode.")
    parser.add_argument("--max_budget", type=float, default=10.0, help="Max USD for agent mode.")
    parser.add_argument("--max_iterations", type=int, default=5, help="Max iterations for no-agent mode.")
    args = parser.parse_args()

    source_path = Path(args.source).resolve()
    if not source_path.exists():
        logger.error(f"Source not found: {source_path}"); sys.exit(1)

    if args.agent:
        logger.info("Mode: AGENT")
        _run_agent(source_path, Path(args.work_dir).resolve(),
                   Path(args.output_dir).resolve(), args.model, args.max_budget)
    else:
        logger.info("Mode: NO-AGENT (structured loop)")
        _run_no_agent(source_path, args.work_dir, args.output_dir, args.max_iterations)


if __name__ == "__main__":
    main()
