#!/usr/bin/env python3
"""Naive Prompting ablation for NeurIPS paper.

End-to-end comparison: the LLM starts from the same host-side CUDA+NCCL code
that CUCo starts from. It must convert host-side NCCL to device-side GIN AND
optimize — all via iterative prompting with evaluation feedback.

CUCo's full workflow:
  1. Pre-transform pipeline (analyze → host_to_device → evolve_markers → warmup)
  2. Evolutionary search (population, crossover, islands, meta-recommendations,
     explore/exploit phases, directives)

This ablation replaces BOTH with a simple loop:
  Show LLM the current code + performance feedback → ask for improvement
  → evaluate → repeat.

The LLM gets the SAME knowledge (GIN API, strategies, hardware context,
NCCL docs/headers, GIN-specific rules) — the only difference is the search
and transformation strategy: naive iterative prompting vs CUCo's structured
pipeline + evolutionary search.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sys
import time
from pathlib import Path

_EXAMPLE_DIR = Path(__file__).resolve().parent
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))

_CUDA_EVOLVE_DIR = Path(__file__).resolve().parent.parent.parent
if str(_CUDA_EVOLVE_DIR) not in sys.path:
    sys.path.insert(0, str(_CUDA_EVOLVE_DIR))

try:
    from dotenv import load_dotenv
    load_dotenv(_CUDA_EVOLVE_DIR / ".env")
except ImportError:
    pass

from cuco.llm.query import query
import evaluate
from evaluate import main as evaluate_main, get_hardware_context

import run_evo
from run_transform import NCCL_DOCS as _TRANSFORM_NCCL_DOCS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Same timeout as CUCo's evaluator (fair comparison)
evaluate.RUN_TIMEOUT = 120

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LLM_MODEL = "bedrock/us.anthropic.claude-sonnet-4-6"
MAX_TOKENS = 32768
NUM_GENERATIONS = 20
INIT_PROGRAM = "ds_v3_moe.cu"
SOURCE_NAME = "ds_v3_moe.cu"
RESULTS_DIR = "results_naive_prompting"
API_TYPE = "gin"
TEMPERATURE = 0.2

# ---------------------------------------------------------------------------
# Response format
# ---------------------------------------------------------------------------
_RESPONSE_FORMAT = """
## Response Format

Provide the complete rewritten program. You MUST respond using:

<NAME>
A shortened name summarizing the code you are proposing. Lowercase, no spaces, underscores allowed.
</NAME>

<DESCRIPTION>
A description and argumentation process of the code you are proposing.
</DESCRIPTION>

<CODE>
```cuda
// The complete rewritten program here.
```
</CODE>

* You have FULL autonomy to rewrite the entire program. You may restructure
  everything: replace host-side NCCL calls with device-side GIN, add new
  kernels, change memory allocation patterns, add streams, etc.
* Keep "// EVOLVE-BLOCK-START" and "// EVOLVE-BLOCK-END" markers if they exist.
* The program must compile with nvcc and run with mpirun -np 2.
* The program must print "Verification: PASS" and "Time: X.XXXX ms".
* Use the <NAME>, <DESCRIPTION>, and <CODE> delimiters. It will be parsed."""

# ---------------------------------------------------------------------------
# User message templates
# ---------------------------------------------------------------------------

# First generation: the LLM sees the host code and must convert to device-side GIN
_FIRST_GEN_MSG = """# Current program (host-side NCCL)

This program uses HOST-SIDE ncclAlltoAll calls. Your goal is to transform it
to use DEVICE-SIDE GIN (GPU Initiated Networking) for communication, which
enables compute-communication overlap and dramatically reduces runtime.

```cuda
{code_content}
```

## Current performance

{performance_info}

{feedback_section}

# Task

Transform this program from host-side NCCL to device-side GIN:
1. Replace `cudaMalloc` for communicated buffers with `ncclMemAlloc`
2. Add `ncclCommWindowRegister` for all communicated buffers
3. Create `ncclDevComm` with `ncclDevCommCreate`; clear stale error with
   `(void)cudaGetLastError()` immediately after.
4. Replace host-side `ncclAlltoAll` calls with GIN kernel(s) using
   `gin.put` / `gin.flush` / `gin.waitSignal`
5. Add communication warmup to eliminate cold-start latency
6. Overlap compute and communication where possible

The program must still print "Verification: PASS" and minimize "Time: X.XXXX ms".
Provide the COMPLETE rewritten program.
"""

# Subsequent generations: iterative optimization
_ITER_MSG = """# Current program

Here is the current program (you will propose an improved version with the
same inputs and outputs but better performance):

```cuda
{code_content}
```

## Performance metrics

{performance_info}

{feedback_section}

# Task

Rewrite the program to improve its performance. Minimize the "Time: X.XXXX ms"
metric while keeping "Verification: PASS". Provide the complete rewritten
program.
"""

# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------

def extract_code_from_response(response: str) -> str | None:
    """Extract code from <CODE>...</CODE> tags in LLM response."""
    match = re.search(r"<CODE>\s*```(?:cuda|cpp|c\+\+|c)?\s*\n(.*?)```\s*</CODE>", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"<CODE>\s*(.*?)\s*</CODE>", response, re.DOTALL)
    if match:
        code = match.group(1).strip()
        fenced = re.search(r"```(?:cuda|cpp|c\+\+|c)?\s*\n(.*?)```", code, re.DOTALL)
        if fenced:
            return fenced.group(1).strip()
        return code
    match = re.search(r"```(?:cuda|cpp|c\+\+|c)?\s*\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def extract_name_from_response(response: str) -> str:
    match = re.search(r"<NAME>\s*(.*?)\s*</NAME>", response, re.DOTALL)
    if match:
        return match.group(1).strip().replace(" ", "_")[:50]
    return "unnamed"


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    results_path = Path(RESULTS_DIR)
    results_path.mkdir(parents=True, exist_ok=True)

    init_program_path = Path(INIT_PROGRAM)
    if not init_program_path.exists():
        logger.error(f"Initial program not found: {INIT_PROGRAM}")
        sys.exit(1)

    current_code = init_program_path.read_text(encoding="utf-8")
    best_score = 0.0
    best_time_ms = float("inf")
    best_code = current_code
    best_gen = -1

    # Build system message with the SAME knowledge CUCo gets — including GIN
    # knowledge. The ablation is about search strategy, not knowledge access.
    header = (
        "You are an expert in CUDA and NCCL GIN (GPU Integrated Networking) kernels.\n"
        "Your goal: transform a host-side CUDA+NCCL program to use device-side GIN "
        "communication, and then optimize it to minimize kernel runtime "
        "(Time in ms printed by the program).\n\n"
        "The program may start with host-side ncclAlltoAll calls — these must be "
        "replaced with device-side GIN (gin.put / gin.flush / gin.waitSignal) to "
        "enable compute-communication overlap.\n\n"
        "Analyze the current program structure, identify bottlenecks "
        "(sequential execution, missing overlap, cold-start penalties, host-side "
        "collectives blocking the GPU pipeline, etc.), "
        "and apply the optimization strategies described below. "
        "You have full autonomy over the approach.\n"
    )
    system_msg = (
        header
        + run_evo._COMMON_CONSTRAINTS + "\n"
        + run_evo._STRATEGIES + "\n"
        + run_evo._GIN_KNOWLEDGE + "\n"
        + run_evo._HARDWARE_CONTEXT + "\n"
        + run_evo._PROFILING_INFO
    )
    # Give the same NCCL docs the fast-path agent gets (includes HOST_TO_DEVICE_COOKBOOK)
    if _TRANSFORM_NCCL_DOCS:
        system_msg += "\n\n---\n\n## NCCL GIN API Reference (with Host-to-Device Cookbook)\n\n" + _TRANSFORM_NCCL_DOCS
    if run_evo._NCCL_GIN_HEADERS:
        system_msg += "\n\n---\n\n## NCCL GIN Header Files\n\n" + run_evo._NCCL_GIN_HEADERS
    system_msg += _RESPONSE_FORMAT

    history = []
    start_gen = 1
    score = 0.0
    time_ms = None
    feedback = ""

    # Resume from existing history if available
    history_file = results_path / "history.json"
    if history_file.exists():
        history = json.loads(history_file.read_text())
        if history:
            last_gen = history[-1]["gen"]
            start_gen = last_gen + 1
            for entry in history:
                s = entry.get("score", 0.0)
                t = entry.get("time_ms")
                if s > best_score:
                    best_score = s
                    best_time_ms = t if t else float("inf")
                    best_gen = entry["gen"]
            # Recover current code from last generation's source file
            last_gen_dir = results_path / f"gen_{last_gen}"
            last_source = last_gen_dir / SOURCE_NAME
            if last_source.exists():
                current_code = last_source.read_text(encoding="utf-8")
            # Recover last feedback
            last_metrics = last_gen_dir / "results" / "metrics.json"
            if last_metrics.exists():
                m = json.loads(last_metrics.read_text())
                score = m.get("combined_score", 0.0)
                time_ms = m.get("public", {}).get("time_ms", None)
                feedback = m.get("text_feedback", "")
            # Recover best code
            if best_gen >= 0:
                best_source = results_path / f"gen_{best_gen}" / SOURCE_NAME
                if best_source.exists():
                    best_code = best_source.read_text(encoding="utf-8")
            logger.info(f"RESUMING from generation {start_gen} (last completed: gen_{last_gen})")

    logger.info("=" * 60)
    logger.info("NAIVE PROMPTING ABLATION")
    logger.info(f"  Model: {LLM_MODEL}")
    logger.info(f"  Generations: {NUM_GENERATIONS}")
    logger.info(f"  Temperature: {TEMPERATURE}")
    logger.info(f"  Init program: {INIT_PROGRAM}")
    logger.info(f"  Results dir: {RESULTS_DIR}")
    logger.info(f"  System msg length: {len(system_msg)} chars")
    logger.info("=" * 60)

    # Only run gen_0 evaluation if starting fresh
    if start_gen == 1 and not history:
        gen_dir = results_path / "gen_0"
        gen_dir.mkdir(parents=True, exist_ok=True)
        source_path = gen_dir / SOURCE_NAME
        source_path.write_text(current_code, encoding="utf-8")

        main_path = gen_dir / "main.cu"
        main_path.write_text(current_code, encoding="utf-8")

        results_subdir = gen_dir / "results"
        results_subdir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\n--- Generation 0 (baseline) ---")
        evaluate_main(str(main_path), str(results_subdir))

        metrics_file = results_subdir / "metrics.json"
        if metrics_file.exists():
            metrics = json.loads(metrics_file.read_text())
            score = metrics.get("combined_score", 0.0)
            time_ms = metrics.get("public", {}).get("time_ms", None)
            feedback = metrics.get("text_feedback", "")
            best_score = score
            if time_ms:
                best_time_ms = time_ms
            best_code = current_code
            best_gen = 0
            logger.info(f"  Score: {score:.2f}, Time: {time_ms} ms")
        else:
            feedback = "Initial evaluation failed."
            score = 0.0
            time_ms = None
            logger.warning("  Initial evaluation produced no metrics.")

        history.append({
            "gen": 0,
            "score": score,
            "time_ms": time_ms,
            "name": "baseline",
        })

    # Track whether we've successfully converted to GIN yet
    has_gin = "nccl_device.h" in current_code

    # Iterative improvement loop
    for gen in range(start_gen, NUM_GENERATIONS + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Generation {gen}/{NUM_GENERATIONS}")
        logger.info(f"{'='*60}")

        # Build performance info string
        if time_ms is not None:
            perf_info = f"Combined score: {score:.2f}\nTime: {time_ms:.4f} ms (lower is better)"
        else:
            perf_info = f"Combined score: {score:.2f}\nProgram failed to run correctly or timed out."

        if best_time_ms < float("inf"):
            perf_info += f"\nBest so far: {best_time_ms:.4f} ms (gen {best_gen}, score {best_score:.2f})"

        # Build feedback section
        feedback_section = ""
        if feedback:
            feedback_section = f"\n# Evaluation Feedback\n\n{feedback}"

        # Use the first-gen template if we haven't converted to GIN yet
        if not has_gin:
            user_msg = _FIRST_GEN_MSG.format(
                code_content=current_code,
                performance_info=perf_info,
                feedback_section=feedback_section,
            )
        else:
            user_msg = _ITER_MSG.format(
                code_content=current_code,
                performance_info=perf_info,
                feedback_section=feedback_section,
            )

        # Query LLM
        logger.info(f"  Querying {LLM_MODEL} (temp={TEMPERATURE})...")
        try:
            result = query(
                model_name=LLM_MODEL,
                msg=user_msg,
                system_msg=system_msg,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            response_text = result.content
        except Exception as e:
            logger.error(f"  LLM query failed: {e}")
            history.append({"gen": gen, "score": 0.0, "time_ms": None, "name": "llm_error", "error": str(e)})
            continue

        # Extract code
        new_code = extract_code_from_response(response_text)
        name = extract_name_from_response(response_text)

        if new_code is None:
            logger.warning(f"  Failed to extract code from LLM response")
            history.append({"gen": gen, "score": 0.0, "time_ms": None, "name": "parse_error"})
            continue

        if len(new_code) < 500:
            logger.warning(f"  Extracted code too short ({len(new_code)} chars), skipping")
            history.append({"gen": gen, "score": 0.0, "time_ms": None, "name": "too_short"})
            continue

        # Save and evaluate
        gen_dir = results_path / f"gen_{gen}"
        gen_dir.mkdir(parents=True, exist_ok=True)

        # Save LLM response for debugging
        (gen_dir / "rewrite.txt").write_text(response_text, encoding="utf-8")

        source_path = gen_dir / SOURCE_NAME
        source_path.write_text(new_code, encoding="utf-8")
        main_path = gen_dir / "main.cu"
        main_path.write_text(new_code, encoding="utf-8")

        results_subdir = gen_dir / "results"
        results_subdir.mkdir(parents=True, exist_ok=True)

        logger.info(f"  Evaluating '{name}'...")
        evaluate_main(str(main_path), str(results_subdir))

        metrics_file = results_subdir / "metrics.json"
        if metrics_file.exists():
            metrics = json.loads(metrics_file.read_text())
            new_score = metrics.get("combined_score", 0.0)
            new_time_ms = metrics.get("public", {}).get("time_ms", None)
            new_feedback = metrics.get("text_feedback", "")
        else:
            new_score = 0.0
            new_time_ms = None
            new_feedback = "Evaluation produced no metrics."

        # Update state: always use the latest attempt as the next input
        # (even if it regressed — the LLM sees the feedback and can correct)
        if new_score > 0:
            current_code = new_code
            score = new_score
            time_ms = new_time_ms
            feedback = new_feedback
            has_gin = "nccl_device.h" in current_code

            if new_score > best_score:
                best_score = new_score
                best_time_ms = new_time_ms
                best_code = new_code
                best_gen = gen
                logger.info(f"  NEW BEST! Score: {new_score:.2f}, Time: {new_time_ms:.4f} ms")
            else:
                logger.info(f"  Score: {new_score:.2f}, Time: {new_time_ms} ms (best: {best_time_ms:.4f} ms)")
        else:
            # Failed — but if it attempted GIN, use it as new current code
            # (so feedback helps it fix the GIN conversion errors)
            if "nccl_device.h" in new_code:
                current_code = new_code
                has_gin = True
            feedback = new_feedback
            logger.info(f"  FAILED (score=0). Keeping {'new GIN attempt' if 'nccl_device.h' in new_code else 'previous code'}.")

        history.append({
            "gen": gen,
            "score": new_score,
            "time_ms": new_time_ms,
            "name": name,
            "has_gin": "nccl_device.h" in new_code,
        })

        # Save running history
        (results_path / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )

    # Save best
    best_dir = results_path / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    (best_dir / SOURCE_NAME).write_text(best_code, encoding="utf-8")
    (best_dir / "main.cu").write_text(best_code, encoding="utf-8")

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("NAIVE PROMPTING ABLATION COMPLETE")
    logger.info(f"  Best score: {best_score:.2f}")
    logger.info(f"  Best time: {best_time_ms:.4f} ms")
    logger.info(f"  Best generation: {best_gen}")
    logger.info(f"  Total generations: {NUM_GENERATIONS}")
    logger.info("=" * 60)

    # Save final summary
    summary = {
        "ablation": "naive_prompting",
        "model": LLM_MODEL,
        "temperature": TEMPERATURE,
        "num_generations": NUM_GENERATIONS,
        "best_score": best_score,
        "best_time_ms": best_time_ms,
        "best_generation": best_gen,
        "history": history,
    }
    (results_path / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
