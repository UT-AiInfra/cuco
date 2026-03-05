"""Pre-transform pipeline: ordered, conditional steps to prepare a seed for evolution.

Steps (run in order, each conditional):
    1. analyze         — CUDAAnalyzer (Python/regex), always runs
    2. host_to_device  — HostToDeviceTransformer (LLM), skipped if no host NCCL
    3. evolve_markers  — insert_evolve_markers (LLM + regex fallback), skipped if present
    4. warmup          — LLM injection + build/verify, skipped if warmup present
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
from .transformer import (
    HostToDeviceTransformer,
    TransformConfig,
    TransformResult,
    insert_evolve_markers,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default warmup system prompt (generic, works for GIN/LSA/Host NCCL)
# ---------------------------------------------------------------------------

_DEFAULT_WARMUP_PROMPT = """\
You are a CUDA expert. Add communication warmup to the given CUDA program.

The first GIN/LSA/NCCL communication call in a process triggers lazy RDMA/NIC
initialization that adds 10-50ms to timing. Your job: add warmup rounds BEFORE
the timing section to amortize this cost.

Rules:
1. Find the timing start (typically cudaEventRecord with the start event).
2. Add 2 rounds of dummy communication calls IMMEDIATELY BEFORE that line.
3. Use the SAME communication kernels/functions and buffers that appear in the
   timed section. For GIN: call the same put/wait kernels with the same params.
   For cooperative kernels: launch the kernel once as warmup. For host NCCL:
   call the same ncclAlltoAll/ncclSend/ncclRecv.
4. Also add one dummy launch of each compute kernel type to prime instruction caches.
5. After all warmup rounds, add cudaDeviceSynchronize() and MPI_Barrier(MPI_COMM_WORLD).
6. Return the COMPLETE modified .cu file — every single line, nothing omitted.
7. Do NOT change anything else. Only add the warmup section.
8. Do NOT wrap the output in markdown code fences. Return raw C/CUDA code only."""


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PipelineStepResult:
    """Outcome of a single pipeline step."""
    name: str
    skipped: bool
    method: str = ""          # "regex", "llm", "python", "llm+regex_fallback"
    duration_sec: float = 0.0
    error: str = ""
    cost: float = 0.0
    details: str = ""         # e.g. analysis summary, marker count


@dataclass
class PipelineResult:
    """Outcome of the full pipeline run."""
    final_code: str
    final_path: Optional[str] = None
    step_results: List[PipelineStepResult] = field(default_factory=list)
    total_cost: float = 0.0
    total_duration_sec: float = 0.0
    any_step_failed: bool = False


@dataclass
class _PipelineContext:
    """Shared mutable state passed between steps."""
    analysis: Optional[AnalysisReport] = None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class PreTransformPipeline:
    """Configurable, ordered pre-transform pipeline.

    Each step checks whether it's needed (regex/Python) and applies its
    transformation (LLM or regex) only when necessary.  Steps share a
    context object so later steps can use information from earlier ones
    (e.g. the analysis report).
    """

    _STEP_REGISTRY = {
        "analyze",
        "host_to_device",
        "evolve_markers",
        "warmup",
    }

    def __init__(
        self,
        config: TransformConfig,
        steps: List[str],
        *,
        warmup_model: Optional[str] = None,
        warmup_prompt: str = "",
        marker_model: Optional[str] = None,
    ):
        for s in steps:
            if s not in self._STEP_REGISTRY:
                raise ValueError(
                    f"Unknown pipeline step {s!r}. "
                    f"Valid steps: {sorted(self._STEP_REGISTRY)}"
                )
        self.config = config
        self.steps = steps
        self.warmup_model = warmup_model or config.rewrite_model
        self.warmup_prompt = warmup_prompt or _DEFAULT_WARMUP_PROMPT
        self.marker_model = marker_model or config.rewrite_model
        self._accumulated_cost: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        source_path: Path,
        output_dir: Path,
        *,
        work_dir: Optional[Path] = None,
    ) -> PipelineResult:
        """Run every configured step in order.

        Args:
            source_path: Path to the seed .cu file.
            output_dir:  Directory for pipeline outputs and per-step artifacts.
            work_dir:    Working directory for builds (defaults to output_dir/_work).

        Returns:
            PipelineResult with final code and per-step details.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        if work_dir is None:
            work_dir = output_dir / "_work"
        work_dir.mkdir(parents=True, exist_ok=True)

        source_path = Path(source_path).resolve()
        code = source_path.read_text(encoding="utf-8")
        ctx = _PipelineContext()
        step_results: List[PipelineStepResult] = []
        pipeline_start = time.perf_counter()
        any_failed = False

        dispatch = {
            "analyze": self._step_analyze,
            "host_to_device": self._step_host_to_device,
            "evolve_markers": self._step_evolve_markers,
            "warmup": self._step_warmup,
        }

        for step_name in self.steps:
            logger.info(f"Pipeline step: {step_name}")
            step_fn = dispatch[step_name]
            step_start = time.perf_counter()

            try:
                new_code, sr = step_fn(code, source_path, ctx, output_dir, work_dir)
            except Exception as exc:
                sr = PipelineStepResult(
                    name=step_name, skipped=False,
                    error=f"Unhandled exception: {exc}",
                )
                new_code = code
                any_failed = True
                logger.error(f"Pipeline step {step_name} raised: {exc}", exc_info=True)

            sr.duration_sec = time.perf_counter() - step_start
            step_results.append(sr)
            self._accumulated_cost += sr.cost

            if sr.error:
                any_failed = True
                logger.warning(
                    f"  Step {step_name} had error (using previous code): {sr.error[:200]}"
                )
            elif sr.skipped:
                logger.info(f"  Step {step_name} skipped (already satisfied)")
            else:
                code = new_code
                logger.info(
                    f"  Step {step_name} applied ({sr.method}, "
                    f"{sr.duration_sec:.1f}s, ${sr.cost:.4f})"
                )

            # Save intermediate code for debuggability
            step_dir = output_dir / f"_step_{step_name}"
            step_dir.mkdir(parents=True, exist_ok=True)
            (step_dir / source_path.name).write_text(code, encoding="utf-8")
            (step_dir / "result.json").write_text(
                json.dumps({
                    "name": sr.name,
                    "skipped": sr.skipped,
                    "method": sr.method,
                    "duration_sec": round(sr.duration_sec, 2),
                    "cost": round(sr.cost, 4),
                    "error": sr.error,
                    "details": sr.details,
                }, indent=2),
                encoding="utf-8",
            )

        # Write final output
        final_path = output_dir / source_path.name
        final_path.write_text(code, encoding="utf-8")

        total_dur = time.perf_counter() - pipeline_start
        logger.info(
            f"Pipeline finished: {len(step_results)} steps, "
            f"{total_dur:.1f}s, ${self._accumulated_cost:.4f}"
        )

        return PipelineResult(
            final_code=code,
            final_path=str(final_path),
            step_results=step_results,
            total_cost=self._accumulated_cost,
            total_duration_sec=total_dur,
            any_step_failed=any_failed,
        )

    # ------------------------------------------------------------------
    # Step 1: Analyze
    # ------------------------------------------------------------------

    def _step_analyze(
        self,
        code: str,
        source_path: Path,
        ctx: _PipelineContext,
        output_dir: Path,
        work_dir: Path,
    ) -> Tuple[str, PipelineStepResult]:
        """Always runs.  Uses CUDAAnalyzer (pure Python/regex)."""
        analyzer = CUDAAnalyzer(source_path)
        report = analyzer.analyze()
        ctx.analysis = report

        has_host = report.has_host_communication()
        collective_names = [c.name for c in report.nccl_collectives] if has_host else []
        details = (
            f"host_collectives={collective_names}, "
            f"kernels={len(report.kernel_definitions)}, "
            f"allocs={len(report.memory_allocations)}"
        )

        # Save the full LLM-formatted analysis for debugging
        step_dir = output_dir / "_step_analyze"
        step_dir.mkdir(parents=True, exist_ok=True)
        (step_dir / "analysis.txt").write_text(
            report.format_for_llm(), encoding="utf-8"
        )

        return code, PipelineStepResult(
            name="analyze",
            skipped=False,
            method="python",
            details=details,
        )

    # ------------------------------------------------------------------
    # Step 2: Host-to-Device
    # ------------------------------------------------------------------

    def _step_host_to_device(
        self,
        code: str,
        source_path: Path,
        ctx: _PipelineContext,
        output_dir: Path,
        work_dir: Path,
    ) -> Tuple[str, PipelineStepResult]:
        """Skipped if no host-side NCCL collectives.  Uses HostToDeviceTransformer (LLM)."""
        if ctx.analysis and not ctx.analysis.has_host_communication():
            return code, PipelineStepResult(
                name="host_to_device", skipped=True,
                details="No host-side NCCL collectives found",
            )

        h2d_work = work_dir / "_host_to_device"
        h2d_work.mkdir(parents=True, exist_ok=True)
        h2d_output = output_dir / "_step_host_to_device"
        h2d_output.mkdir(parents=True, exist_ok=True)

        # Write current code to a temp file for the transformer
        temp_source = h2d_work / source_path.name
        temp_source.write_text(code, encoding="utf-8")

        transformer = HostToDeviceTransformer(self.config)
        result: TransformResult = transformer.transform(
            temp_source, work_dir=h2d_work
        )
        transformer.save_result(result, h2d_output)

        if result.success:
            return result.final_code, PipelineStepResult(
                name="host_to_device",
                skipped=False,
                method="llm",
                cost=result.total_cost,
                details=f"Succeeded in {len(result.iterations)} iteration(s)",
            )
        else:
            return code, PipelineStepResult(
                name="host_to_device",
                skipped=False,
                method="llm",
                cost=result.total_cost,
                error=(
                    f"Failed after {len(result.iterations)} iteration(s): "
                    f"{(result.error or 'unknown')[:200]}"
                ),
            )

    # ------------------------------------------------------------------
    # Step 3: Evolve Markers
    # ------------------------------------------------------------------

    def _step_evolve_markers(
        self,
        code: str,
        source_path: Path,
        ctx: _PipelineContext,
        output_dir: Path,
        work_dir: Path,
    ) -> Tuple[str, PipelineStepResult]:
        """Skipped if EVOLVE-BLOCK-START already present.  Uses LLM with regex fallback."""
        if "EVOLVE-BLOCK-START" in code:
            return code, PipelineStepResult(
                name="evolve_markers", skipped=True,
                details="EVOLVE-BLOCK markers already present",
            )

        marked = insert_evolve_markers(code, llm_model=self.marker_model)
        if "EVOLVE-BLOCK-START" in marked:
            n_blocks = marked.count("EVOLVE-BLOCK-START")
            return marked, PipelineStepResult(
                name="evolve_markers",
                skipped=False,
                method="llm+regex_fallback",
                details=f"Inserted {n_blocks} EVOLVE-BLOCK region(s)",
            )
        else:
            return code, PipelineStepResult(
                name="evolve_markers",
                skipped=False,
                method="llm+regex_fallback",
                error="Failed to insert EVOLVE-BLOCK markers (LLM and regex both failed)",
            )

    # ------------------------------------------------------------------
    # Step 4: Warmup
    # ------------------------------------------------------------------

    def _step_warmup(
        self,
        code: str,
        source_path: Path,
        ctx: _PipelineContext,
        output_dir: Path,
        work_dir: Path,
    ) -> Tuple[str, PipelineStepResult]:
        """Skipped if warmup already present.  Uses LLM injection + build/verify."""
        if re.search(r"warmup|warm.?up", code, re.IGNORECASE):
            return code, PipelineStepResult(
                name="warmup", skipped=True,
                details="Warmup already present in code",
            )

        modified = self._call_warmup_llm(code)
        if not modified or not re.search(r"warmup|warm.?up", modified, re.IGNORECASE):
            return code, PipelineStepResult(
                name="warmup",
                skipped=False,
                method="llm",
                error="LLM did not produce warmup code",
            )

        warmup_work = work_dir / "_warmup"
        warmup_work.mkdir(parents=True, exist_ok=True)

        ok, build_err = self._build(modified, warmup_work)
        if not ok:
            return code, PipelineStepResult(
                name="warmup",
                skipped=False,
                method="llm",
                error=f"Build failed after warmup injection: {build_err[:200]}",
            )

        run_ok, run_output = self._run(warmup_work)
        if not run_ok or self.config.verification_pass_str not in run_output:
            return code, PipelineStepResult(
                name="warmup",
                skipped=False,
                method="llm",
                error=f"Verification failed after warmup injection",
            )

        return modified, PipelineStepResult(
            name="warmup",
            skipped=False,
            method="llm",
            details="Warmup injected, built, and verified successfully",
        )

    # ------------------------------------------------------------------
    # LLM call for warmup
    # ------------------------------------------------------------------

    def _call_warmup_llm(self, code: str) -> str:
        """Single LLM call to inject warmup. Returns modified code or empty."""
        try:
            import anthropic
        except ImportError:
            logger.error("anthropic package required for warmup injection")
            return ""

        model_name = self.warmup_model
        try:
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

            logger.info(f"Warmup LLM call: {actual_model}")
            with client.messages.stream(
                model=actual_model,
                system=self.warmup_prompt,
                messages=[{"role": "user", "content": code}],
                max_tokens=32768,
                temperature=0.0,
            ) as stream:
                response = stream.get_final_message()

            if response.usage:
                from ..llm.models.pricing import CLAUDE_MODELS, BEDROCK_MODELS
                pricing = (
                    BEDROCK_MODELS.get(model_name)
                    or CLAUDE_MODELS.get(actual_model)
                    or {}
                )
                in_cost = pricing.get("input_price", 0) * response.usage.input_tokens
                out_cost = pricing.get("output_price", 0) * response.usage.output_tokens
                self._accumulated_cost += in_cost + out_cost

            if response.content and len(response.content) > 0:
                result = response.content[0].text.strip()
                if result.startswith("```"):
                    lines = result.split("\n")
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].strip() == "```":
                        lines = lines[:-1]
                    result = "\n".join(lines)
                return result
            return ""
        except Exception as exc:
            logger.warning(f"Warmup LLM call failed: {exc}")
            return ""

    # ------------------------------------------------------------------
    # Build & Run (reuses TransformConfig settings)
    # ------------------------------------------------------------------

    def _build(self, code: str, work_dir: Path) -> Tuple[bool, str]:
        """Compile code using nvcc with TransformConfig paths."""
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
                cmd, cwd=str(work_dir),
                capture_output=True, text=True, timeout=60, env=env,
            )
            if result.returncode != 0:
                return False, (result.stderr or result.stdout or "")[:4000]
            return True, ""
        except subprocess.TimeoutExpired:
            return False, "Build timed out (60s)"
        except Exception as exc:
            return False, str(exc)

    def _run(self, work_dir: Path) -> Tuple[bool, str]:
        """Run the compiled binary with mpirun."""
        binary = work_dir / self.config.binary_name
        if not binary.exists():
            return False, f"Binary not found: {binary}"

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

        cmd = ["mpirun", "-np", str(self.config.num_mpi_ranks), str(binary)]

        try:
            result = subprocess.run(
                cmd, cwd=str(work_dir), env=env,
                capture_output=True, text=True, timeout=self.config.run_timeout,
            )
            output = (result.stdout or "") + "\n" + (result.stderr or "")
            if result.returncode != 0:
                return False, output[:4000]
            return True, output[:4000]
        except subprocess.TimeoutExpired:
            return False, f"Run timed out ({self.config.run_timeout}s)"
        except Exception as exc:
            return False, str(exc)
