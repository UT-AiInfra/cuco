"""CUDA C++ source compilation via nvcc.

GPU architecture is queried directly from the CUDA runtime
(cuda.bindings.runtime) so no nvidia-smi subprocess is needed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


# ── GPU arch detection ────────────────────────────────────────────────────

def _gpu_arch(device: int = 0) -> str:
    """Return the SM architecture string for *device* (e.g. ``"sm_90"``)."""
    try:
        from cuda.bindings import runtime as cudart
        err, props = cudart.cudaGetDeviceProperties(device)
        if err.value == 0:
            return f"sm_{props.major}{props.minor}"
    except Exception:
        pass
    # Fallback: nvidia-smi (no CUDA runtime available)
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0 and r.stdout.strip():
            major, _, minor = r.stdout.splitlines()[0].strip().partition(".")
            if major.isdigit():
                return f"sm_{major}{minor or '0'}"
    except Exception:
        pass
    return "sm_80"


# ── nvcc discovery ────────────────────────────────────────────────────────

def _find_nvcc() -> str:
    if nvcc := os.environ.get("NVCC"):
        return nvcc
    if nvcc := shutil.which("nvcc"):
        return nvcc
    for p in sorted(Path("/usr/local").glob("cuda-*/bin/nvcc"), reverse=True):
        if p.exists():
            return str(p)
    if Path("/usr/local/cuda/bin/nvcc").exists():
        return "/usr/local/cuda/bin/nvcc"
    if cuda_home := os.environ.get("CUDA_HOME"):
        candidate = os.path.join(cuda_home, "bin", "nvcc")
        if os.path.exists(candidate):
            return candidate
    raise RuntimeError("nvcc not found; set CUDA_HOME or add nvcc to PATH")


# ── include discovery ─────────────────────────────────────────────────────

def _nccl_include() -> list[str]:
    try:
        import nvidia.nccl as _nccl_pkg
        return [os.path.join(list(_nccl_pkg.__path__)[0], "include")]
    except ImportError:
        return []


# ── compilation options ───────────────────────────────────────────────────

@dataclass
class CompileOptions:
    """nvcc compilation options, mirroring NVRTC's ProgramOptions.

    Example::

        opts = CompileOptions(std="c++17", use_fast_math=True, defines={"DEBUG": "1"})
        cubin = compile_cuda(source, options=opts)
    """
    arch: str | None = None
    """GPU architecture, e.g. ``"sm_90"``. Auto-detected if ``None``."""

    nvcc: str | None = None
    """Path to the nvcc binary. Auto-detected if ``None``."""

    includes: list[str] = field(default_factory=list)
    """Additional ``-I`` directories."""

    std: str | None = None
    """C++ standard passed to ``--std``, e.g. ``"c++17"``."""

    defines: dict[str, str | None] = field(default_factory=dict)
    """Preprocessor macros: ``{"NAME": "VALUE"}`` or ``{"FLAG": None}``."""

    device_debug: bool = False
    """Embed device-side debug info (``-G``)."""

    lineinfo: bool = False
    """Embed source line-number info (``-lineinfo``)."""

    use_fast_math: bool = False
    """Enable fast-math optimisations (``--use_fast_math``)."""

    maxrregcount: int | None = None
    """Maximum registers per thread (``--maxrregcount``)."""

    relocatable_device_code: bool = False
    """Enable separate compilation / RDC (``--relocatable-device-code=true``)."""

    extra_flags: list[str] = field(default_factory=list)
    """Raw flags appended verbatim to the nvcc command line."""

    def _to_flags(self, gpu_arch: str) -> list[str]:
        flags = [f"-arch={gpu_arch}"]
        if self.std:
            flags.append(f"--std={self.std}")
        for name, val in self.defines.items():
            flags.append(f"-D{name}" if val is None else f"-D{name}={val}")
        if self.device_debug:
            flags.append("-G")
        if self.lineinfo:
            flags.append("-lineinfo")
        if self.use_fast_math:
            flags.append("--use_fast_math")
        if self.maxrregcount is not None:
            flags.append(f"--maxrregcount={self.maxrregcount}")
        if self.relocatable_device_code:
            flags.append("--relocatable-device-code=true")
        flags.extend(self.extra_flags)
        return flags


# ── public API ────────────────────────────────────────────────────────────

def compile_cuda(source: str, options: CompileOptions | None = None) -> bytes:
    """Compile a CUDA C++ source string to a cubin and return its bytes.

    GPU architecture is resolved from the CUDA runtime; nvidia-smi is used
    only as a fallback. The nvidia.nccl include directory is added
    automatically when available.

    Args:
        source: CUDA C++ source code.
        options: Compilation options.  Uses defaults when ``None``.

    Returns:
        Raw cubin bytes.

    Raises:
        RuntimeError: If nvcc is not found or compilation fails.
    """
    opts = options or CompileOptions()
    gpu_arch = opts.arch or _gpu_arch()
    nvcc_bin = opts.nvcc or _find_nvcc()
    # Preserve order while dropping duplicate include directories.
    includes = list(dict.fromkeys([*_nccl_include(), *opts.includes]))

    with tempfile.TemporaryDirectory() as tmpdir:
        src = Path(tmpdir) / "kernel.cu"
        out = Path(tmpdir) / "kernel.cubin"
        src.write_text(source)

        cmd = [nvcc_bin]
        for inc in includes:
            cmd.extend(["-I", inc])
        cmd.extend(opts._to_flags(gpu_arch))
        cmd.extend(["-cubin", str(src), "-o", str(out)])

        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            log = (r.stderr + r.stdout).strip()
            raise RuntimeError(f"nvcc compilation failed (arch={gpu_arch}):\n{log}")

        return out.read_bytes()
