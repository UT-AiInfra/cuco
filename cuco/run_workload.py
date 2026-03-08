"""
Runner logic for ``cuco_run``.

Resolves a workload directory by name and launches run_evo.py with the
appropriate arguments.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def find_workload_dir(name: str) -> Path:
    """Resolve a workload name to its directory under workloads/.

    Tries: workloads/<name>, then treats *name* as a direct path.
    """
    workloads_root = _PROJECT_ROOT / "workloads"
    candidate = workloads_root / name
    if candidate.is_dir():
        return candidate

    # Maybe user passed a direct path
    p = Path(name)
    if p.is_dir():
        return p.resolve()

    raise FileNotFoundError(
        f"Workload '{name}' not found. Looked in:\n"
        f"  - {candidate}\n"
        f"  - {p.resolve()}\n"
        f"\nAvailable workloads: {', '.join(d.name for d in workloads_root.iterdir() if d.is_dir()) if workloads_root.exists() else '(none)'}"
    )


def run(
    workload_name: str,
    num_generations: int = 50,
    extra_args: list[str] | None = None,
) -> int:
    """Launch run_evo.py for the given workload.

    Returns the subprocess exit code.
    """
    workload_dir = find_workload_dir(workload_name)
    run_evo = workload_dir / "run_evo.py"
    if not run_evo.exists():
        raise FileNotFoundError(f"run_evo.py not found in {workload_dir}")

    # Detect seed file: first .cu file that matches the workload name,
    # else first .cu that is not a results artifact
    seed_name = None
    for cu in sorted(workload_dir.glob("*.cu")):
        if cu.stem == workload_dir.name:
            seed_name = cu.name
            break
    if seed_name is None:
        cu_files = [f.name for f in workload_dir.glob("*.cu")]
        if cu_files:
            seed_name = cu_files[0]
        else:
            raise FileNotFoundError(f"No .cu seed file found in {workload_dir}")

    results_dir = f"results_{workload_dir.name}"

    cmd = [
        sys.executable, str(run_evo),
        "--init_program", seed_name,
        "--results_dir", results_dir,
        "--num_generations", str(num_generations),
    ]
    if extra_args:
        cmd.extend(extra_args)

    print(f"  Launching evolution in {workload_dir}")
    print(f"  Seed: {seed_name}  |  Generations: {num_generations}")
    print(f"  Results: {results_dir}")
    print()

    return subprocess.call(cmd, cwd=str(workload_dir))
