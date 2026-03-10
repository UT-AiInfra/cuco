"""Build script for Cython extensions.

Setuptools calls this automatically when building the package.  The nccl
extension requires the nvidia.nccl package to be installed (provides headers
and libnccl.so.2).
"""

from __future__ import annotations

from pathlib import Path

from setuptools import setup


def _nccl_paths() -> tuple[str, str]:
    """Return (include_dir, lib_dir) from the nvidia.nccl wheel."""
    try:
        import nvidia.nccl as _pkg
        base = Path(list(_pkg.__path__)[0])
        return str(base / "include"), str(base / "lib")
    except ImportError:
        # Fall back to system NCCL
        return "/usr/local/include", "/usr/local/lib"


def _build_nccl_ext():
    try:
        from Cython.Build import cythonize
        from setuptools import Extension
    except ImportError:
        return []

    nccl_inc, nccl_lib = _nccl_paths()

    ext = Extension(
        name="cuco.nccl._nccl",
        sources=["cuco/nccl/_nccl.pyx"],
        language="c++",
        include_dirs=[nccl_inc],
        library_dirs=[nccl_lib],
        # Use the colon form so the linker finds libnccl.so.2 even without an
        # unversioned symlink.  Bake the rpath so it resolves at runtime too.
        extra_link_args=[
            f"-Wl,-rpath,{nccl_lib}",
            f"-L{nccl_lib}",
            "-l:libnccl.so.2",
        ],
    )
    return cythonize(
        [ext],
        compiler_directives={"language_level": "3"},
        build_dir="build",
    )


setup(ext_modules=_build_nccl_ext())
