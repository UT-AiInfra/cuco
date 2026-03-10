"""NCCL 2.29+ device-comm bindings for cuco.

Exposes the three host-side APIs added in NCCL 2.29 that are not yet part of
nccl4py, plus the associated wrapper types.

Typical usage::

    from cuco.nccl import comm_query_properties, DevCommRequirements, dev_comm_create, dev_comm_destroy

    props = comm_query_properties(comm._comm)
    if props.device_api_support and props.n_lsa_teams > 0:
        reqs  = DevCommRequirements(lsa_barrier_count=1)
        dcomm = dev_comm_create(comm._comm, reqs)
        # pass dcomm.address + dcomm.nbytes to cudaMemcpy, then launch kernel
        dev_comm_destroy(comm._comm, dcomm)

The Cython extension (_nccl) must be compiled before use::

    pip install -e ".[nccl]"
"""

# Pre-load the correct libnccl.so.2 from the nvidia wheel before importing the
# Cython extension.  Without this, LD_LIBRARY_PATH may resolve an older system
# NCCL (e.g. 2.28.x) before our RUNPATH, causing undefined-symbol errors for
# APIs that were added in 2.29 (ncclCommQueryProperties, ncclDevCommCreate,
# ncclDevCommDestroy).  RTLD_GLOBAL makes the symbols available to _nccl.so
# without a second dlopen.
import ctypes as _ctypes
try:
    from cuda.pathfinder import load_nvidia_dynamic_lib as _load
    _ctypes.CDLL(_load("nccl").abs_path, mode=_ctypes.RTLD_GLOBAL)
except Exception:
    pass

from cuco.nccl._nccl import (  # noqa: F401
    API_MAGIC,
    VERSION_CODE,
    DEVCOMM_NBYTES,
    CommProperties,
    DevCommRequirements,
    DevComm,
    comm_query_properties,
    dev_comm_create,
    dev_comm_destroy,
)
