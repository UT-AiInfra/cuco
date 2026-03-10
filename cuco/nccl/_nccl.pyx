# cython: language_level=3
# distutils: language = c++
"""Cython bindings for the NCCL 2.29+ device-comm API.

Wraps ncclCommQueryProperties, ncclDevCommCreate, ncclDevCommDestroy and the
associated structs from nccl_device/core.h.  All three functions and their
struct types live in that header (added in NCCL 2.29).

Usage::

    from cuco.nccl import comm_query_properties, DevCommRequirements, dev_comm_create, dev_comm_destroy

    props = comm_query_properties(comm._comm)   # comm._comm is an int handle from nccl4py
    if props.device_api_support and props.n_lsa_teams > 0:
        reqs  = DevCommRequirements(lsa_barrier_count=1)
        dcomm = dev_comm_create(comm._comm, reqs)
        # dcomm.address / dcomm.nbytes for cudaMemcpy
        dev_comm_destroy(comm._comm, dcomm)
"""

from libc.stdlib cimport malloc, free
from libc.string cimport memset
from libc.stdint cimport uintptr_t
from libcpp cimport bool as cbool


# ---------------------------------------------------------------------------
# C declarations
# ---------------------------------------------------------------------------

cdef extern from "nccl.h" nogil:
    ctypedef void* ncclComm_t

    # Expose compile-time constants as Cython-visible C values.
    # NCCL_API_MAGIC is an unsigned int #define (0xcafebeef).
    # NCCL_VERSION_CODE is an int #define (e.g. 22907 for 2.29.7).
    unsigned int NCCL_API_MAGIC
    int NCCL_VERSION_CODE


cdef extern from "nccl_device/core.h" nogil:

    ctypedef int ncclGinType_t
    ctypedef int ncclGinConnectionType_t

    ctypedef struct ncclDevCommRequirements_t:
        size_t           size
        unsigned int     magic
        unsigned int     version
        void*            resourceRequirementsList
        void*            teamRequirementsList
        cbool            lsaMultimem
        int              barrierCount
        int              lsaBarrierCount
        int              railGinBarrierCount
        int              lsaLLA2ABlockCount
        int              lsaLLA2ASlotCount
        cbool            ginForceEnable
        int              ginContextCount
        int              ginSignalCount
        int              ginCounterCount
        ncclGinConnectionType_t ginConnectionType
        cbool            ginExclusiveContexts
        int              ginQueueDepth

    ctypedef struct ncclCommProperties_t:
        size_t           size
        unsigned int     magic
        unsigned int     version
        int              rank
        int              nRanks
        int              cudaDev
        int              nvmlDev
        cbool            deviceApiSupport
        cbool            multimemSupport
        ncclGinType_t    ginType
        int              nLsaTeams
        cbool            hostRmaSupport
        ncclGinType_t    railedGinType

    # ncclDevComm is an opaque device-side struct; we hold a malloc'd buffer
    # of NCCL_DEVCOMM_NBYTES bytes and cast to this pointer type.
    ctypedef struct ncclDevComm_t:
        pass

    int ncclCommQueryProperties(ncclComm_t, ncclCommProperties_t*)
    int ncclDevCommCreate(ncclComm_t, const ncclDevCommRequirements_t*, ncclDevComm_t*)
    int ncclDevCommDestroy(ncclComm_t, const ncclDevComm_t*)


# ---------------------------------------------------------------------------
# Python-visible constants
# ---------------------------------------------------------------------------

#: NCCL_API_MAGIC — must match the magic field in all versioned structs.
API_MAGIC: int = <unsigned int>NCCL_API_MAGIC

#: NCCL_VERSION_CODE baked in at compile time (e.g. 22907 for NCCL 2.29.7).
VERSION_CODE: int = NCCL_VERSION_CODE

#: Allocation size for the opaque ncclDevComm struct.
#: sizeof(ncclDevComm) == 224 on NCCL 2.29.x (verified with nvcc).
#: We allocate 256 bytes to leave headroom for minor ABI changes.
DEVCOMM_NBYTES: int = 256


# ---------------------------------------------------------------------------
# CommProperties — read-only result of ncclCommQueryProperties
# ---------------------------------------------------------------------------

cdef class CommProperties:
    """Properties of an NCCL communicator returned by :func:`comm_query_properties`."""

    cdef ncclCommProperties_t _p

    @property
    def rank(self) -> int:
        return self._p.rank

    @property
    def n_ranks(self) -> int:
        return self._p.nRanks

    @property
    def cuda_dev(self) -> int:
        return self._p.cudaDev

    @property
    def nvml_dev(self) -> int:
        return self._p.nvmlDev

    @property
    def device_api_support(self) -> bool:
        return bool(self._p.deviceApiSupport)

    @property
    def multimem_support(self) -> bool:
        return bool(self._p.multimemSupport)

    @property
    def gin_type(self) -> int:
        return int(self._p.ginType)

    @property
    def n_lsa_teams(self) -> int:
        return self._p.nLsaTeams

    @property
    def host_rma_support(self) -> bool:
        return bool(self._p.hostRmaSupport)

    @property
    def railed_gin_type(self) -> int:
        return int(self._p.railedGinType)

    def __repr__(self) -> str:
        return (
            f"CommProperties(rank={self.rank}, n_ranks={self.n_ranks}, "
            f"cuda_dev={self.cuda_dev}, device_api_support={self.device_api_support}, "
            f"n_lsa_teams={self.n_lsa_teams})"
        )


# ---------------------------------------------------------------------------
# DevCommRequirements — input to ncclDevCommCreate
# ---------------------------------------------------------------------------

cdef class DevCommRequirements:
    """Writable wrapper around ``ncclDevCommRequirements_t``.

    ``size``, ``magic``, and ``version`` are pre-filled from the compile-time
    NCCL constants; all other fields default to zero / False.

    Example::

        reqs = DevCommRequirements(lsa_barrier_count=1)
    """

    cdef ncclDevCommRequirements_t _r

    def __cinit__(
        self,
        int lsa_barrier_count=0,
        int barrier_count=0,
        int rail_gin_barrier_count=0,
        cbool lsa_multimem=False,
        int gin_context_count=4,   # NCCL default hint
        int gin_signal_count=0,
        int gin_counter_count=0,
        int lsa_lla2a_block_count=0,
        int lsa_lla2a_slot_count=0,
        cbool gin_force_enable=False,
        int gin_connection_type=0,  # NCCL_GIN_CONNECTION_NONE
        cbool gin_exclusive_contexts=False,
        int gin_queue_depth=0,
    ):
        memset(&self._r, 0, sizeof(ncclDevCommRequirements_t))
        self._r.size    = sizeof(ncclDevCommRequirements_t)
        self._r.magic   = NCCL_API_MAGIC
        self._r.version = NCCL_VERSION_CODE
        self._r.lsaBarrierCount      = lsa_barrier_count
        self._r.barrierCount         = barrier_count
        self._r.railGinBarrierCount  = rail_gin_barrier_count
        self._r.lsaMultimem          = lsa_multimem
        self._r.ginContextCount      = gin_context_count
        self._r.ginSignalCount       = gin_signal_count
        self._r.ginCounterCount      = gin_counter_count
        self._r.lsaLLA2ABlockCount   = lsa_lla2a_block_count
        self._r.lsaLLA2ASlotCount    = lsa_lla2a_slot_count
        self._r.ginForceEnable       = gin_force_enable
        self._r.ginConnectionType    = <ncclGinConnectionType_t>gin_connection_type
        self._r.ginExclusiveContexts = gin_exclusive_contexts
        self._r.ginQueueDepth        = gin_queue_depth

    @property
    def lsa_barrier_count(self) -> int:
        return self._r.lsaBarrierCount

    @lsa_barrier_count.setter
    def lsa_barrier_count(self, int v):
        self._r.lsaBarrierCount = v


# ---------------------------------------------------------------------------
# DevComm — opaque device comm handle returned by ncclDevCommCreate
# ---------------------------------------------------------------------------

cdef class DevComm:
    """Holds the opaque ``ncclDevComm_t`` buffer produced by :func:`dev_comm_create`.

    Attributes
    ----------
    address : int
        Raw host-side address of the ``ncclDevComm`` buffer, suitable for
        passing to ``cudaMemcpy`` as the source.
    nbytes : int
        Allocation size (``DEVCOMM_NBYTES``).
    """

    cdef void* _buf

    def __cinit__(self):
        self._buf = malloc(DEVCOMM_NBYTES)
        if self._buf == NULL:
            raise MemoryError("failed to allocate ncclDevComm buffer")
        memset(self._buf, 0, DEVCOMM_NBYTES)

    def __dealloc__(self):
        if self._buf != NULL:
            free(self._buf)
            self._buf = NULL

    @property
    def address(self) -> int:
        """Host address of the ncclDevComm buffer as a Python int."""
        return <uintptr_t>self._buf

    @property
    def nbytes(self) -> int:
        return DEVCOMM_NBYTES


# ---------------------------------------------------------------------------
# Public API functions
# ---------------------------------------------------------------------------

def comm_query_properties(comm_handle: int) -> CommProperties:
    """Query properties of an NCCL communicator.

    Parameters
    ----------
    comm_handle:
        Raw integer handle from ``nccl4py`` (``comm._comm``).

    Returns
    -------
    CommProperties
        Filled properties struct.

    Raises
    ------
    RuntimeError
        If ``ncclCommQueryProperties`` returns a non-zero result code.
    """
    cdef CommProperties props = CommProperties.__new__(CommProperties)
    memset(&props._p, 0, sizeof(ncclCommProperties_t))
    props._p.size    = sizeof(ncclCommProperties_t)
    props._p.magic   = NCCL_API_MAGIC
    props._p.version = NCCL_VERSION_CODE
    cdef ncclComm_t raw = <ncclComm_t><uintptr_t>comm_handle
    cdef int rc = ncclCommQueryProperties(raw, &props._p)
    if rc != 0:
        raise RuntimeError(f"ncclCommQueryProperties failed (rc={rc})")
    return props


def dev_comm_create(comm_handle: int, DevCommRequirements reqs) -> DevComm:
    """Create a device communicator for use in kernels.

    Parameters
    ----------
    comm_handle:
        Raw integer handle from ``nccl4py`` (``comm._comm``).
    reqs:
        Requirements specifying what resources the device comm needs.

    Returns
    -------
    DevComm
        Allocated and initialised device comm buffer.

    Raises
    ------
    RuntimeError
        If ``ncclDevCommCreate`` returns a non-zero result code.
    """
    cdef DevComm dcomm = DevComm.__new__(DevComm)
    cdef ncclComm_t raw = <ncclComm_t><uintptr_t>comm_handle
    cdef int rc = ncclDevCommCreate(raw, &reqs._r, <ncclDevComm_t*>dcomm._buf)
    if rc != 0:
        raise RuntimeError(f"ncclDevCommCreate failed (rc={rc})")
    return dcomm


def dev_comm_destroy(comm_handle: int, DevComm dcomm) -> None:
    """Destroy a device communicator created by :func:`dev_comm_create`.

    Parameters
    ----------
    comm_handle:
        Raw integer handle from ``nccl4py`` (``comm._comm``).
    dcomm:
        Device comm to destroy.

    Raises
    ------
    RuntimeError
        If ``ncclDevCommDestroy`` returns a non-zero result code.
    """
    cdef ncclComm_t raw = <ncclComm_t><uintptr_t>comm_handle
    cdef int rc = ncclDevCommDestroy(raw, <ncclDevComm_t*>dcomm._buf)
    if rc != 0:
        raise RuntimeError(f"ncclDevCommDestroy failed (rc={rc})")
