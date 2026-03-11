"""Integration tests for CUDA compilation and 2-GPU NCCL device-API flows."""

from __future__ import annotations

from contextlib import ExitStack

import pytest

from cuco.compile import CompileOptions, compile_cuda

WORLD_SIZE = 2
LSA_INIT_METHOD = "tcp://127.0.0.1:29500"
ALLREDUCE_INIT_METHOD = "tcp://127.0.0.1:29501"
LSA_SKIP_REASON = "LSA not available (deviceApiSupport=False or nLsaTeams=0)"

# Kernel launch config — CTA count must equal lsaBarrierCount
_CTA_COUNT = 4
_THREADS_PER_CTA = 128
_COUNT = 1 << 20  # float32 elements per rank (~4 MB)

# Full LSA allreduce kernel:
#   - send/recv buffers are ncclMemAlloc symmetric memory registered as windows
#   - each CTA owns one LSA barrier slot (blockIdx.x)
#   - ncclGetLsaPointer gives direct peer memory access without explicit transfers
#   - two bar.sync fences bracket the reduction (relaxed acquire, release at end)
_LSA_ALLREDUCE_SRC = r"""
#include <nccl_device.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

extern "C" __global__ void lsa_allreduce_kernel(
    ncclWindow_t sendwin,
    ncclWindow_t recvwin,
    size_t       count,
    const ncclDevComm* devCommPtr)
{
    const ncclDevComm devComm = *devCommPtr;
    auto coop = cg::this_thread_block();

    // One barrier slot per CTA — index == blockIdx.x, count == gridDim.x
    ncclLsaBarrierSession<cg::thread_block> bar(
        coop, devComm, ncclTeamTagLsa{}, blockIdx.x);

    // Acquire fence: wait until all peers have written their send buffers
    bar.sync(coop, cuda::memory_order_relaxed);

    const int rank          = devComm.rank;
    const int nRanks        = devComm.nRanks;
    const int globalTid     = threadIdx.x + blockDim.x * (rank + blockIdx.x * nRanks);
    const int globalStride  = blockDim.x * gridDim.x * nRanks;

    for (size_t i = globalTid; i < count; i += globalStride) {
        // Reduce: sum over all peers' send windows
        float v = 0.f;
        for (int peer = 0; peer < nRanks; peer++) {
            float* p = (float*)ncclGetLsaPointer(sendwin, 0, peer);
            v += p[i];
        }
        // Scatter result into all peers' recv windows
        for (int peer = 0; peer < nRanks; peer++) {
            float* p = (float*)ncclGetLsaPointer(recvwin, 0, peer);
            p[i] = v;
        }
    }

    // Release fence: signal that recv buffers are fully written before stream proceeds
    bar.sync(coop, cuda::memory_order_release);
}
"""


@pytest.fixture(scope="session")
def lsa_cubin():
    try:
        return compile_cuda(_LSA_ALLREDUCE_SRC, CompileOptions(std="c++17"))
    except RuntimeError as exc:
        pytest.skip(f"nvcc failed: {exc}")


def _require_lsa_dependencies():
    pytest.importorskip("nccl.core", reason="nccl4py not installed")
    pytest.importorskip(
        "cuco.nccl",
        reason="cuco.nccl not built (run: pip install -e '.[nccl]')",
    )


def _require_2_gpus():
    import torch

    n = torch.cuda.device_count()
    if n < WORLD_SIZE:
        pytest.skip(f"need >={WORLD_SIZE} GPUs, found {n}")


def _init_process_group(rank, init_method):
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=WORLD_SIZE,
        init_method=init_method,
    )
    return dist


def _cleanup_process_group(dist):
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _maybe_set_skip_reason(skip_q, rank, reason):
    if rank == 0:
        skip_q.put(reason)


def _lsa_worker(rank, cubin_bytes, results, skip_q):
    import nccl.core as nccl
    from nccl.core.constants import WindowFlag
    from nccl.core.interop.torch import empty as nccl_empty
    import torch
    from cuda.bindings import runtime as cudart
    from cuda.core import Device, LaunchConfig, ObjectCode, launch
    from cuda.core._utils.cuda_utils import handle_return

    from cuco.nccl import (
        DEVCOMM_NBYTES,
        DevCommRequirements,
        comm_query_properties,
        dev_comm_create,
        dev_comm_destroy,
    )

    dist = _init_process_group(rank, LSA_INIT_METHOD)
    try:
        with ExitStack() as cleanup:
            dev = Device(rank)
            dev.set_current()

            uid_bytes = [nccl.get_unique_id().as_bytes if rank == 0 else None]
            dist.broadcast_object_list(uid_bytes, src=0)
            comm = nccl.Communicator.init(
                WORLD_SIZE,
                rank,
                nccl.UniqueId.from_bytes(uid_bytes[0]),
            )
            cleanup.callback(comm.destroy)

            props = comm_query_properties(comm._comm)
            if not props.device_api_support or props.n_lsa_teams == 0:
                _maybe_set_skip_reason(skip_q, rank, LSA_SKIP_REASON)
                return

            send_buf = nccl_empty(_COUNT, dtype=torch.float32, device=rank)
            recv_buf = nccl_empty(_COUNT, dtype=torch.float32, device=rank)
            send_buf.fill_(float(rank))
            recv_buf.zero_()

            send_win = comm.register_window(send_buf, WindowFlag.CollSymmetric)
            recv_win = comm.register_window(recv_buf, WindowFlag.CollSymmetric)
            if send_win is None or recv_win is None:
                _maybe_set_skip_reason(
                    skip_q,
                    rank,
                    "ncclCommWindowRegister returned NULL (unsupported platform)",
                )
                return
            cleanup.callback(recv_win.close)
            cleanup.callback(send_win.close)

            dcomm = dev_comm_create(
                comm._comm,
                DevCommRequirements(lsa_barrier_count=_CTA_COUNT),
            )
            cleanup.callback(dev_comm_destroy, comm._comm, dcomm)

            devcomm_buf = torch.zeros(
                DEVCOMM_NBYTES,
                dtype=torch.uint8,
                device=f"cuda:{rank}",
            )
            handle_return(
                cudart.cudaMemcpy(
                    devcomm_buf.data_ptr(),
                    dcomm.address,
                    DEVCOMM_NBYTES,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                )
            )

            stream = dev.create_stream()
            cleanup.callback(stream.close)

            kernel = ObjectCode.from_cubin(cubin_bytes).get_kernel("lsa_allreduce_kernel")
            launch(
                stream,
                LaunchConfig(grid=_CTA_COUNT, block=_THREADS_PER_CTA),
                kernel,
                send_win.handle,
                recv_win.handle,
                _COUNT,
                devcomm_buf.data_ptr(),
            )
            stream.sync()
            results[rank].copy_(recv_buf.cpu())

    finally:
        _cleanup_process_group(dist)


def _dist_worker(rank, results):
    import torch

    dist = _init_process_group(rank, ALLREDUCE_INIT_METHOD)
    try:
        t = torch.full((_COUNT,), float(rank), device=f"cuda:{rank}")
        dist.all_reduce(t)
        results[rank].copy_(t.cpu())
    finally:
        _cleanup_process_group(dist)


def test_lsa_allreduce(lsa_cubin):
    """Run LSA allreduce over 2 ranks; compare against dist.all_reduce baseline."""
    import torch
    import torch.multiprocessing as mp

    _require_lsa_dependencies()
    _require_2_gpus()

    # --- LSA kernel results ---
    lsa_results = torch.zeros(WORLD_SIZE, _COUNT).share_memory_()
    skip_q = mp.get_context("spawn").SimpleQueue()
    mp.spawn(_lsa_worker, args=(lsa_cubin, lsa_results, skip_q), nprocs=WORLD_SIZE, join=True)

    if not skip_q.empty():
        pytest.skip(skip_q.get())

    # --- dist.all_reduce baseline ---
    dist_results = torch.zeros(WORLD_SIZE, _COUNT).share_memory_()
    mp.spawn(_dist_worker, args=(dist_results,), nprocs=WORLD_SIZE, join=True)

    for rank in range(WORLD_SIZE):
        assert lsa_results[rank].allclose(dist_results[rank]), (
            f"rank {rank}: LSA result differs from dist.all_reduce baseline; "
            f"lsa[:4]={lsa_results[rank][:4].tolist()}, "
            f"dist[:4]={dist_results[rank][:4].tolist()}"
        )


def test_torch_dist_allreduce():
    """Smoke-test dist.all_reduce independently."""
    import torch
    import torch.multiprocessing as mp

    _require_2_gpus()
    results = torch.zeros(WORLD_SIZE, _COUNT).share_memory_()
    mp.spawn(_dist_worker, args=(results,), nprocs=WORLD_SIZE, join=True)
    expected_val = float(sum(range(WORLD_SIZE)))  # 0+1 = 1.0
    expected = torch.full((_COUNT,), expected_val)
    for rank in range(WORLD_SIZE):
        assert results[rank].allclose(expected), (
            f"rank {rank}: expected {expected_val}, got {results[rank][:4].tolist()}"
        )
