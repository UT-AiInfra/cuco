#include <cuda_runtime.h>
#include <nccl.h>
#include <nccl_device.h>
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstdint>
#include <cmath>
#include <cstring>

// =========================================================================
// DeepSeek-V3 MoE Expert Parallelism — 2-GPU AlltoAll Benchmark
// SPARSE ROUTING: each rank sends {3072, 1024} tokens to {expert0, expert1}.
// GIN-transformed version: host-side ncclAlltoAll replaced with device-side
// GIN kernels. Sequential pipeline — no overlap, no merged kernels.
// =========================================================================

#define HIDDEN_DIM        7168
#define INTERMEDIATE_DIM  2048
#define GEMM1_OUT_DIM     (INTERMEDIATE_DIM * 2)  // 4096: gate + up projection
#define BLOCK_SIZE        128
#define NUM_HIDDEN_BLOCKS (HIDDEN_DIM / BLOCK_SIZE)
#define TILE              32

// GIN kernel launch parameters: 1 block (must match railGinBarrierCount=1), 512 threads
#define GIN_BLOCKS   1
#define GIN_THREADS  512

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define NCCL_CHECK(call) do { \
    ncclResult_t res = call; \
    if (res != ncclSuccess) { \
        fprintf(stderr, "NCCL error at %s:%d: %s\n", __FILE__, __LINE__, ncclGetErrorString(res)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// EVOLVE-BLOCK-START
// -----------------------------------------------------------------
// Block-Scale FP8 Quantization (int8 stand-in for float8_e4m3fn)
// -----------------------------------------------------------------
__global__ void quantizeBlockScale(
    const float* __restrict__ in,
    int8_t*      __restrict__ out,
    const float* __restrict__ scales,
    int num_tokens, int hidden_dim)
{
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y * blockDim.y + threadIdx.y;
    if (t >= num_tokens || h >= hidden_dim) return;

    int block_idx = h / BLOCK_SIZE;
    float scale = scales[block_idx * num_tokens + t];
    float val = in[t * hidden_dim + h] / scale;
    val = fminf(fmaxf(val, -127.0f), 127.0f);
    out[t * hidden_dim + h] = static_cast<int8_t>(rintf(val));
}

// -----------------------------------------------------------------
// Block-Scale Dequantization: int8 → float32
// -----------------------------------------------------------------
__global__ void dequantizeBlockScale(
    const int8_t* __restrict__ in,
    float*        __restrict__ out,
    const float*  __restrict__ scales,
    int num_tokens, int hidden_dim)
{
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y * blockDim.y + threadIdx.y;
    if (t >= num_tokens || h >= hidden_dim) return;

    int block_idx = h / BLOCK_SIZE;
    float scale = scales[block_idx * num_tokens + t];
    out[t * hidden_dim + h] = static_cast<float>(in[t * hidden_dim + h]) * scale;
}

// -----------------------------------------------------------------
// Shared-Memory Tiled Matrix Multiply: C[M,N] = A[M,K] × B[K,N]
// -----------------------------------------------------------------
__global__ void tiledMatmul(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int M, int N, int K)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;
    int numTiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < numTiles; t++) {
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

// -----------------------------------------------------------------
// SwiGLU activation
// -----------------------------------------------------------------
__global__ void swiGLU(
    const float* __restrict__ in,
    float*       __restrict__ out,
    int num_tokens, int intermediate_dim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y * blockDim.y + threadIdx.y;
    if (t >= num_tokens || i >= intermediate_dim) return;

    int gemm1_out_dim = intermediate_dim * 2;
    float gate = in[t * gemm1_out_dim + i];
    float up   = in[t * gemm1_out_dim + i + intermediate_dim];
    float silu_gate = gate / (1.0f + expf(-gate));
    out[t * intermediate_dim + i] = silu_gate * up;
}

// -----------------------------------------------------------------
// Simple memcpy kernel for self-copy (runs on compute stream)
// -----------------------------------------------------------------
__global__ void memcpyInt8Kernel(
    int8_t* __restrict__ dst,
    const int8_t* __restrict__ src,
    size_t count)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    for (size_t i = idx; i < count; i += stride) {
        dst[i] = src[i];
    }
}

__global__ void memcpyFloatKernel(
    float* __restrict__ dst,
    const float* __restrict__ src,
    size_t count)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    for (size_t i = idx; i < count; i += stride) {
        dst[i] = src[i];
    }
}

// -----------------------------------------------------------------
// GIN PUT kernel with self-copy fused - issues transfers AND copies local data
// -----------------------------------------------------------------
__global__ void ginDispatchPutWithSelfCopyKernel(
    ncclDevComm devComm,
    ncclWindow_t send_win,
    ncclWindow_t recv_win,
    int8_t* __restrict__ quant_send,
    int8_t* __restrict__ quant_recv,
    const size_t* __restrict__ d_send_bytes,
    const size_t* __restrict__ d_send_offsets,
    const size_t* __restrict__ d_recv_offsets,
    size_t self_copy_elements,
    size_t self_copy_offset)
{
    ncclGin gin(devComm, 0);
    ncclTeam world = ncclTeamWorld(devComm);
    int my_rank = world.rank;
    int nranks = world.nRanks;

    // Set up shadow for expected incoming signals
    if (threadIdx.x == 0) {
        gin.increaseSignalShadow(0, nranks - 1);
    }
    __syncthreads();

    // Thread 0 issues puts to all remote peers
    if (threadIdx.x == 0) {
        for (int r = 0; r < nranks; r++) {
            if (r != my_rank) {
                size_t bytes = d_send_bytes[r];
                if (bytes > 0) {
                    size_t src_offset = d_send_offsets[r];
                    size_t dst_offset = d_recv_offsets[my_rank];
                    gin.put(world, r,
                            recv_win, dst_offset,
                            send_win, src_offset,
                            bytes,
                            ncclGin_SignalInc{0},
                            ncclGin_None{},
                            ncclCoopThread());
                }
            }
        }
    }

    // All threads participate in self-copy while PUT is in flight
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (size_t i = idx; i < self_copy_elements; i += stride) {
        quant_recv[self_copy_offset + i] = quant_send[self_copy_offset + i];
    }
}

// -----------------------------------------------------------------
// GIN WAIT kernel - waits for all incoming transfers to complete
// -----------------------------------------------------------------
__global__ void ginDispatchWaitKernel(ncclDevComm devComm)
{
    ncclGin gin(devComm, 0);
    
    // Wait for signal to meet shadow value
    gin.waitSignalMeetShadow(ncclCoopThread{}, 0);
    
    // Flush to ensure source buffers are safe
    if (threadIdx.x == 0) {
        gin.flush(ncclCoopThread());
    }
}

// -----------------------------------------------------------------
// GIN Combine PUT kernel with self-copy fused
// -----------------------------------------------------------------
__global__ void ginCombinePutWithSelfCopyKernel(
    ncclDevComm devComm,
    ncclWindow_t send_win,
    ncclWindow_t recv_win,
    float* __restrict__ expert_output,
    float* __restrict__ final_output,
    const size_t* __restrict__ d_send_bytes,
    const size_t* __restrict__ d_send_offsets,
    const size_t* __restrict__ d_recv_offsets,
    size_t self_copy_elements,
    size_t self_copy_offset)
{
    ncclGin gin(devComm, 0);
    ncclTeam world = ncclTeamWorld(devComm);
    int my_rank = world.rank;
    int nranks = world.nRanks;

    // Set up shadow for expected incoming signals
    if (threadIdx.x == 0) {
        gin.increaseSignalShadow(1, nranks - 1);
    }
    __syncthreads();

    // Thread 0 issues puts to all remote peers
    if (threadIdx.x == 0) {
        for (int r = 0; r < nranks; r++) {
            if (r != my_rank) {
                size_t bytes = d_send_bytes[r];
                if (bytes > 0) {
                    size_t src_offset = d_send_offsets[r];
                    size_t dst_offset = d_recv_offsets[my_rank];
                    gin.put(world, r,
                            recv_win, dst_offset,
                            send_win, src_offset,
                            bytes,
                            ncclGin_SignalInc{1},
                            ncclGin_None{},
                            ncclCoopThread());
                }
            }
        }
    }

    // All threads participate in self-copy
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (size_t i = idx; i < self_copy_elements; i += stride) {
        final_output[self_copy_offset + i] = expert_output[self_copy_offset + i];
    }
}

// -----------------------------------------------------------------
// GIN Combine WAIT kernel
// -----------------------------------------------------------------
__global__ void ginCombineWaitKernel(ncclDevComm devComm)
{
    ncclGin gin(devComm, 0);
    
    // Wait for signal to meet shadow value (using signal index 1 for combine)
    gin.waitSignalMeetShadow(ncclCoopThread{}, 1);
    
    // Flush
    if (threadIdx.x == 0) {
        gin.flush(ncclCoopThread());
    }
}

// -----------------------------------------------------------------
// Combined AlltoAll kernels for warmup (uses barriers)
// -----------------------------------------------------------------
__global__ void ginAlltoAllInt8Kernel(
    ncclDevComm devComm,
    ncclWindow_t send_win,
    ncclWindow_t recv_win,
    int8_t* __restrict__ quant_send,
    int8_t* __restrict__ quant_recv,
    const size_t* __restrict__ d_send_bytes,
    const size_t* __restrict__ d_send_offsets,
    const size_t* __restrict__ d_recv_offsets,
    size_t self_copy_elements,
    size_t self_copy_offset)
{
    ncclGin gin(devComm, 0);
    ncclTeam world = ncclTeamWorld(devComm);
    int my_rank = world.rank;
    int nranks = world.nRanks;

    uint64_t signalValue = gin.readSignal(0);

    ncclGinBarrierSession<ncclCoopCta> bar(
        ncclCoopCta(), gin, ncclTeamWorld(devComm),
        devComm.railGinBarrier, blockIdx.x);
    bar.sync(ncclCoopCta(), cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);

    if (threadIdx.x == 0) {
        for (int r = 0; r < nranks; r++) {
            if (r != my_rank) {
                size_t bytes = d_send_bytes[r];
                if (bytes > 0) {
                    gin.put(world, r,
                            recv_win, d_recv_offsets[my_rank],
                            send_win, d_send_offsets[r],
                            bytes,
                            ncclGin_SignalInc{0},
                            ncclGin_None{},
                            ncclCoopThread());
                }
            }
        }
    }

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (size_t i = idx; i < self_copy_elements; i += stride) {
        quant_recv[self_copy_offset + i] = quant_send[self_copy_offset + i];
    }

    gin.waitSignal(ncclCoopCta(), 0, signalValue + (uint64_t)(nranks - 1));
    if (threadIdx.x == 0) gin.flush(ncclCoopThread());
    __syncthreads();
}

__global__ void ginAlltoAllFloat32Kernel(
    ncclDevComm devComm,
    ncclWindow_t send_win,
    ncclWindow_t recv_win,
    float* __restrict__ expert_output,
    float* __restrict__ final_output,
    const size_t* __restrict__ d_send_bytes,
    const size_t* __restrict__ d_send_offsets,
    const size_t* __restrict__ d_recv_offsets,
    size_t self_copy_elements,
    size_t self_copy_offset)
{
    ncclGin gin(devComm, 0);
    ncclTeam world = ncclTeamWorld(devComm);
    int my_rank = world.rank;
    int nranks = world.nRanks;

    uint64_t signalValue = gin.readSignal(1);

    ncclGinBarrierSession<ncclCoopCta> bar(
        ncclCoopCta(), gin, ncclTeamWorld(devComm),
        devComm.railGinBarrier, blockIdx.x);
    bar.sync(ncclCoopCta(), cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);

    if (threadIdx.x == 0) {
        for (int r = 0; r < nranks; r++) {
            if (r != my_rank) {
                size_t bytes = d_send_bytes[r];
                if (bytes > 0) {
                    gin.put(world, r,
                            recv_win, d_recv_offsets[my_rank],
                            send_win, d_send_offsets[r],
                            bytes,
                            ncclGin_SignalInc{1},
                            ncclGin_None{},
                            ncclCoopThread());
                }
            }
        }
    }

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (size_t i = idx; i < self_copy_elements; i += stride) {
        final_output[self_copy_offset + i] = expert_output[self_copy_offset + i];
    }

    gin.waitSignal(ncclCoopCta(), 1, signalValue + (uint64_t)(nranks - 1));
    if (threadIdx.x == 0) gin.flush(ncclCoopThread());
    __syncthreads();
}

// EVOLVE-BLOCK-END
// -----------------------------------------------------------------
// Host Orchestration
// -----------------------------------------------------------------
void run_moe_alltoall(int rank, int nranks, ncclComm_t comm) {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // --- Sparse routing table ---
    // Each rank sends 3072 tokens to expert 0 and 1024 tokens to expert 1
    int send_counts[2] = {3072, 1024};
    int max_chunk_tokens = 3072;
    int total_send_tokens = send_counts[0] + send_counts[1];  // 4096

    // recv_counts[src] = tokens received from rank src
    // Each rank sends send_counts[my_rank_as_expert] tokens to me
    int recv_counts[2];
    for (int src = 0; src < nranks; src++)
        recv_counts[src] = send_counts[rank];

    int total_recv_tokens = 0;
    for (int src = 0; src < nranks; src++) total_recv_tokens += recv_counts[src];

    // Buffer layout: use max_chunk_tokens per slot for simplicity
    const size_t max_chunk_elems = (size_t)max_chunk_tokens * HIDDEN_DIM;
    const size_t padded_total_elems = max_chunk_elems * nranks;
    const int    max_total_tokens = max_chunk_tokens * nranks;
    const size_t w1_elems = (size_t)HIDDEN_DIM * GEMM1_OUT_DIM;
    const size_t w2_elems = (size_t)INTERMEDIATE_DIM * HIDDEN_DIM;

    if (rank == 0) {
        printf("DeepSeek-V3 MoE SPARSE Config (GIN sequential):\n");
        printf("  Routing: each rank sends {3072, 1024} tokens to {expert0, expert1}\n");
        printf("  Expert 0 receives %d tokens, Expert 1 receives %d tokens\n",
               send_counts[0] * nranks, send_counts[1] * nranks);
        printf("  ranks=%d, hidden=%d, intermediate=%d\n", nranks, HIDDEN_DIM, INTERMEDIATE_DIM);
        printf("  GEMM1: [tokens, %d] x [%d, %d]  GEMM2: [tokens, %d] x [%d, %d]\n",
               HIDDEN_DIM, HIDDEN_DIM, GEMM1_OUT_DIM,
               INTERMEDIATE_DIM, INTERMEDIATE_DIM, HIDDEN_DIM);
        fflush(stdout);
    }

    // --- Buffers ---
    float  *d_input_float, *d_deq_float, *d_expert_output, *d_final_output;
    float  *d_scales;
    float  *d_gemm1_out, *d_swiglu_out, *d_W1, *d_W2;
    int8_t *d_quant_send, *d_quant_recv;

    CUDA_CHECK(cudaMalloc(&d_input_float,  (size_t)total_send_tokens * HIDDEN_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_deq_float,    padded_total_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scales,       NUM_HIDDEN_BLOCKS * max_total_tokens * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gemm1_out,    (size_t)max_total_tokens * GEMM1_OUT_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_swiglu_out,   (size_t)max_total_tokens * INTERMEDIATE_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W1,           w1_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W2,           w2_elems * sizeof(float)));

    // Timer events (used in frozen verification section below)
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    // Phase timing events
    cudaEvent_t ev_dispatch_put_start, ev_dispatch_put_end;
    cudaEvent_t ev_dispatch_wait_end;
    cudaEvent_t ev_self_dequant_end, ev_self_gemm1_end, ev_self_swiglu_end, ev_self_gemm2_end;
    cudaEvent_t ev_remote_dequant_end, ev_remote_gemm1_end, ev_remote_swiglu_end, ev_remote_gemm2_end;
    cudaEvent_t ev_combine_put_end, ev_combine_wait_end;

    CUDA_CHECK(cudaEventCreate(&ev_dispatch_put_start));
    CUDA_CHECK(cudaEventCreate(&ev_dispatch_put_end));
    CUDA_CHECK(cudaEventCreate(&ev_dispatch_wait_end));
    CUDA_CHECK(cudaEventCreate(&ev_self_dequant_end));
    CUDA_CHECK(cudaEventCreate(&ev_self_gemm1_end));
    CUDA_CHECK(cudaEventCreate(&ev_self_swiglu_end));
    CUDA_CHECK(cudaEventCreate(&ev_self_gemm2_end));
    CUDA_CHECK(cudaEventCreate(&ev_remote_dequant_end));
    CUDA_CHECK(cudaEventCreate(&ev_remote_gemm1_end));
    CUDA_CHECK(cudaEventCreate(&ev_remote_swiglu_end));
    CUDA_CHECK(cudaEventCreate(&ev_remote_gemm2_end));
    CUDA_CHECK(cudaEventCreate(&ev_combine_put_end));
    CUDA_CHECK(cudaEventCreate(&ev_combine_wait_end));

    // Communicated buffers: use ncclMemAlloc (required for GIN windows)
// EVOLVE-BLOCK-START
    NCCL_CHECK(ncclMemAlloc((void**)&d_quant_send,    padded_total_elems * sizeof(int8_t)));
    NCCL_CHECK(ncclMemAlloc((void**)&d_quant_recv,    padded_total_elems * sizeof(int8_t)));
    NCCL_CHECK(ncclMemAlloc((void**)&d_expert_output, padded_total_elems * sizeof(float)));
    NCCL_CHECK(ncclMemAlloc((void**)&d_final_output,  padded_total_elems * sizeof(float)));

    // --- GIN Infrastructure Setup ---
    ncclWindow_t quant_send_win, quant_recv_win;
    ncclWindow_t expert_output_win, final_output_win;

    NCCL_CHECK(ncclCommWindowRegister(comm, d_quant_send,
               padded_total_elems * sizeof(int8_t), &quant_send_win, NCCL_WIN_COLL_SYMMETRIC));
    NCCL_CHECK(ncclCommWindowRegister(comm, d_quant_recv,
               padded_total_elems * sizeof(int8_t), &quant_recv_win, NCCL_WIN_COLL_SYMMETRIC));
    NCCL_CHECK(ncclCommWindowRegister(comm, d_expert_output,
               padded_total_elems * sizeof(float), &expert_output_win, NCCL_WIN_COLL_SYMMETRIC));
    NCCL_CHECK(ncclCommWindowRegister(comm, d_final_output,
               padded_total_elems * sizeof(float), &final_output_win, NCCL_WIN_COLL_SYMMETRIC));

    // Create device communicator with 2 signals (dispatch=0, combine=1)
    ncclDevComm devComm;
    ncclDevCommRequirements reqs;
    memset(&reqs, 0, sizeof(reqs));
    reqs.ginContextCount     = 1;
    reqs.ginSignalCount      = 2;  // Signal 0 for dispatch, signal 1 for combine
    reqs.railGinBarrierCount = GIN_BLOCKS;

    NCCL_CHECK(ncclDevCommCreate(comm, &reqs, &devComm));
    (void)cudaGetLastError();

    // --- Create multiple streams for maximum overlap ---
    cudaStream_t comm_stream, compute_stream;
    CUDA_CHECK(cudaStreamCreateWithPriority(&comm_stream, cudaStreamNonBlocking, -1));  // High priority for comm
    CUDA_CHECK(cudaStreamCreateWithPriority(&compute_stream, cudaStreamNonBlocking, 0)); // Normal priority for compute

    // Events for synchronization
    cudaEvent_t dispatch_done, dispatch_wait_done;
    cudaEvent_t self_compute_done, all_compute_done;
    CUDA_CHECK(cudaEventCreate(&dispatch_done));
    CUDA_CHECK(cudaEventCreate(&dispatch_wait_done));
    CUDA_CHECK(cudaEventCreate(&self_compute_done));
    CUDA_CHECK(cudaEventCreate(&all_compute_done));

    // --- Weight initialization ---
    std::vector<float> h_W1(w1_elems, 1.0f / (float)HIDDEN_DIM);
    float expert_scale = (float)(rank + 1);
    std::vector<float> h_W2(w2_elems, expert_scale / (float)INTERMEDIATE_DIM);
    CUDA_CHECK(cudaMemcpy(d_W1, h_W1.data(), w1_elems * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W2, h_W2.data(), w2_elems * sizeof(float), cudaMemcpyHostToDevice));

    // --- Input initialization ---
    std::vector<float> h_input((size_t)total_send_tokens * HIDDEN_DIM);
    size_t off = 0;
    for (int e = 0; e < nranks; e++) {
        float value = (float)(1 + e);
        size_t chunk = (size_t)send_counts[e] * HIDDEN_DIM;
        for (size_t i = 0; i < chunk; i++) h_input[off + i] = value;
        off += chunk;
    }
    CUDA_CHECK(cudaMemcpy(d_input_float, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice));

    std::vector<float> h_scales(NUM_HIDDEN_BLOCKS * max_total_tokens, 1.0f);
    CUDA_CHECK(cudaMemcpy(d_scales, h_scales.data(), h_scales.size() * sizeof(float), cudaMemcpyHostToDevice));

    // --- Quantize into padded send buffer ---
    {
        int8_t* d_quant_contig;
        CUDA_CHECK(cudaMalloc(&d_quant_contig, (size_t)total_send_tokens * HIDDEN_DIM * sizeof(int8_t)));
        float* d_scales_send;
        CUDA_CHECK(cudaMalloc(&d_scales_send, NUM_HIDDEN_BLOCKS * total_send_tokens * sizeof(float)));
        std::vector<float> h_ss(NUM_HIDDEN_BLOCKS * total_send_tokens, 1.0f);
        CUDA_CHECK(cudaMemcpy(d_scales_send, h_ss.data(), h_ss.size() * sizeof(float), cudaMemcpyHostToDevice));

        dim3 qblk(16, 16);
        dim3 qgrd((HIDDEN_DIM + 15) / 16, (total_send_tokens + 15) / 16);
        quantizeBlockScale<<<qgrd, qblk, 0, stream>>>(
            d_input_float, d_quant_contig, d_scales_send, total_send_tokens, HIDDEN_DIM);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        CUDA_CHECK(cudaMemset(d_quant_send, 0, padded_total_elems * sizeof(int8_t)));
        size_t src_off = 0;
        for (int e = 0; e < nranks; e++) {
            size_t dst_off = (size_t)e * max_chunk_elems;
            size_t count = (size_t)send_counts[e] * HIDDEN_DIM;
            CUDA_CHECK(cudaMemcpy(d_quant_send + dst_off, d_quant_contig + src_off,
                       count * sizeof(int8_t), cudaMemcpyDeviceToDevice));
            src_off += count;
        }
        CUDA_CHECK(cudaFree(d_quant_contig));
        CUDA_CHECK(cudaFree(d_scales_send));
    }

    // --- Per-peer byte counts and offsets ---
    size_t h_dispatch_send_bytes[2], h_dispatch_send_offsets[2], h_dispatch_recv_offsets[2];
    for (int p = 0; p < nranks; p++) {
        h_dispatch_send_bytes[p]   = (size_t)send_counts[p] * HIDDEN_DIM * sizeof(int8_t);
        h_dispatch_send_offsets[p] = (size_t)p * max_chunk_elems * sizeof(int8_t);
        h_dispatch_recv_offsets[p] = (size_t)p * max_chunk_elems * sizeof(int8_t);
    }

    size_t h_combine_send_bytes[2], h_combine_send_offsets[2], h_combine_recv_offsets[2];
    for (int p = 0; p < nranks; p++) {
        h_combine_send_bytes[p]   = (size_t)recv_counts[p] * HIDDEN_DIM * sizeof(float);
        h_combine_send_offsets[p] = (size_t)p * max_chunk_elems * sizeof(float);
        h_combine_recv_offsets[p] = (size_t)p * max_chunk_elems * sizeof(float);
    }

    size_t *d_dispatch_send_bytes, *d_dispatch_send_offsets, *d_dispatch_recv_offsets;
    size_t *d_combine_send_bytes, *d_combine_send_offsets, *d_combine_recv_offsets;
    CUDA_CHECK(cudaMalloc(&d_dispatch_send_bytes,   nranks * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_dispatch_send_offsets,  nranks * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_dispatch_recv_offsets,  nranks * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_combine_send_bytes,    nranks * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_combine_send_offsets,   nranks * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_combine_recv_offsets,   nranks * sizeof(size_t)));
    CUDA_CHECK(cudaMemcpy(d_dispatch_send_bytes,   h_dispatch_send_bytes,   nranks * sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dispatch_send_offsets,  h_dispatch_send_offsets,  nranks * sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dispatch_recv_offsets,  h_dispatch_recv_offsets,  nranks * sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_combine_send_bytes,    h_combine_send_bytes,    nranks * sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_combine_send_offsets,   h_combine_send_offsets,   nranks * sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_combine_recv_offsets,   h_combine_recv_offsets,   nranks * sizeof(size_t), cudaMemcpyHostToDevice));

    size_t dispatch_self_elems  = (size_t)send_counts[rank] * HIDDEN_DIM;
    size_t dispatch_self_offset = (size_t)rank * max_chunk_elems;
    size_t combine_self_elems  = (size_t)recv_counts[rank] * HIDDEN_DIM;
    size_t combine_self_offset = (size_t)rank * max_chunk_elems;

    // --- WARMUP ---
    for (int w = 0; w < 3; w++) {
        ginAlltoAllInt8Kernel<<<GIN_BLOCKS, GIN_THREADS, 0, stream>>>(
            devComm, quant_send_win, quant_recv_win,
            d_quant_send, d_quant_recv,
            d_dispatch_send_bytes, d_dispatch_send_offsets, d_dispatch_recv_offsets,
            dispatch_self_elems, dispatch_self_offset);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        ginAlltoAllFloat32Kernel<<<GIN_BLOCKS, GIN_THREADS, 0, stream>>>(
            devComm, expert_output_win, final_output_win,
            d_expert_output, d_final_output,
            d_combine_send_bytes, d_combine_send_offsets, d_combine_recv_offsets,
            combine_self_elems, combine_self_offset);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // Compute warmup
    {
        int tokens = max_chunk_tokens;
        dim3 qblk(16, 16), qgrd((HIDDEN_DIM + 15) / 16, (tokens + 15) / 16);
        dequantizeBlockScale<<<qgrd, qblk, 0, stream>>>(
            d_quant_recv, d_deq_float, d_scales, tokens, HIDDEN_DIM);
        dim3 g1blk(TILE, TILE), g1grd((GEMM1_OUT_DIM + TILE - 1) / TILE, (tokens + TILE - 1) / TILE);
        tiledMatmul<<<g1grd, g1blk, 0, stream>>>(
            d_deq_float, d_W1, d_gemm1_out, tokens, GEMM1_OUT_DIM, HIDDEN_DIM);
        dim3 sblk(16, 16), sgrd((INTERMEDIATE_DIM + 15) / 16, (tokens + 15) / 16);
        swiGLU<<<sgrd, sblk, 0, stream>>>(
            d_gemm1_out, d_swiglu_out, tokens, INTERMEDIATE_DIM);
        dim3 g2blk(TILE, TILE), g2grd((HIDDEN_DIM + TILE - 1) / TILE, (tokens + TILE - 1) / TILE);
        tiledMatmul<<<g2grd, g2blk, 0, stream>>>(
            d_swiglu_out, d_W2, d_expert_output, tokens, HIDDEN_DIM, INTERMEDIATE_DIM);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    // --- TIMING: Optimized overlapped execution ---
    CUDA_CHECK(cudaEventRecord(ev_start, stream));

    // PHASE 1: Launch dispatch PUT+self-copy on comm_stream (fused kernel)
    CUDA_CHECK(cudaEventRecord(ev_dispatch_put_start, comm_stream));
    ginDispatchPutWithSelfCopyKernel<<<GIN_BLOCKS, GIN_THREADS, 0, comm_stream>>>(
        devComm, quant_send_win, quant_recv_win,
        d_quant_send, d_quant_recv,
        d_dispatch_send_bytes, d_dispatch_send_offsets, d_dispatch_recv_offsets,
        dispatch_self_elems, dispatch_self_offset);
    CUDA_CHECK(cudaEventRecord(ev_dispatch_put_end, comm_stream));
    CUDA_CHECK(cudaEventRecord(dispatch_done, comm_stream));

    // PHASE 2: Immediately launch dispatch wait on comm_stream (polls while compute runs)
    ginDispatchWaitKernel<<<1, 32, 0, comm_stream>>>(devComm);
    CUDA_CHECK(cudaEventRecord(ev_dispatch_wait_end, comm_stream));
    CUDA_CHECK(cudaEventRecord(dispatch_wait_done, comm_stream));

    // PHASE 3: Self-compute on compute_stream (waits for dispatch_done for self-copy completion)
    CUDA_CHECK(cudaStreamWaitEvent(compute_stream, dispatch_done, 0));
    {
        int tokens = recv_counts[rank];
        size_t recv_off = (size_t)rank * max_chunk_elems;
        size_t tok_off = (size_t)rank * max_chunk_tokens;

        dim3 qblk(16, 16);
        dim3 qgrd((HIDDEN_DIM + 15) / 16, (tokens + 15) / 16);
        dequantizeBlockScale<<<qgrd, qblk, 0, compute_stream>>>(
            d_quant_recv + recv_off,
            d_deq_float + recv_off,
            d_scales, tokens, HIDDEN_DIM);
        CUDA_CHECK(cudaEventRecord(ev_self_dequant_end, compute_stream));

        dim3 g1blk(TILE, TILE);
        dim3 g1grd((GEMM1_OUT_DIM + TILE - 1) / TILE, (tokens + TILE - 1) / TILE);
        tiledMatmul<<<g1grd, g1blk, 0, compute_stream>>>(
            d_deq_float + recv_off, d_W1,
            d_gemm1_out + tok_off * GEMM1_OUT_DIM,
            tokens, GEMM1_OUT_DIM, HIDDEN_DIM);
        CUDA_CHECK(cudaEventRecord(ev_self_gemm1_end, compute_stream));

        dim3 sblk(16, 16);
        dim3 sgrd((INTERMEDIATE_DIM + 15) / 16, (tokens + 15) / 16);
        swiGLU<<<sgrd, sblk, 0, compute_stream>>>(
            d_gemm1_out + tok_off * GEMM1_OUT_DIM,
            d_swiglu_out + tok_off * INTERMEDIATE_DIM,
            tokens, INTERMEDIATE_DIM);
        CUDA_CHECK(cudaEventRecord(ev_self_swiglu_end, compute_stream));

        dim3 g2blk(TILE, TILE);
        dim3 g2grd((HIDDEN_DIM + TILE - 1) / TILE, (tokens + TILE - 1) / TILE);
        tiledMatmul<<<g2grd, g2blk, 0, compute_stream>>>(
            d_swiglu_out + tok_off * INTERMEDIATE_DIM, d_W2,
            d_expert_output + recv_off,
            tokens, HIDDEN_DIM, INTERMEDIATE_DIM);
        CUDA_CHECK(cudaEventRecord(ev_self_gemm2_end, compute_stream));
    }
    CUDA_CHECK(cudaEventRecord(self_compute_done, compute_stream));

    // PHASE 4: Remote compute (waits for dispatch_wait_done)
    CUDA_CHECK(cudaStreamWaitEvent(compute_stream, dispatch_wait_done, 0));
    int remote_src = 1 - rank;
    {
        int tokens = recv_counts[remote_src];
        if (tokens > 0) {
            size_t recv_off = (size_t)remote_src * max_chunk_elems;
            size_t tok_off = (size_t)remote_src * max_chunk_tokens;

            dim3 qblk(16, 16);
            dim3 qgrd((HIDDEN_DIM + 15) / 16, (tokens + 15) / 16);
            dequantizeBlockScale<<<qgrd, qblk, 0, compute_stream>>>(
                d_quant_recv + recv_off,
                d_deq_float + recv_off,
                d_scales, tokens, HIDDEN_DIM);
            CUDA_CHECK(cudaEventRecord(ev_remote_dequant_end, compute_stream));

            dim3 g1blk(TILE, TILE);
            dim3 g1grd((GEMM1_OUT_DIM + TILE - 1) / TILE, (tokens + TILE - 1) / TILE);
            tiledMatmul<<<g1grd, g1blk, 0, compute_stream>>>(
                d_deq_float + recv_off, d_W1,
                d_gemm1_out + tok_off * GEMM1_OUT_DIM,
                tokens, GEMM1_OUT_DIM, HIDDEN_DIM);
            CUDA_CHECK(cudaEventRecord(ev_remote_gemm1_end, compute_stream));

            dim3 sblk(16, 16);
            dim3 sgrd((INTERMEDIATE_DIM + 15) / 16, (tokens + 15) / 16);
            swiGLU<<<sgrd, sblk, 0, compute_stream>>>(
                d_gemm1_out + tok_off * GEMM1_OUT_DIM,
                d_swiglu_out + tok_off * INTERMEDIATE_DIM,
                tokens, INTERMEDIATE_DIM);
            CUDA_CHECK(cudaEventRecord(ev_remote_swiglu_end, compute_stream));

            dim3 g2blk(TILE, TILE);
            dim3 g2grd((HIDDEN_DIM + TILE - 1) / TILE, (tokens + TILE - 1) / TILE);
            tiledMatmul<<<g2grd, g2blk, 0, compute_stream>>>(
                d_swiglu_out + tok_off * INTERMEDIATE_DIM, d_W2,
                d_expert_output + recv_off,
                tokens, HIDDEN_DIM, INTERMEDIATE_DIM);
            CUDA_CHECK(cudaEventRecord(ev_remote_gemm2_end, compute_stream));
        }
    }
    CUDA_CHECK(cudaEventRecord(all_compute_done, compute_stream));

    // PHASE 5: Combine PUT+self-copy (waits for all compute)
    CUDA_CHECK(cudaStreamWaitEvent(comm_stream, all_compute_done, 0));
    ginCombinePutWithSelfCopyKernel<<<GIN_BLOCKS, GIN_THREADS, 0, comm_stream>>>(
        devComm, expert_output_win, final_output_win,
        d_expert_output, d_final_output,
        d_combine_send_bytes, d_combine_send_offsets, d_combine_recv_offsets,
        combine_self_elems, combine_self_offset);
    CUDA_CHECK(cudaEventRecord(ev_combine_put_end, comm_stream));

    // PHASE 6: Wait for combine to complete
    ginCombineWaitKernel<<<1, 32, 0, comm_stream>>>(devComm);
    CUDA_CHECK(cudaEventRecord(ev_combine_wait_end, comm_stream));

    // Synchronize everything
    CUDA_CHECK(cudaStreamSynchronize(comm_stream));
    CUDA_CHECK(cudaStreamSynchronize(compute_stream));
// EVOLVE-BLOCK-END

    // --- STOP TIMER ---
    CUDA_CHECK(cudaEventRecord(ev_stop, stream));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, ev_start, ev_stop));

    // --- Phase timing output ---
    float elapsed;
    
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, ev_dispatch_put_start, ev_dispatch_put_end));
    printf("Phase: dispatch_put = %.4f ms\n", elapsed);
    
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, ev_dispatch_put_end, ev_dispatch_wait_end));
    printf("Phase: dispatch_wait = %.4f ms\n", elapsed);
    
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, ev_dispatch_put_end, ev_self_dequant_end));
    printf("Phase: self_dequant = %.4f ms\n", elapsed);
    
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, ev_self_dequant_end, ev_self_gemm1_end));
    printf("Phase: self_gemm1 = %.4f ms\n", elapsed);
    
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, ev_self_gemm1_end, ev_self_swiglu_end));
    printf("Phase: self_swiglu = %.4f ms\n", elapsed);
    
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, ev_self_swiglu_end, ev_self_gemm2_end));
    printf("Phase: self_gemm2 = %.4f ms\n", elapsed);
    
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, ev_dispatch_wait_end, ev_remote_dequant_end));
    printf("Phase: remote_dequant = %.4f ms\n", elapsed);
    
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, ev_remote_dequant_end, ev_remote_gemm1_end));
    printf("Phase: remote_gemm1 = %.4f ms\n", elapsed);
    
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, ev_remote_gemm1_end, ev_remote_swiglu_end));
    printf("Phase: remote_swiglu = %.4f ms\n", elapsed);
    
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, ev_remote_swiglu_end, ev_remote_gemm2_end));
    printf("Phase: remote_gemm2 = %.4f ms\n", elapsed);
    
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, ev_remote_gemm2_end, ev_combine_put_end));
    printf("Phase: combine_put = %.4f ms\n", elapsed);
    
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, ev_combine_put_end, ev_combine_wait_end));
    printf("Phase: combine_wait = %.4f ms\n", elapsed);

    // --- VERIFICATION ---
    std::vector<float> h_output(padded_total_elems);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_final_output,
               padded_total_elems * sizeof(float), cudaMemcpyDeviceToHost));

    bool pass = true;
    float max_err = 0.0f;

    for (int src = 0; src < nranks && pass; src++) {
        int tokens = send_counts[src];
        float v = (float)(1 + src);
        float silu_v = v / (1.0f + expf(-v));
        float expected = silu_v * v * (float)(src + 1);

        size_t base = (size_t)src * max_chunk_elems;
        size_t count = (size_t)tokens * HIDDEN_DIM;

        for (size_t i = 0; i < count; i++) {
            float actual = h_output[base + i];
            float err = fabsf(actual - expected);
            if (err > max_err) max_err = err;
            if (err > 0.05f) {
                fprintf(stderr, "Rank %d: MISMATCH src=%d idx=%zu: "
                        "got %.6f expected %.6f (err=%.6f)\n",
                        rank, src, i, actual, expected, err);
                pass = false;
                break;
            }
        }
    }

    float global_max_err = max_err;
    int   local_pass = pass ? 1 : 0;
    int   global_pass = local_pass;
    MPI_Reduce(&max_err,    &global_max_err, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_pass, &global_pass,    1, MPI_INT,   MPI_MIN, 0, MPI_COMM_WORLD);

    for (int r = 0; r < nranks; r++) {
        if (rank == r) {
            printf("\n--- RESULTS (Rank %d: Expert %d, receives %d tokens) ---\n",
                   rank, rank, total_recv_tokens);
            printf("Time: %.4f ms\n", milliseconds);
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (rank == 0) {
        printf("Verification: %s (max error: %.6f)\n",
               global_pass ? "PASS" : "FAIL", global_max_err);
    }
    fflush(stdout);

    // --- CLEANUP ---
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));

    // Destroy phase timing events
    CUDA_CHECK(cudaEventDestroy(ev_dispatch_put_start));
    CUDA_CHECK(cudaEventDestroy(ev_dispatch_put_end));
    CUDA_CHECK(cudaEventDestroy(ev_dispatch_wait_end));
    CUDA_CHECK(cudaEventDestroy(ev_self_dequant_end));
    CUDA_CHECK(cudaEventDestroy(ev_self_gemm1_end));
    CUDA_CHECK(cudaEventDestroy(ev_self_swiglu_end));
    CUDA_CHECK(cudaEventDestroy(ev_self_gemm2_end));
    CUDA_CHECK(cudaEventDestroy(ev_remote_dequant_end));
    CUDA_CHECK(cudaEventDestroy(ev_remote_gemm1_end));
    CUDA_CHECK(cudaEventDestroy(ev_remote_swiglu_end));
    CUDA_CHECK(cudaEventDestroy(ev_remote_gemm2_end));
    CUDA_CHECK(cudaEventDestroy(ev_combine_put_end));
    CUDA_CHECK(cudaEventDestroy(ev_combine_wait_end));

    NCCL_CHECK(ncclCommWindowDeregister(comm, quant_send_win));
    NCCL_CHECK(ncclCommWindowDeregister(comm, quant_recv_win));
    NCCL_CHECK(ncclCommWindowDeregister(comm, expert_output_win));
    NCCL_CHECK(ncclCommWindowDeregister(comm, final_output_win));
    NCCL_CHECK(ncclDevCommDestroy(comm, &devComm));

    CUDA_CHECK(cudaFree(d_input_float)); CUDA_CHECK(cudaFree(d_deq_float));
    CUDA_CHECK(cudaFree(d_scales)); CUDA_CHECK(cudaFree(d_gemm1_out));
    CUDA_CHECK(cudaFree(d_swiglu_out)); CUDA_CHECK(cudaFree(d_W1)); CUDA_CHECK(cudaFree(d_W2));
    CUDA_CHECK(cudaFree(d_dispatch_send_bytes)); CUDA_CHECK(cudaFree(d_dispatch_send_offsets));
    CUDA_CHECK(cudaFree(d_dispatch_recv_offsets));
    CUDA_CHECK(cudaFree(d_combine_send_bytes)); CUDA_CHECK(cudaFree(d_combine_send_offsets));
    CUDA_CHECK(cudaFree(d_combine_recv_offsets));
    NCCL_CHECK(ncclMemFree(d_quant_send)); NCCL_CHECK(ncclMemFree(d_quant_recv));
    NCCL_CHECK(ncclMemFree(d_expert_output)); NCCL_CHECK(ncclMemFree(d_final_output));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    if (nranks != 2) {
        if (rank == 0) printf("This sparse benchmark requires exactly 2 ranks.\n");
        MPI_Finalize(); return 0;
    }
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    int local_rank; MPI_Comm_rank(local_comm, &local_rank);
    CUDA_CHECK(cudaSetDevice(local_rank));

    ncclUniqueId id;
    if (rank == 0) NCCL_CHECK(ncclGetUniqueId(&id));
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    ncclComm_t comm;
    NCCL_CHECK(ncclCommInitRank(&comm, nranks, id, rank));
    MPI_Barrier(MPI_COMM_WORLD);

    run_moe_alltoall(rank, nranks, comm);

    NCCL_CHECK(ncclCommDestroy(comm));
    MPI_Comm_free(&local_comm);
    MPI_Finalize(); return 0;
}
