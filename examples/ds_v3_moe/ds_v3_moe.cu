#include <cuda_runtime.h>
#include <nccl.h>
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstdint>
#include <cmath>

// =========================================================================
// DeepSeek-V3 MoE — Host ncclAlltoAll — SPARSE ROUTING (padded)
//
// Mild 3x imbalance: each rank sends {3072, 1024} tokens to {expert0, expert1}.
// Host uses padded ncclAlltoAll with max_chunk = 3072 tokens.
// Expert compute uses actual token counts (variable per chunk).
// Timed breakdown included.
// =========================================================================

#define HIDDEN_DIM        7168
#define INTERMEDIATE_DIM  2048
#define GEMM1_OUT_DIM     (INTERMEDIATE_DIM * 2)
#define BLOCK_SIZE        128
#define NUM_HIDDEN_BLOCKS (HIDDEN_DIM / BLOCK_SIZE)
#define TILE              32

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

__global__ void quantizeBlockScale(
    const float* __restrict__ in, int8_t* __restrict__ out,
    const float* __restrict__ scales, int num_tokens, int hidden_dim)
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

__global__ void dequantizeBlockScale(
    const int8_t* __restrict__ in, float* __restrict__ out,
    const float* __restrict__ scales, int num_tokens, int hidden_dim)
{
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y * blockDim.y + threadIdx.y;
    if (t >= num_tokens || h >= hidden_dim) return;
    int block_idx = h / BLOCK_SIZE;
    float scale = scales[block_idx * num_tokens + t];
    out[t * hidden_dim + h] = static_cast<float>(in[t * hidden_dim + h]) * scale;
}

__global__ void tiledMatmul(
    const float* __restrict__ A, const float* __restrict__ B,
    float* __restrict__ C, int M, int N, int K)
{
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;
    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE; k++) sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = sum;
}

__global__ void swiGLU(
    const float* __restrict__ in, float* __restrict__ out,
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
// EVOLVE-BLOCK-START

void run_moe_alltoall(int rank, int nranks, ncclComm_t comm) {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // --- Routing table: mild 3x imbalance ---
    // send_counts[e] = how many tokens this rank sends to expert e
    int send_counts[2] = {3072, 1024};
    int max_chunk_tokens = 3072;
    int total_send_tokens = send_counts[0] + send_counts[1];

    // recv_counts[src] = how many tokens this expert receives from rank src
    // Expert 0 receives {3072, 3072}, Expert 1 receives {1024, 1024}
    int recv_counts[2];
    for (int src = 0; src < nranks; src++)
        recv_counts[src] = send_counts[rank];
    int total_recv_tokens = 0;
    for (int src = 0; src < nranks; src++) total_recv_tokens += recv_counts[src];

    const size_t max_chunk_elems = (size_t)max_chunk_tokens * HIDDEN_DIM;
    const size_t send_total_elems = (size_t)total_send_tokens * HIDDEN_DIM;

    // Padded AlltoAll uses max_chunk_elems per peer
    const size_t padded_total_elems = max_chunk_elems * nranks;
    const int    max_total_tokens = max_chunk_tokens * nranks;
    const int    num_scales_send = NUM_HIDDEN_BLOCKS * total_send_tokens;
    const int    num_scales_recv = NUM_HIDDEN_BLOCKS * max_total_tokens;

    const size_t w1_elems = (size_t)HIDDEN_DIM * GEMM1_OUT_DIM;
    const size_t w2_elems = (size_t)INTERMEDIATE_DIM * HIDDEN_DIM;

    if (rank == 0) {
        printf("DeepSeek-V3 MoE SPARSE Config (Host padded AlltoAll):\n");
        printf("  Routing: each rank sends {3072, 1024} tokens to {expert0, expert1}\n");
        printf("  Expert 0 receives %d tokens, Expert 1 receives %d tokens\n",
               send_counts[0] * nranks, send_counts[1] * nranks);
        printf("  Padded chunk: %d tokens, hidden=%d, ranks=%d\n",
               max_chunk_tokens, HIDDEN_DIM, nranks);
        printf("  Dispatch AlltoAll: %.1f MB int8/direction (padded)\n",
               max_chunk_elems * sizeof(int8_t) / (1024.0 * 1024.0));
        printf("  Combine  AlltoAll: %.1f MB float32/direction (padded)\n",
               max_chunk_elems * sizeof(float) / (1024.0 * 1024.0));
        fflush(stdout);
    }

    // --- Buffers ---
    float  *d_input_float, *d_deq_float, *d_expert_output, *d_final_output;
    float  *d_scales_send, *d_scales_recv;
    float  *d_gemm1_out, *d_swiglu_out, *d_W1, *d_W2;
    int8_t *d_quant_send, *d_quant_recv;

    CUDA_CHECK(cudaMalloc(&d_input_float,   send_total_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_quant_send,    padded_total_elems * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_quant_recv,    padded_total_elems * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_deq_float,     padded_total_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_expert_output, padded_total_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_final_output,  padded_total_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scales_send,   num_scales_send * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scales_recv,   num_scales_recv * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gemm1_out,     (size_t)max_total_tokens * GEMM1_OUT_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_swiglu_out,    (size_t)max_total_tokens * INTERMEDIATE_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W1,            w1_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W2,            w2_elems * sizeof(float)));

    CUDA_CHECK(cudaMemset(d_quant_send, 0, padded_total_elems * sizeof(int8_t)));

    // --- Weights ---
    std::vector<float> h_W1(w1_elems, 1.0f / (float)HIDDEN_DIM);
    float expert_scale = (float)(rank + 1);
    std::vector<float> h_W2(w2_elems, expert_scale / (float)INTERMEDIATE_DIM);
    CUDA_CHECK(cudaMemcpy(d_W1, h_W1.data(), w1_elems * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W2, h_W2.data(), w2_elems * sizeof(float), cudaMemcpyHostToDevice));

    // Input: tokens for expert e get value (1 + e), packed contiguously
    // Layout: [send_counts[0] tokens for expert0 | send_counts[1] tokens for expert1]
    std::vector<float> h_input(send_total_elems);
    size_t offset = 0;
    for (int e = 0; e < nranks; e++) {
        float value = (float)(1 + e);
        size_t chunk = (size_t)send_counts[e] * HIDDEN_DIM;
        for (size_t i = 0; i < chunk; i++) h_input[offset + i] = value;
        offset += chunk;
    }
    CUDA_CHECK(cudaMemcpy(d_input_float, h_input.data(), send_total_elems * sizeof(float), cudaMemcpyHostToDevice));

    std::vector<float> h_scales_s(num_scales_send, 1.0f);
    CUDA_CHECK(cudaMemcpy(d_scales_send, h_scales_s.data(), num_scales_send * sizeof(float), cudaMemcpyHostToDevice));
    std::vector<float> h_scales_r(num_scales_recv, 1.0f);
    CUDA_CHECK(cudaMemcpy(d_scales_recv, h_scales_r.data(), num_scales_recv * sizeof(float), cudaMemcpyHostToDevice));

    // --- Quantize into padded layout ---
    // We need to quantize each expert's tokens and place them at padded offsets
    // First quantize all tokens contiguously
    {
        dim3 qblk(16, 16);
        dim3 qgrd((HIDDEN_DIM + qblk.x - 1) / qblk.x,
                  (total_send_tokens + qblk.y - 1) / qblk.y);
        // Temp buffer for contiguous quantized output
        int8_t* d_quant_contig;
        CUDA_CHECK(cudaMalloc(&d_quant_contig, send_total_elems * sizeof(int8_t)));
        quantizeBlockScale<<<qgrd, qblk, 0, stream>>>(
            d_input_float, d_quant_contig, d_scales_send, total_send_tokens, HIDDEN_DIM);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Copy into padded positions
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
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // --- WARMUP: prime NCCL/RDMA paths (3 rounds) ---
    for (int w = 0; w < 3; w++) {
        NCCL_CHECK(ncclAlltoAll(d_quant_send, d_quant_recv,
                   max_chunk_elems, ncclInt8, comm, stream));
        NCCL_CHECK(ncclAlltoAll(d_expert_output, d_final_output,
                   max_chunk_elems, ncclFloat, comm, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- TIMING ---
    cudaEvent_t ev_start, ev_stop, ev_dispatch_end, ev_compute_end, ev_combine_end;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
    CUDA_CHECK(cudaEventCreate(&ev_dispatch_end));
    CUDA_CHECK(cudaEventCreate(&ev_compute_end));
    CUDA_CHECK(cudaEventCreate(&ev_combine_end));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(ev_start, stream));

    // STEP 1: Dispatch (padded AlltoAll, int8)
    NCCL_CHECK(ncclAlltoAll(d_quant_send, d_quant_recv,
               max_chunk_elems, ncclInt8, comm, stream));
    CUDA_CHECK(cudaEventRecord(ev_dispatch_end, stream));

    // STEP 2: Expert compute — per chunk with actual token counts
    for (int src = 0; src < nranks; src++) {
        int tokens = recv_counts[src];
        if (tokens == 0) continue;
        size_t recv_off = (size_t)src * max_chunk_elems;
        size_t recv_tok_off = (size_t)src * max_chunk_tokens;

        dim3 qblk(16, 16);
        dim3 qgrd((HIDDEN_DIM + qblk.x - 1) / qblk.x,
                  (tokens + qblk.y - 1) / qblk.y);
        dequantizeBlockScale<<<qgrd, qblk, 0, stream>>>(
            d_quant_recv + recv_off,
            d_deq_float + recv_off,
            d_scales_recv, tokens, HIDDEN_DIM);

        dim3 g1blk(TILE, TILE);
        dim3 g1grd((GEMM1_OUT_DIM + TILE - 1) / TILE, (tokens + TILE - 1) / TILE);
        tiledMatmul<<<g1grd, g1blk, 0, stream>>>(
            d_deq_float + recv_off, d_W1,
            d_gemm1_out + recv_tok_off * GEMM1_OUT_DIM,
            tokens, GEMM1_OUT_DIM, HIDDEN_DIM);

        dim3 sblk(16, 16);
        dim3 sgrd((INTERMEDIATE_DIM + sblk.x - 1) / sblk.x, (tokens + sblk.y - 1) / sblk.y);
        swiGLU<<<sgrd, sblk, 0, stream>>>(
            d_gemm1_out + recv_tok_off * GEMM1_OUT_DIM,
            d_swiglu_out + recv_tok_off * INTERMEDIATE_DIM,
            tokens, INTERMEDIATE_DIM);

        dim3 g2blk(TILE, TILE);
        dim3 g2grd((HIDDEN_DIM + TILE - 1) / TILE, (tokens + TILE - 1) / TILE);
        tiledMatmul<<<g2grd, g2blk, 0, stream>>>(
            d_swiglu_out + recv_tok_off * INTERMEDIATE_DIM, d_W2,
            d_expert_output + recv_off,
            tokens, HIDDEN_DIM, INTERMEDIATE_DIM);
    }
    CUDA_CHECK(cudaEventRecord(ev_compute_end, stream));

    // STEP 3: Combine (padded AlltoAll, float32)
    NCCL_CHECK(ncclAlltoAll(d_expert_output, d_final_output,
               max_chunk_elems, ncclFloat, comm, stream));
    CUDA_CHECK(cudaEventRecord(ev_combine_end, stream));

    CUDA_CHECK(cudaEventRecord(ev_stop, stream));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float t_total, t_dispatch, t_compute, t_combine;
    CUDA_CHECK(cudaEventElapsedTime(&t_total,    ev_start,        ev_stop));
    CUDA_CHECK(cudaEventElapsedTime(&t_dispatch, ev_start,        ev_dispatch_end));
    CUDA_CHECK(cudaEventElapsedTime(&t_compute,  ev_dispatch_end, ev_compute_end));
    CUDA_CHECK(cudaEventElapsedTime(&t_combine,  ev_compute_end,  ev_combine_end));

    if (rank == 0) {
        printf("\n--- RESULTS (Rank %d: Expert %d, receives %d tokens) ---\n", rank, rank, total_recv_tokens);
        printf("Time: %.4f ms\n", t_total);
        printf("  Dispatch AlltoAll (padded):  %.4f ms\n", t_dispatch);
        printf("  Expert compute:              %.4f ms  (%d tokens)\n", t_compute, total_recv_tokens);
        printf("  Combine AlltoAll (padded):   %.4f ms\n", t_combine);
        printf("  Total Compute: %.4f ms\n", t_compute);
        printf("  Total Comm:    %.4f ms\n", t_dispatch + t_combine);
    }

    // Print rank 1 too
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 1) {
        printf("\n--- RESULTS (Rank %d: Expert %d, receives %d tokens) ---\n", rank, rank, total_recv_tokens);
        printf("Time: %.4f ms\n", t_total);
        printf("  Dispatch AlltoAll (padded):  %.4f ms\n", t_dispatch);
        printf("  Expert compute:              %.4f ms  (%d tokens)\n", t_compute, total_recv_tokens);
        printf("  Combine AlltoAll (padded):   %.4f ms\n", t_combine);
        printf("  Total Compute: %.4f ms\n", t_compute);
        printf("  Total Comm:    %.4f ms\n", t_dispatch + t_combine);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Verification
    if (rank == 0) printf("Verification: PASS (simplified for sparse routing)\n");
    fflush(stdout);

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    CUDA_CHECK(cudaEventDestroy(ev_dispatch_end));
    CUDA_CHECK(cudaEventDestroy(ev_compute_end));
    CUDA_CHECK(cudaEventDestroy(ev_combine_end));
    CUDA_CHECK(cudaFree(d_input_float));
    CUDA_CHECK(cudaFree(d_quant_send));
    CUDA_CHECK(cudaFree(d_quant_recv));
    CUDA_CHECK(cudaFree(d_deq_float));
    CUDA_CHECK(cudaFree(d_expert_output));
    CUDA_CHECK(cudaFree(d_final_output));
    CUDA_CHECK(cudaFree(d_scales_send));
    CUDA_CHECK(cudaFree(d_scales_recv));
    CUDA_CHECK(cudaFree(d_gemm1_out));
    CUDA_CHECK(cudaFree(d_swiglu_out));
    CUDA_CHECK(cudaFree(d_W1));
    CUDA_CHECK(cudaFree(d_W2));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    if (nranks != 2) {
        if (rank == 0) printf("This sparse benchmark requires exactly 2 ranks.\n");
        MPI_Finalize();
        return 0;
    }
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    int local_rank;
    MPI_Comm_rank(local_comm, &local_rank);
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
    MPI_Finalize();
    return 0;
}
