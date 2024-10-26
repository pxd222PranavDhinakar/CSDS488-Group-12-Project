// flash_attention.cu
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Constants for SRAM tiling and thread organization
#define BLOCK_M 64            // Sequence dimension tile size
#define BLOCK_N 64            // Sequence dimension tile size
#define BLOCK_K 64            // Head dimension (must match head_dim)
#define WARPS_PER_BLOCK 4     // Number of warps per thread block
#define THREADS_PER_WARP 32   // Standard warp size
#define THREAD_N 8            // Elements per thread in N dimension
#define THREAD_M 8            // Elements per thread in M dimension
#define THREADS_PER_BLOCK (WARPS_PER_BLOCK * THREADS_PER_WARP)

// Derived constants
#define ELEMS_PER_THREAD (THREAD_M * THREAD_N)

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Helper to compute softmax scaling factor
__device__ float compute_softmax_scaling(int head_dim) {
    return 1.0f / sqrtf(float(head_dim));
}

// Shared memory structure for Q, K, V tiles
struct TiledAttention {
    __align__(16) __half q[BLOCK_M][BLOCK_K];  // Add alignment
    __align__(16) __half k[BLOCK_N][BLOCK_K];
    __align__(16) __half v[BLOCK_N][BLOCK_K];
    __align__(8) float s[BLOCK_M][BLOCK_N];
    __align__(8) float m[BLOCK_M];
    __align__(8) float l[BLOCK_M];
};

__global__ void flash_attention_forward(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half* __restrict__ O,
    float* __restrict__ L,
    float* __restrict__ M,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim) {

    extern __shared__ TiledAttention shared_mem;
    
    // Thread indexing
    const int tid = threadIdx.x;
    const int warp_id = tid / THREADS_PER_WARP;
    const int lane_id = tid % THREADS_PER_WARP;
    
    // Block indices
    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int tile_idx_m = blockIdx.z;
    
    // Calculate row and col for this thread
    const int row = tile_idx_m * BLOCK_M + (tid / THREAD_N);
    const int col = (tid % THREAD_N) * (BLOCK_N / THREAD_N);
    
    // Base pointers for this batch and head
    const size_t batch_head_offset = (batch_idx * num_heads + head_idx) * seq_len * head_dim;
    const __half* q_base = Q + batch_head_offset;
    const __half* k_base = K + batch_head_offset;
    const __half* v_base = V + batch_head_offset;
    __half* o_base = O + batch_head_offset;
    
    // Scale factor for attention scores
    const float scale = 1.0f / sqrtf(float(head_dim));
    
    // Load and process tiles
    for (int tile_idx_n = 0; tile_idx_n < seq_len; tile_idx_n += BLOCK_N) {
        // Load K,V tiles
        for (int k = tid; k < BLOCK_K * BLOCK_N; k += blockDim.x) {
            const int tile_row = k / BLOCK_K;
            const int tile_col = k % BLOCK_K;
            if (tile_idx_n + tile_row < seq_len) {
                const int k_idx = (tile_idx_n + tile_row) * head_dim + tile_col;
                shared_mem.k[tile_row][tile_col] = k_base[k_idx];
                shared_mem.v[tile_row][tile_col] = v_base[k_idx];
            }
        }
        __syncthreads();
        
        // Load Q tile
        for (int k = tid; k < BLOCK_M * BLOCK_K; k += blockDim.x) {
            const int tile_row = k / BLOCK_K;
            const int tile_col = k % BLOCK_K;
            if (tile_idx_m * BLOCK_M + tile_row < seq_len) {
                const int q_idx = (tile_idx_m * BLOCK_M + tile_row) * head_dim + tile_col;
                shared_mem.q[tile_row][tile_col] = q_base[q_idx];
            }
        }
        __syncthreads();
        
        // Compute attention for this thread's elements
        if (row < seq_len) {
            float acc[THREAD_N] = {0.0f};
            float max_val = -INFINITY;
            float sum_exp = 0.0f;
            
            // Compute dot products and find max
            for (int k = 0; k < BLOCK_K; k++) {
                const float q_val = __half2float(shared_mem.q[tid/THREAD_N][k]) * scale;
                #pragma unroll
                for (int n = 0; n < THREAD_N; n++) {
                    const float k_val = __half2float(shared_mem.k[col + n][k]);
                    acc[n] += q_val * k_val;
                    max_val = fmaxf(max_val, acc[n]);
                }
            }
            
            // Compute exponentials and sum
            #pragma unroll
            for (int n = 0; n < THREAD_N; n++) {
                const float exp_val = expf(acc[n] - max_val);
                acc[n] = exp_val;
                sum_exp += exp_val;
            }
            
            // Write outputs
            #pragma unroll
            for (int n = 0; n < THREAD_N; n++) {
                const float attn_weight = acc[n] / sum_exp;
                float o_val = 0.0f;
                for (int k = 0; k < BLOCK_K; k++) {
                    o_val += attn_weight * __half2float(shared_mem.v[col + n][k]);
                }
                if (col + n < seq_len) {
                    const int out_idx = row * head_dim + (col + n);
                    o_base[out_idx] = __float2half(o_val);
                }
            }
            
            // Store statistics (only once per row)
            if (lane_id == 0) {
                M[batch_head_offset/head_dim + row] = max_val;
                L[batch_head_offset/head_dim + row] = sum_exp;
            }
        }
        __syncthreads();
    }
}

// Helper function to calculate required shared memory size
inline size_t get_shared_memory_size() {
    return sizeof(TiledAttention);
}

inline void get_launch_config(
    dim3& grid_dim, 
    dim3& block_dim,
    const int batch_size,
    const int num_heads,
    const int seq_len) {
    
    // Each block processes BLOCK_M rows of the sequence
    const int blocks_per_seq = (seq_len + BLOCK_M - 1) / BLOCK_M;
    
    grid_dim = dim3(
        batch_size,          // Batch dimension
        num_heads,           // Number of heads
        blocks_per_seq       // Blocks for sequence length
    );
    
    // Use fixed thread block size
    block_dim = dim3(THREADS_PER_WARP * WARPS_PER_BLOCK);
    
    printf("\nLaunch configuration details:\n");
    printf("Grid dim: (%d, %d, %d)\n", grid_dim.x, grid_dim.y, grid_dim.z);
    printf("Block dim: (%d, 1, 1)\n", block_dim.x);
    printf("Threads per block: %d\n", block_dim.x);
    printf("Shared memory size: %zu bytes\n", get_shared_memory_size());
}

void validateDimensions(int batch_size, int num_heads, int seq_len, int head_dim) {
    printf("\nValidating dimensions:\n");
    printf("Sequence length: %d\n", seq_len);
    printf("Head dimension: %d\n", head_dim);
    printf("Block M/N: %d\n", BLOCK_M);
    printf("Block K: %d\n", BLOCK_K);
    
    if (batch_size <= 0 || num_heads <= 0 || seq_len <= 0 || head_dim <= 0) {
        printf("Error: Invalid dimensions\n");
        exit(1);
    }
    
    if (head_dim != BLOCK_K) {
        printf("Error: head_dim must match BLOCK_K (%d != %d)\n", 
               head_dim, BLOCK_K);
        exit(1);
    }
    
    if (seq_len % BLOCK_M != 0) {
        printf("Error: seq_len must be multiple of BLOCK_M (%d %% %d = %d)\n",
               seq_len, BLOCK_M, seq_len % BLOCK_M);
        exit(1);
    }
    
    const size_t shared_mem_size = get_shared_memory_size();
    printf("Required shared memory: %zu bytes\n", shared_mem_size);
    
    // Verify thread block configuration
    const int total_threads = THREADS_PER_BLOCK;
    printf("Threads per block: %d\n", total_threads);
    
    if (total_threads > 1024) {  // Maximum threads per block on most GPUs
        printf("Error: Too many threads per block (%d)\n", total_threads);
        exit(1);
    }
    
    printf("All dimensions validated successfully\n");
}

void validateLaunchConfig(const dim3& grid, const dim3& block, size_t shared_mem_size) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("\nDevice Properties:\n");
    printf("Device name: %s\n", prop.name);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Max shared memory per block: %zu bytes\n", prop.sharedMemPerBlock);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    
    // Validate thread configuration
    int total_threads = block.x * block.y * block.z;
    if (total_threads > prop.maxThreadsPerBlock) {
        printf("Error: Thread block size (%d) exceeds maximum (%d)\n",
               total_threads, prop.maxThreadsPerBlock);
        exit(1);
    }
    
    // Validate shared memory
    if (shared_mem_size > prop.sharedMemPerBlock) {
        printf("Error: Shared memory size (%zu) exceeds maximum (%zu)\n",
               shared_mem_size, prop.sharedMemPerBlock);
        exit(1);
    }
}

// Modify main() to use smaller dimensions and add validation:

int main() {
    // Ensure dimensions are compatible with block sizes
    const int batch_size = 1;
    const int num_heads = 8;
    const int seq_len = 512;  // Must be multiple of BLOCK_M
    const int head_dim = 64;  // Must equal BLOCK_K
    
    // Print dimensions before validation
    printf("\nRequested dimensions:\n");
    printf("batch_size: %d\n", batch_size);
    printf("num_heads: %d\n", num_heads);
    printf("seq_len: %d\n", seq_len);
    printf("head_dim: %d\n", head_dim);
    
    // Validate dimensions
    validateDimensions(batch_size, num_heads, seq_len, head_dim);
    
    // Print configuration
    printf("Problem Size:\n");
    printf("batch_size: %d\n", batch_size);
    printf("num_heads: %d\n", num_heads);
    printf("seq_len: %d\n", seq_len);
    printf("head_dim: %d\n", head_dim);
    printf("total elements processed: %d\n", 
           batch_size * num_heads * seq_len * head_dim);
        
    // Calculate sizes
    const size_t qkv_size = batch_size * num_heads * seq_len * head_dim * sizeof(__half);
    const size_t o_size = qkv_size;  // Fix: o_size was incomplete
    const size_t softmax_stats_size = batch_size * num_heads * seq_len * sizeof(float);  // Add: missing size calculation
    
    printf("\nMemory sizes:\n");
    printf("QKV size: %zu bytes\n", qkv_size);
    printf("Output size: %zu bytes\n", o_size);
    printf("Softmax stats size: %zu bytes\n", softmax_stats_size);
    
    // Allocate host memory
    __half* h_Q = (__half*)malloc(qkv_size);
    __half* h_K = (__half*)malloc(qkv_size);
    __half* h_V = (__half*)malloc(qkv_size);
    __half* h_O = (__half*)malloc(o_size);
    float* h_L = (float*)malloc(softmax_stats_size);
    float* h_M = (float*)malloc(softmax_stats_size);
    
    // Initialize with a more meaningful pattern
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < num_heads; h++) {
            for (int s = 0; s < seq_len; s++) {
                for (int d = 0; d < head_dim; d++) {
                    size_t idx = ((b * num_heads + h) * seq_len + s) * head_dim + d;
                    // Initialize Q with alternating pattern
                    h_Q[idx] = __float2half((float)(s % 10) / 10.0f);
                    // Initialize K with increasing pattern
                    h_K[idx] = __float2half((float)(d % 10) / 10.0f);
                    // Initialize V with alternating pattern
                    h_V[idx] = __float2half(1.0f - ((float)(s % 2) * 0.5f));
                }
            }
        }
    }
    
    // Print some input values for verification
    printf("Input values (first few elements):\n");
    printf("Q: ");
    for (int i = 0; i < 5; i++) {
        printf("%f ", __half2float(h_Q[i]));
    }
    printf("\nK: ");
    for (int i = 0; i < 5; i++) {
        printf("%f ", __half2float(h_K[i]));
    }
    printf("\nV: ");
    for (int i = 0; i < 5; i++) {
        printf("%f ", __half2float(h_V[i]));
    }
    printf("\n");
    
    // Allocate device memory
    __half *d_Q, *d_K, *d_V, *d_O;
    float *d_L, *d_M;
    CUDA_CHECK(cudaMalloc(&d_Q, qkv_size));
    CUDA_CHECK(cudaMalloc(&d_K, qkv_size));
    CUDA_CHECK(cudaMalloc(&d_V, qkv_size));
    CUDA_CHECK(cudaMalloc(&d_O, o_size));
    CUDA_CHECK(cudaMalloc(&d_L, softmax_stats_size));
    CUDA_CHECK(cudaMalloc(&d_M, softmax_stats_size));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, qkv_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K, qkv_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, qkv_size, cudaMemcpyHostToDevice));
    
    // Launch configuration
    dim3 grid_dim, block_dim;
    get_launch_config(grid_dim, block_dim, batch_size, num_heads, seq_len);
    
    // Print launch configuration
    printf("\nLaunch configuration:\n");
    printf("Grid dimensions: (%d, %d, %d)\n", grid_dim.x, grid_dim.y, grid_dim.z);
    printf("Block dimensions: (%d, %d)\n", block_dim.x, block_dim.y);
    printf("Shared memory size: %zu bytes\n", get_shared_memory_size());
    
    
    // Before kernel launch
    size_t shared_mem_size = get_shared_memory_size();
    printf("Required shared memory: %zu bytes\n", shared_mem_size);
    
    // Get device properties
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device shared memory per block: %zu bytes\n", prop.sharedMemPerBlock);
    
    if (shared_mem_size > prop.sharedMemPerBlock) {
        printf("Error: Required shared memory (%zu) exceeds device capability (%zu)\n",
               shared_mem_size, prop.sharedMemPerBlock);
        exit(1);
    }

    // Add this timing code to both implementations
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch kernel with additional error checking
    printf("\nLaunching kernel...\n");
    flash_attention_forward<<<grid_dim, block_dim, shared_mem_size>>>(
        d_Q, d_K, d_V, d_O, d_L, d_M,
        batch_size, num_heads, seq_len, head_dim
    );
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(error));
        exit(1);
    }
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    
    // Record stop time
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    // Calculate elapsed time
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("\nFlash Attention kernel execution time: %f ms\n", milliseconds);
    
    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_O, d_O, o_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_L, d_L, softmax_stats_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_M, d_M, softmax_stats_size, cudaMemcpyDeviceToHost));
    
    // Print output statistics
    printf("\nOutput values:\n");
    printf("First few elements: ");
    for (int i = 0; i < 5; i++) {
        printf("%f ", __half2float(h_O[i]));
    }
    printf("\n");
    
    printf("Softmax statistics for first row:\n");
    printf("Max value (M): %f\n", h_M[0]);
    printf("Sum value (L): %f\n", h_L[0]);
    
    // Cleanup
    free(h_Q); free(h_K); free(h_V); free(h_O); free(h_L); free(h_M);
    CUDA_CHECK(cudaFree(d_Q)); CUDA_CHECK(cudaFree(d_K)); 
    CUDA_CHECK(cudaFree(d_V)); CUDA_CHECK(cudaFree(d_O));
    CUDA_CHECK(cudaFree(d_L)); CUDA_CHECK(cudaFree(d_M));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return 0;
}