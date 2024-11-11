#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <math.h>

// Configuration
#define TILE_SIZE 32
#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32
#define MAX_HEAD_DIM 64  // Maximum supported head dimension
#define OPTIMAL_BLOCKS 32  // Tune this based on your GPU

// Usage information
void print_usage() {
    printf("Usage: ./flash_attention <batch_size> <num_heads> <seq_len> <head_dim>\n");
    printf("Example: ./flash_attention 32 8 512 64\n");
    printf("Parameters:\n");
    printf("  batch_size: Number of sequences to process in parallel\n");
    printf("  num_heads: Number of attention heads\n");
    printf("  seq_len: Length of input sequences\n");
    printf("  head_dim: Dimension of each attention head\n");
}

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

// Helper function to calculate shared memory size
__host__ size_t calculateSharedMemorySize(int head_dim) {
    return sizeof(float) * TILE_SIZE * head_dim * 3;  // For Q, K, and V
}

// Memory management structure
struct DeviceMemoryPool {
    char* base_ptr;
    float *d_input, *d_W_Q, *d_W_K, *d_W_V;
    float *d_Q, *d_K, *d_V, *d_output;
    float *d_m, *d_l;
    
    void allocate(size_t input_size, size_t qkv_size, size_t model_size, size_t softmax_size) {
        size_t total_size = input_size + (model_size * 3) + (qkv_size * 4) + (softmax_size * 2);
        CUDA_CHECK(cudaMalloc(&base_ptr, total_size));
        
        size_t offset = 0;
        d_input = (float*)(base_ptr + offset); offset += input_size;
        d_W_Q = (float*)(base_ptr + offset); offset += model_size;
        d_W_K = (float*)(base_ptr + offset); offset += model_size;
        d_W_V = (float*)(base_ptr + offset); offset += model_size;
        d_Q = (float*)(base_ptr + offset); offset += qkv_size;
        d_K = (float*)(base_ptr + offset); offset += qkv_size;
        d_V = (float*)(base_ptr + offset); offset += qkv_size;
        d_output = (float*)(base_ptr + offset); offset += qkv_size;
        d_m = (float*)(base_ptr + offset); offset += softmax_size;
        d_l = (float*)(base_ptr + offset);
    }
    
    void free() {
        CUDA_CHECK(cudaFree(base_ptr));
    }
};

__device__ void load_matrix_coalesced(float* shared_dest, const float* global_src, 
                                    int rows, int cols, int tid, int stride) {
    #pragma unroll 4
    for(int i = tid; i < rows * cols; i += stride) {
        int row = i / cols;
        int col = i % cols;
        shared_dest[col * rows + row] = global_src[row * cols + col];
    }
}

__global__ void initialize_random_matrix(float* matrix, int rows, int cols, 
                                       unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    curandState state;

    for (int i = idx; i < rows * cols; i += stride) {
        curand_init(seed, i, 0, &state);
        matrix[i] = 0.1f * curand_uniform(&state);
    }
}

__global__ void matmul_optimized(float* A, float* B, float* C, 
                                int M, int N, int K, cudaStream_t stream) {
    extern __shared__ float shared_mem[];
    float* shared_A = shared_mem;
    float* shared_B = shared_A + TILE_SIZE * TILE_SIZE;
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    float sum = 0.0f;
    
    for (int i = 0; i < (N + TILE_SIZE - 1) / TILE_SIZE; i++) {
        if (row < M && i * TILE_SIZE + tx < N)
            shared_A[ty * TILE_SIZE + tx] = A[row * N + i * TILE_SIZE + tx];
        else
            shared_A[ty * TILE_SIZE + tx] = 0.0f;
            
        if (i * TILE_SIZE + ty < N && col < K)
            shared_B[ty * TILE_SIZE + tx] = B[(i * TILE_SIZE + ty) * K + col];
        else
            shared_B[ty * TILE_SIZE + tx] = 0.0f;
            
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++)
            sum += shared_A[ty * TILE_SIZE + k] * shared_B[k * TILE_SIZE + tx];
            
        __syncthreads();
    }
    
    if (row < M && col < K)
        C[row * K + col] = sum;
}

__global__ void flash_attention_persistent(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    float* __restrict__ M,
    float* __restrict__ L,
    const int seq_len,
    const int head_dim,
    const int num_heads,
    const int batch_size) {
    
    extern __shared__ char shared_mem[];
    float* q_ptr = (float*)shared_mem;
    float* k_ptr = q_ptr + TILE_SIZE * head_dim;
    float* v_ptr = k_ptr + TILE_SIZE * head_dim;
    
    const int block_id = blockIdx.x;
    const int thread_id = threadIdx.x;
    const int heads_per_block = (num_heads + gridDim.x - 1) / gridDim.x;
    const int start_head = block_id * heads_per_block;
    const int end_head = min(start_head + heads_per_block, num_heads);
    
    float acc[MAX_HEAD_DIM];  // Register cache for accumulator
    __shared__ float scores[TILE_SIZE * TILE_SIZE];
    const float scale = 1.0f / sqrtf(head_dim);
    
    // Process multiple heads without kernel relaunch
    for(int b = 0; b < batch_size; b++) {
        for(int h = start_head; h < end_head; h++) {
            const float* Q_head = Q + (b * num_heads + h) * seq_len * head_dim;
            const float* K_head = K + (b * num_heads + h) * seq_len * head_dim;
            const float* V_head = V + (b * num_heads + h) * seq_len * head_dim;
            float* O_head = O + (b * num_heads + h) * seq_len * head_dim;
            
            for(int row = thread_id; row < seq_len; row += THREADS_PER_BLOCK) {
                float max_val = -INFINITY;
                float sum_exp = 0.0f;
                
                // Initialize accumulator
                #pragma unroll
                for(int k = 0; k < head_dim; k++) {
                    acc[k] = 0.0f;
                }
                
                // Load Q row into shared memory
                load_matrix_coalesced(q_ptr, Q_head + row * head_dim, 
                                    1, head_dim, thread_id, THREADS_PER_BLOCK);
                
                for(int tile = 0; tile < seq_len; tile += TILE_SIZE) {
                    __syncthreads();
                    
                    // Load K,V tiles
                    load_matrix_coalesced(k_ptr, K_head + tile * head_dim,
                                        TILE_SIZE, head_dim, thread_id, THREADS_PER_BLOCK);
                    load_matrix_coalesced(v_ptr, V_head + tile * head_dim,
                                        TILE_SIZE, head_dim, thread_id, THREADS_PER_BLOCK);
                    
                    __syncthreads();
                    
                    // Compute attention scores
                    int valid_cols = min(TILE_SIZE, seq_len - tile);
                    float local_max = -INFINITY;
                    
                    #pragma unroll 8
                    for(int j = 0; j < valid_cols; j++) {
                        float qk_sum = 0.0f;
                        #pragma unroll
                        for(int k = 0; k < head_dim; k++) {
                            qk_sum += q_ptr[k] * k_ptr[j * head_dim + k];
                        }
                        scores[j] = qk_sum * scale;
                        local_max = fmaxf(local_max, scores[j]);
                    }
                    
                    // Update global max
                    if(local_max > max_val) {
                        float scale_factor = expf(max_val - local_max);
                        #pragma unroll
                        for(int k = 0; k < head_dim; k++) {
                            acc[k] *= scale_factor;
                        }
                        sum_exp *= scale_factor;
                        max_val = local_max;
                    }
                    
                    // Compute attention weights and update accumulator
                    float local_sum = 0.0f;
                    #pragma unroll 8
                    for(int j = 0; j < valid_cols; j++) {
                        float exp_val = expf(scores[j] - max_val);
                        local_sum += exp_val;
                        #pragma unroll
                        for(int k = 0; k < head_dim; k++) {
                            acc[k] += exp_val * v_ptr[j * head_dim + k];
                        }
                    }
                    sum_exp += local_sum;
                }
                
                // Write output
                if(row < seq_len) {
                    M[row] = max_val;
                    L[row] = sum_exp;
                    float inv_sum = 1.0f / sum_exp;
                    #pragma unroll
                    for(int k = 0; k < head_dim; k++) {
                        O_head[row * head_dim + k] = acc[k] * inv_sum;
                    }
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 5) {
        printf("Error: Incorrect number of arguments\n");
        print_usage();
        return 1;
    }

    const int batch_size = atoi(argv[1]);
    const int num_heads = atoi(argv[2]);
    const int seq_len = atoi(argv[3]);
    const int head_dim = atoi(argv[4]);
    const int d_model = head_dim * num_heads;

    // Validation
    if (batch_size <= 0 || num_heads <= 0 || seq_len <= 0 || head_dim <= 0) {
        printf("Error: All parameters must be positive integers\n");
        print_usage();
        return 1;
    }

    if (head_dim > MAX_HEAD_DIM || head_dim % 4 != 0) {
        printf("Error: head_dim must be <= %d and a multiple of 4\n", MAX_HEAD_DIM);
        return 1;
    }
    
    printf("\nProblem Configuration:\n");
    printf("Batch size: %d\n", batch_size);
    printf("Number of attention heads: %d\n", num_heads);
    printf("Sequence length: %d\n", seq_len);
    printf("Head dimension: %d\n", head_dim);
    printf("Model dimension: %d\n", d_model);
    
    // Calculate sizes
    size_t input_size = batch_size * seq_len * d_model * sizeof(float);
    size_t qkv_size = batch_size * num_heads * seq_len * head_dim * sizeof(float);
    size_t model_size = d_model * d_model * sizeof(float);
    size_t softmax_size = batch_size * num_heads * seq_len * sizeof(float);
    
    // Create CUDA stream
    cudaStream_t compute_stream;
    CUDA_CHECK(cudaStreamCreate(&compute_stream));
    
    // Allocate pinned host memory
    float *h_input, *h_output, *h_Q, *h_m, *h_l;
    CUDA_CHECK(cudaMallocHost(&h_input, input_size));
    CUDA_CHECK(cudaMallocHost(&h_output, input_size));
    CUDA_CHECK(cudaMallocHost(&h_Q, qkv_size));
    CUDA_CHECK(cudaMallocHost(&h_m, softmax_size));
    CUDA_CHECK(cudaMallocHost(&h_l, softmax_size));
    
    // Allocate device memory pool
    DeviceMemoryPool d_mem;
    d_mem.allocate(input_size, qkv_size, model_size, softmax_size);
    
    // Initialize matrices
    int blockSize = 256;
    int numBlocks = (seq_len * d_model + blockSize - 1) / blockSize;
    initialize_random_matrix<<<numBlocks, blockSize, 0, compute_stream>>>(
        d_mem.d_input, seq_len, d_model, 1234ULL);
    initialize_random_matrix<<<numBlocks, blockSize, 0, compute_stream>>>(
        d_mem.d_W_Q, d_model, d_model, 1235ULL);
    initialize_random_matrix<<<numBlocks, blockSize, 0, compute_stream>>>(
        d_mem.d_W_K, d_model, d_model, 1236ULL);
    initialize_random_matrix<<<numBlocks, blockSize, 0, compute_stream>>>(
        d_mem.d_W_V, d_model, d_model, 1237ULL);
    
    // Create events for timing
    cudaEvent_t start, stop, proj_start, proj_stop, attn_start, attn_stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventCreate(&proj_start));
    CUDA_CHECK(cudaEventCreate(&proj_stop));
    CUDA_CHECK(cudaEventCreate(&attn_start));
    CUDA_CHECK(cudaEventCreate(&attn_stop));
    
    // Start timing
    CUDA_CHECK(cudaEventRecord(start, compute_stream));
    CUDA_CHECK(cudaEventRecord(proj_start, compute_stream));
    
    // Linear projections with optimized matmul
    dim3 proj_block(TILE_SIZE, TILE_SIZE);
    dim3 proj_grid((d_model + TILE_SIZE - 1) / TILE_SIZE, 
                   (seq_len + TILE_SIZE - 1) / TILE_SIZE);
    size_t matmul_shared_mem = 2 * TILE_SIZE * TILE_SIZE * sizeof(float);
    
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < num_heads; h++) {
            float* Q_head = d_mem.d_Q + (b * num_heads + h) * seq_len * head_dim;
            float* K_head = d_mem.d_K + (b * num_heads + h) * seq_len * head_dim;
            float* V_head = d_mem.d_V + (b * num_heads + h) * seq_len * head_dim;
            
            matmul_optimized<<<proj_grid, proj_block, matmul_shared_mem, compute_stream>>>(
                d_mem.d_input, d_mem.d_W_Q, Q_head, seq_len, d_model, head_dim);
            matmul_optimized<<<proj_grid, proj_block, matmul_shared_mem, compute_stream>>>(
                d_mem.d_input, d_mem.d_W_K, K_head, seq_len, d_model, head_dim);
            matmul_optimized<<<proj_grid, proj_block, matmul_shared_mem, compute_stream>>>(
                d_mem.d_input, d_mem.d_W_V, V_head, seq_len, d_model, head_dim);
        }
    }
    
    CUDA_CHECK(cudaEventRecord(proj_stop, compute_stream));
    
    // Flash attention computation
    CUDA_CHECK(cudaEventRecord(attn_start, compute_stream));
    
    size_t shared_mem_size = calculateSharedMemorySize(head_dim) + 
                            TILE_SIZE * TILE_SIZE * sizeof(float); // For scores
    
    printf("\nLaunching flash attention with configuration:\n");
    printf("Tile size: %d\n", TILE_SIZE);
    printf("Threads per block: %d\n", THREADS_PER_BLOCK);
    printf("Number of blocks: %d\n", OPTIMAL_BLOCKS);
    printf("Shared memory size: %zu bytes\n", shared_mem_size);
    
    flash_attention_persistent<<<OPTIMAL_BLOCKS, THREADS_PER_BLOCK, shared_mem_size, compute_stream>>>(
        d_mem.d_Q, d_mem.d_K, d_mem.d_V, d_mem.d_output, 
        d_mem.d_m, d_mem.d_l, seq_len, head_dim, num_heads, batch_size
    );
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaEventRecord(attn_stop, compute_stream));
    CUDA_CHECK(cudaEventRecord(stop, compute_stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    // Calculate timings
    float total_time = 0;
    float proj_time = 0;
    float attn_time = 0;
    CUDA_CHECK(cudaEventElapsedTime(&total_time, start, stop));
    CUDA_CHECK(cudaEventElapsedTime(&proj_time, proj_start, proj_stop));
    CUDA_CHECK(cudaEventElapsedTime(&attn_time, attn_start, attn_stop));
    
    printf("\nTiming Results:\n");
    printf("Linear Projections time: %.3f ms\n", proj_time);
    printf("Flash Attention time: %.3f ms\n", attn_time);
    printf("Total execution time: %.3f ms\n", total_time);
    
    // Verify results
    CUDA_CHECK(cudaMemcpyAsync(h_output, d_mem.d_output, qkv_size, 
                              cudaMemcpyDeviceToHost, compute_stream));
    CUDA_CHECK(cudaMemcpyAsync(h_Q, d_mem.d_Q, qkv_size, 
                              cudaMemcpyDeviceToHost, compute_stream));
    CUDA_CHECK(cudaMemcpyAsync(h_m, d_mem.d_m, softmax_size, 
                              cudaMemcpyDeviceToHost, compute_stream));
    CUDA_CHECK(cudaMemcpyAsync(h_l, d_mem.d_l, softmax_size, 
                              cudaMemcpyDeviceToHost, compute_stream));
    
    CUDA_CHECK(cudaStreamSynchronize(compute_stream));
    
    printf("\nOutput verification (first few values):\n");
    for (int i = 0; i < 5; i++) {
        printf("%.4f ", h_output[i]);
    }
    printf("\n");
    
    printf("\nQ matrix verification (first few values):\n");
    for (int i = 0; i < 5; i++) {
        printf("%.4f ", h_Q[i]);
    }
    printf("\n");
    
    printf("\nSoftmax statistics for first row:\n");
    printf("Max value (m): %.4f\n", h_m[0]);
    printf("Sum value (l): %.4f\n", h_l[0]);
    
    // Validate results
    bool valid_output = false;
    for (int i = 0; i < seq_len * head_dim; i++) {
        if (h_output[i] != 0.0f) {
            valid_output = true;
            break;
        }
    }
    
    if (!valid_output) {
        printf("\nWarning: Output appears to be all zeros. This might indicate a kernel execution problem.\n");
    }
    
    // Cleanup
    CUDA_CHECK(cudaFreeHost(h_input));
    CUDA_CHECK(cudaFreeHost(h_output));
    CUDA_CHECK(cudaFreeHost(h_Q));
    CUDA_CHECK(cudaFreeHost(h_m));
    CUDA_CHECK(cudaFreeHost(h_l));
    
    d_mem.free();
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(proj_start));
    CUDA_CHECK(cudaEventDestroy(proj_stop));
    CUDA_CHECK(cudaEventDestroy(attn_start));
    CUDA_CHECK(cudaEventDestroy(attn_stop));
    
    CUDA_CHECK(cudaStreamDestroy(compute_stream));
    
    return 0;
}