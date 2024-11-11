#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <math.h>

// Configuration
#define TILE_SIZE 32        // Reduced tile size for better occupancy
#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32

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

__global__ void matmul(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < K) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += A[row * N + i] * B[i * K + col];
        }
        C[row * K + col] = sum;
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

__global__ void flash_attention_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    float* __restrict__ M,
    float* __restrict__ L,
    const int seq_len,
    const int head_dim) {
    
    extern __shared__ char shared_mem[];
    float* q_ptr = (float*)shared_mem;
    float* k_ptr = q_ptr + TILE_SIZE * head_dim;
    float* v_ptr = k_ptr + TILE_SIZE * head_dim;
    
    const int row_block = blockIdx.x * TILE_SIZE;
    const int tid = threadIdx.x;
    const float scale = 1.0f / sqrtf(head_dim);
    
    // Each thread handles one or more rows
    for (int row = tid; row < TILE_SIZE && row_block + row < seq_len; row += blockDim.x) {
        float max_val = -INFINITY;
        float sum_exp = 0.0f;
        float* acc = (float*)malloc(head_dim * sizeof(float));
        memset(acc, 0, head_dim * sizeof(float));
        
        // Load Q row into shared memory
        for (int k = 0; k < head_dim; k++) {
            q_ptr[row * head_dim + k] = Q[(row_block + row) * head_dim + k];
        }
        
        // Process K,V tiles
        for (int tile = 0; tile < seq_len; tile += TILE_SIZE) {
            __syncthreads();
            
            // Load K,V tile
            for (int i = tid; i < TILE_SIZE && tile + i < seq_len; i += blockDim.x) {
                for (int k = 0; k < head_dim; k++) {
                    k_ptr[i * head_dim + k] = K[(tile + i) * head_dim + k];
                    v_ptr[i * head_dim + k] = V[(tile + i) * head_dim + k];
                }
            }
            __syncthreads();
            
            // Compute attention scores and update running max
            float local_max = -INFINITY;
            float scores[TILE_SIZE];
            int valid_cols = min(TILE_SIZE, seq_len - tile);
            
            for (int j = 0; j < valid_cols; j++) {
                float qk_sum = 0.0f;
                for (int k = 0; k < head_dim; k++) {
                    qk_sum += q_ptr[row * head_dim + k] * k_ptr[j * head_dim + k];
                }
                scores[j] = qk_sum * scale;
                local_max = fmaxf(local_max, scores[j]);
            }
            
            // Update global max and rescale previous terms if needed
            if (local_max > max_val) {
                float scale_factor = expf(max_val - local_max);
                for (int k = 0; k < head_dim; k++) {
                    acc[k] *= scale_factor;
                }
                sum_exp *= scale_factor;
                max_val = local_max;
            }
            
            // Compute attention and update accumulators
            float local_sum = 0.0f;
            for (int j = 0; j < valid_cols; j++) {
                float exp_val = expf(scores[j] - max_val);
                local_sum += exp_val;
                for (int k = 0; k < head_dim; k++) {
                    acc[k] += exp_val * v_ptr[j * head_dim + k];
                }
            }
            sum_exp += local_sum;
        }
        
        // Write outputs
        if (row_block + row < seq_len) {
            M[row_block + row] = max_val;
            L[row_block + row] = sum_exp;
            float inv_sum = 1.0f / sum_exp;
            for (int k = 0; k < head_dim; k++) {
                O[(row_block + row) * head_dim + k] = acc[k] * inv_sum;
            }
        }
        
        free(acc);
    }
}

int main(int argc, char** argv) {
    // Parameter parsing
    if (argc != 5) {
        printf("Error: Incorrect number of arguments\n");
        print_usage();
        return 1;
    }

    // Parse command line arguments
    const int batch_size = atoi(argv[1]);
    const int num_heads = atoi(argv[2]);
    const int seq_len = atoi(argv[3]);
    const int head_dim = atoi(argv[4]);
    const int d_model = head_dim * num_heads;

    // Add parameter validation
    if (batch_size <= 0 || num_heads <= 0 || seq_len <= 0 || head_dim <= 0) {
        printf("Error: All parameters must be positive integers\n");
        print_usage();
        return 1;
    }

    // Additional validation specific to Flash implementation
    if (head_dim > TILE_SIZE * TILE_SIZE) {
        printf("Error: head_dim (%d) exceeds maximum supported size (%d)\n", 
               head_dim, TILE_SIZE * TILE_SIZE);
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
    size_t softmax_stats_size = batch_size * num_heads * seq_len * sizeof(float);
    
    // Host memory allocation
    float *h_input = (float*)malloc(input_size);
    float *h_W_Q = (float*)malloc(d_model * d_model * sizeof(float));
    float *h_W_K = (float*)malloc(d_model * d_model * sizeof(float));
    float *h_W_V = (float*)malloc(d_model * d_model * sizeof(float));
    float *h_output = (float*)malloc(input_size);
    
    // Device memory allocation
    float *d_input, *d_W_Q, *d_W_K, *d_W_V;
    float *d_Q, *d_K, *d_V, *d_output;
    float *d_m, *d_l;
    
    CUDA_CHECK(cudaMalloc(&d_input, input_size));
    CUDA_CHECK(cudaMalloc(&d_W_Q, d_model * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W_K, d_model * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W_V, d_model * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Q, qkv_size));
    CUDA_CHECK(cudaMalloc(&d_K, qkv_size));
    CUDA_CHECK(cudaMalloc(&d_V, qkv_size));
    CUDA_CHECK(cudaMalloc(&d_output, qkv_size));
    CUDA_CHECK(cudaMalloc(&d_m, softmax_stats_size));
    CUDA_CHECK(cudaMalloc(&d_l, softmax_stats_size));
    
    // Initialize matrices
    int blockSize = 256;
    int numBlocks = (seq_len * d_model + blockSize - 1) / blockSize;
    initialize_random_matrix<<<numBlocks, blockSize>>>(d_input, seq_len, d_model, 1234ULL);
    initialize_random_matrix<<<numBlocks, blockSize>>>(d_W_Q, d_model, d_model, 1235ULL);
    initialize_random_matrix<<<numBlocks, blockSize>>>(d_W_K, d_model, d_model, 1236ULL);
    initialize_random_matrix<<<numBlocks, blockSize>>>(d_W_V, d_model, d_model, 1237ULL);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Create events for timing
    cudaEvent_t start, stop, proj_start, proj_stop, attn_start, attn_stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventCreate(&proj_start));
    CUDA_CHECK(cudaEventCreate(&proj_stop));
    CUDA_CHECK(cudaEventCreate(&attn_start));
    CUDA_CHECK(cudaEventCreate(&attn_stop));
    
    // Start timing
    CUDA_CHECK(cudaEventRecord(start));
    
    // Linear projections
    CUDA_CHECK(cudaEventRecord(proj_start));
    dim3 proj_block(16, 16);
    dim3 proj_grid((d_model + 16 - 1) / 16, (seq_len + 16 - 1) / 16);
    
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < num_heads; h++) {
            float* Q_head = d_Q + (b * num_heads + h) * seq_len * head_dim;
            float* K_head = d_K + (b * num_heads + h) * seq_len * head_dim;
            float* V_head = d_V + (b * num_heads + h) * seq_len * head_dim;
            
            matmul<<<proj_grid, proj_block>>>(d_input, d_W_Q, Q_head, seq_len, d_model, head_dim);
            matmul<<<proj_grid, proj_block>>>(d_input, d_W_K, K_head, seq_len, d_model, head_dim);
            matmul<<<proj_grid, proj_block>>>(d_input, d_W_V, V_head, seq_len, d_model, head_dim);
        }
    }
    CUDA_CHECK(cudaEventRecord(proj_stop));
    
    // Flash attention computation
    CUDA_CHECK(cudaEventRecord(attn_start));
    
    size_t shared_mem_size = calculateSharedMemorySize(head_dim);
    int num_blocks = (seq_len + TILE_SIZE - 1) / TILE_SIZE;
    
    printf("\nLaunching flash attention with configuration:\n");
    printf("Tile size: %d\n", TILE_SIZE);
    printf("Threads per block: %d\n", THREADS_PER_BLOCK);
    printf("Number of blocks: %d\n", num_blocks);
    printf("Shared memory size: %zu bytes\n", shared_mem_size);
    
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < num_heads; h++) {
            float* Q_head = d_Q + (b * num_heads + h) * seq_len * head_dim;
            float* K_head = d_K + (b * num_heads + h) * seq_len * head_dim;
            float* V_head = d_V + (b * num_heads + h) * seq_len * head_dim;
            float* O_head = d_output + (b * num_heads + h) * seq_len * head_dim;
            float* M_head = d_m + (b * num_heads + h) * seq_len;
            float* L_head = d_l + (b * num_heads + h) * seq_len;
            
            flash_attention_kernel<<<num_blocks, THREADS_PER_BLOCK, shared_mem_size>>>(
                Q_head, K_head, V_head, O_head, M_head, L_head, seq_len, head_dim
            );
            CUDA_CHECK(cudaGetLastError());
        }
    }
    
    CUDA_CHECK(cudaEventRecord(attn_stop));
    CUDA_CHECK(cudaEventRecord(stop));
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
    CUDA_CHECK(cudaMemcpy(h_output, d_output, qkv_size, cudaMemcpyDeviceToHost));
    printf("\nOutput verification (first few values):\n");
    for (int i = 0; i < 5; i++) {
        printf("%.4f ", h_output[i]);
    }
    printf("\n");
    
    float *h_Q = (float*)malloc(qkv_size);
    float *h_m = (float*)malloc(softmax_stats_size);
    float *h_l = (float*)malloc(softmax_stats_size);
    
    CUDA_CHECK(cudaMemcpy(h_Q, d_Q, qkv_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_m, d_m, softmax_stats_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_l, d_l, softmax_stats_size, cudaMemcpyDeviceToHost));

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
    free(h_input);
    free(h_W_Q);
    free(h_W_K);
    free(h_W_V);
    free(h_output);
    free(h_Q);
    free(h_m);
    free(h_l);
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_W_Q));
    CUDA_CHECK(cudaFree(d_W_K));
    CUDA_CHECK(cudaFree(d_W_V));
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_m));
    CUDA_CHECK(cudaFree(d_l));
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(proj_start));
    CUDA_CHECK(cudaEventDestroy(proj_stop));
    CUDA_CHECK(cudaEventDestroy(attn_start));
    CUDA_CHECK(cudaEventDestroy(attn_stop));
    
    return 0;
}