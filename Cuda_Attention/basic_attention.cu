// basic_attention.cu
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h> 
#include <curand.h>
#include <curand_kernel.h>

#define BLOCK_SIZE 16

__global__ void matmul(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    
    if (row < M && col < K) {
        for (int i = 0; i < N; i++) {
            sum += A[row * N + i] * B[i * K + col];
        }
        C[row * K + col] = sum;
    }
}

__device__ void softmax(float* input, float* output, int N) {
    int tid = threadIdx.x;  // Use threadIdx for indexing inside a single block
    
    float max_val = input[0];  // No need to multiply tid * N since tid is per row
    for (int i = 1; i < N; i++) {
        max_val = fmaxf(max_val, input[i]);
    }

    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        float exp_val = expf(input[i] - max_val);
        output[i] = exp_val;
        sum += exp_val;
    }

    for (int i = 0; i < N; i++) {
        output[i] /= sum;
    }
}


__global__ void attention(float* Q, float* K, float* V, float* output, int seq_len, int d_model) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    extern __shared__ float s_mem[];
    float* QK = s_mem;
    float* softmax_out = &s_mem[seq_len * seq_len];
    
    if (row < seq_len && col < seq_len) {
        float sum = 0.0f;
        for (int i = 0; i < d_model; i++) {
            sum += Q[row * d_model + i] * K[i * seq_len + col];
        }
        QK[row * seq_len + col] = sum / sqrtf((float)d_model);
    }
    __syncthreads();
    
    if (row < seq_len) {
        softmax(&QK[row * seq_len], &softmax_out[row * seq_len], seq_len);
    }
    __syncthreads();
    
    if (row < seq_len && col < d_model) {
        float sum = 0.0f;
        for (int i = 0; i < seq_len; i++) {
            sum += softmax_out[row * seq_len + i] * V[i * d_model + col];
        }
        output[row * d_model + col] = sum;
    }
}

// Helper function to print matrix
void printMatrix(const char* name, float* matrix, int rows, int cols) {
    printf("%s (%dx%d):\n", name, rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}
__global__ void initializeRandomMatrix(float* matrix, int rows, int cols, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    curandState state;

    for (int i = idx; i < rows * cols; i += stride) {
        curand_init(seed, i, 0, &state);
        matrix[i] = curand_uniform(&state);
    }
}

__global__ void linearProjection(float* input, float* weight, float* output, int seq_len, int d_model, int d_k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < seq_len && col < d_k) {
        float sum = 0.0f;
        for (int i = 0; i < d_model; i++) {
            sum += input[row * d_model + i] * weight[i * d_k + col];
        }
        output[row * d_k + col] = sum;
    }
}

void computeAttention(float* Q, float* K, float* V, float* output, int seq_len, int d_k) {
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE, (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    size_t shared_mem_size = seq_len * seq_len * sizeof(float) * 2;
    attention<<<grid, block, shared_mem_size>>>(Q, K, V, output, seq_len, d_k);
}

int main() {
    // Updated dimensions to match flash attention
    int batch_size = 1;    // Add batch dimension
    int num_heads = 8;     // Add multiple heads
    int seq_len = 512;     // Sequence length
    int d_model = 512;     // Model dimension
    int d_k = 64;         // Head dimension (d_model / num_heads)

    printf("Problem Size:\n");
    printf("batch_size: %d\n", batch_size);
    printf("num_heads: %d\n", num_heads);
    printf("seq_len: %d\n", seq_len);
    printf("d_model: %d\n", d_model);
    printf("d_k: %d\n", d_k);

    // Rest of the allocations need to account for batch_size and num_heads
    size_t qkv_size = batch_size * num_heads * seq_len * d_k * sizeof(float);
    size_t model_size = batch_size * seq_len * d_model * sizeof(float);
    size_t weight_size = d_model * d_k * sizeof(float);

    // Create CUDA events for timing
    cudaEvent_t start, stop, proj_start, proj_stop, attn_start, attn_stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&proj_start);
    cudaEventCreate(&proj_stop);
    cudaEventCreate(&attn_start);
    cudaEventCreate(&attn_stop);

    // Start total timing
    cudaEventRecord(start);

    // Allocate host memory with updated sizes
    float *input_seq = (float*)malloc(model_size);
    float *W_Q = (float*)malloc(weight_size);
    float *W_K = (float*)malloc(weight_size);
    float *W_V = (float*)malloc(weight_size);
    float *Q = (float*)malloc(qkv_size);
    float *K = (float*)malloc(qkv_size);
    float *V = (float*)malloc(qkv_size);
    float *output = (float*)malloc(qkv_size);

    // Allocate device memory
    float *d_input_seq, *d_W_Q, *d_W_K, *d_W_V, *d_Q, *d_K, *d_V, *d_output;
    cudaMalloc(&d_input_seq, seq_len * d_model * sizeof(float));
    cudaMalloc(&d_W_Q, d_model * d_k * sizeof(float));
    cudaMalloc(&d_W_K, d_model * d_k * sizeof(float));
    cudaMalloc(&d_W_V, d_model * d_k * sizeof(float));
    cudaMalloc(&d_Q, seq_len * d_k * sizeof(float));
    cudaMalloc(&d_K, seq_len * d_k * sizeof(float));
    cudaMalloc(&d_V, seq_len * d_k * sizeof(float));
    cudaMalloc(&d_output, seq_len * d_k * sizeof(float));

    // Initialize with random values
    int blockSize = 256;
    int numBlocks = (seq_len * d_model + blockSize - 1) / blockSize;
    unsigned long long seed = 1234ULL;
    
    initializeRandomMatrix<<<numBlocks, blockSize>>>(d_input_seq, seq_len, d_model, seed);
    initializeRandomMatrix<<<numBlocks, blockSize>>>(d_W_Q, d_model, d_k, seed + 1);
    initializeRandomMatrix<<<numBlocks, blockSize>>>(d_W_K, d_model, d_k, seed + 2);
    initializeRandomMatrix<<<numBlocks, blockSize>>>(d_W_V, d_model, d_k, seed + 3);
    cudaDeviceSynchronize();  // Ensure initialization is complete

    // Time linear projections
    cudaEventRecord(proj_start);
    
    dim3 projBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 projGrid((d_k + BLOCK_SIZE - 1) / BLOCK_SIZE, (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    linearProjection<<<projGrid, projBlock>>>(d_input_seq, d_W_Q, d_Q, seq_len, d_model, d_k);
    linearProjection<<<projGrid, projBlock>>>(d_input_seq, d_W_K, d_K, seq_len, d_model, d_k);
    linearProjection<<<projGrid, projBlock>>>(d_input_seq, d_W_V, d_V, seq_len, d_model, d_k);
    
    cudaEventRecord(proj_stop);
    cudaEventSynchronize(proj_stop);
    float proj_time = 0;
    cudaEventElapsedTime(&proj_time, proj_start, proj_stop);
    printf("\nLinear Projections time: %.3f ms\n", proj_time);

    // Time attention computation
    cudaEventRecord(attn_start);
    
    computeAttention(d_Q, d_K, d_V, d_output, seq_len, d_k);
    
    cudaEventRecord(attn_stop);
    cudaEventSynchronize(attn_stop);
    float attn_time = 0;
    cudaEventElapsedTime(&attn_time, attn_start, attn_stop);
    printf("Attention Computation time: %.3f ms\n", attn_time);

    // Copy results back to host
    cudaMemcpy(input_seq, d_input_seq, seq_len * d_model * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Q, d_Q, seq_len * d_k * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(K, d_K, seq_len * d_k * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(V, d_V, seq_len * d_k * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(output, d_output, seq_len * d_k * sizeof(float), cudaMemcpyDeviceToHost);

    // Record total time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float total_time = 0;
    cudaEventElapsedTime(&total_time, start, stop);
    printf("Total GPU execution time (including memory transfers): %.3f ms\n", total_time);

    // Print results (you can use the printMatrix function from before)
    //printf("Input Sequence:\n");
    //printMatrix("Input", input_seq, seq_len, d_model);
    //printf("Q Matrix:\n");
    //printMatrix("Q", Q, seq_len, d_k);
    //printf("K Matrix:\n");
    //printMatrix("K", K, seq_len, d_k);
    //printf("V Matrix:\n");
    //printMatrix("V", V, seq_len, d_k);
    //printf("Output Matrix:\n");
    //printMatrix("Output", output, seq_len, d_k);

    // Free memory
    free(input_seq); free(W_Q); free(W_K); free(W_V); free(Q); free(K); free(V); free(output);
    cudaFree(d_input_seq); cudaFree(d_W_Q); cudaFree(d_W_K); cudaFree(d_W_V);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_output);

    // Cleanup timing events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(proj_start);
    cudaEventDestroy(proj_stop);
    cudaEventDestroy(attn_start);
    cudaEventDestroy(attn_stop);

    return 0;
}



    /*
    Outline of Matrix computations occurring here:
    1. Linear projections of input sequence to Q, K, V matrices: x * W_Q = Q, x * W_K = K, x * W_V = V
    2. Compute attention scores: Q * K^T = QK, softmax(QK) = attention_scores
    3. Compute output: attention_scores * V = output

    Outline of how the code structures the above computations, using CUDA threads and blocks:

    1. LINEAR PROJECTIONS (linearProjection kernel)
    - Dimensions:
        * Input: [seq_len × d_model]
        * Weight: [d_model × d_k]
        * Output: [seq_len × d_k]
    - Thread/Block Structure:
        * Block size: [BLOCK_SIZE × BLOCK_SIZE] (32×32 threads)
        * Grid size: Ceiling(d_k/32) × Ceiling(seq_len/32)
    - Each thread:
        * Computes one element of output matrix
        * Handles dot product of one input row with one weight column
    
    2. ATTENTION SCORES (attentionOptimized kernel)
    - Dimensions:
        * Q: [seq_len × d_k]
        * K: [seq_len × d_k]
        * Output: [seq_len × seq_len]
    - Thread/Block Structure:
        * Block size: [TILE_SIZE × TILE_SIZE] (32×32 threads)
        * Grid size: Ceiling(seq_len/32) × Ceiling(seq_len/32)
    - Shared Memory Usage:
        * s_Q: [TILE_SIZE × D_K] tile of Q matrix
        * s_K: [TILE_SIZE × D_K] tile of K matrix
    - Each thread block:
        * Loads tiles of Q and K into shared memory
        * Computes partial dot products for attention scores
        * Iterates over tiles until full row/column processed

    3. SOFTMAX (softmax kernel)
    - Dimensions:
        * Input/Output: [seq_len × seq_len]
    - Thread/Block Structure:
        * One thread block (256 threads) per sequence position
        * Grid size: seq_len blocks
    - Each thread block:
        * Finds maximum value in row
        * Computes exponentials
        * Sums exponentials
        * Normalizes final values

    4. FINAL OUTPUT (finalMatrixMultiply kernel)
    - Dimensions:
        * Attention Scores: [seq_len × seq_len]
        * V: [seq_len × d_k]
        * Output: [seq_len × d_k]
    - Thread/Block Structure:
        * Block size: [32 × 32] threads
        * Grid size: Ceiling(d_k/32) × Ceiling(seq_len/32)
    - Each thread:
        * Computes one element of final output matrix
        * Performs dot product of attention scores row with V column

    Memory Management:
    - Global Memory: Stores all input/output matrices
    - Shared Memory: Used in attention computation for tiled matrix multiply
    - Register Memory: Thread-local computations and accumulations

    Synchronization Points:
    1. After linear projections
    2. Within attention computation (between tile processing)
    3. After softmax computation
    4. After final matrix multiplication

    Performance Considerations:
    - Uses tiling to maximize shared memory usage in attention computation
    - Coalesced memory access patterns in linear projections
    - Balanced thread block sizes (32×32) for good occupancy
    - Row-wise parallelism in softmax for efficient reduction operations
    */