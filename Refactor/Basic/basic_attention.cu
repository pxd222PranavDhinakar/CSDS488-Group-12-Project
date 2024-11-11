#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <math.h>

// Configuration
#define BLOCK_SIZE 16 // Number of Blocks in Grid
#define MAX_THREADS 1024 // Maximum number of threads per block

// Add this function near the top after includes:
void print_usage() {
    printf("Usage: ./basic_attention <batch_size> <num_heads> <seq_len> <head_dim>\n");
    printf("Example: ./basic_attention 32 8 512 64\n");
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

// Kernel for matrix multiplication
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

// Kernel for softmax operation
__global__ void softmax(float* input, float* output, int seq_len) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < seq_len) {
        float max_val = input[row * seq_len];
        
        // Find max value in row
        for (int i = 1; i < seq_len; i++) {
            max_val = fmaxf(max_val, input[row * seq_len + i]);
        }
        
        // Compute exponentials and sum
        float sum = 0.0f;
        for (int i = 0; i < seq_len; i++) {
            float exp_val = expf(input[row * seq_len + i] - max_val);
            output[row * seq_len + i] = exp_val;
            sum += exp_val;
        }
        
        // Normalize
        for (int i = 0; i < seq_len; i++) {
            output[row * seq_len + i] /= sum;
        }
    }
}

// Kernel for attention scores computation
__global__ void attention_scores(float* Q, float* K, float* scores, 
                               int seq_len, int head_dim) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < seq_len && col < seq_len) {
        float sum = 0.0f;
        float scale = 1.0f / sqrtf(head_dim);
        
        for (int i = 0; i < head_dim; i++) {
            sum += Q[row * head_dim + i] * K[col * head_dim + i];
        }
        scores[row * seq_len + col] = sum * scale;
    }
}

// Initialize random matrix
__global__ void initialize_random_matrix(float* matrix, int rows, int cols, 
                                       unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    curandState state;

    for (int i = idx; i < rows * cols; i += stride) {
        curand_init(seed, i, 0, &state);
        matrix[i] = 0.1f * curand_uniform(&state);  // Scale down values for numerical stability
    }
}

// Function to print matrix dimensions and a few sample values
void print_matrix_info(const char* name, float* matrix, int rows, int cols) {
    printf("%s dimensions: %d x %d\n", name, rows, cols);
    printf("First few values: ");
    for (int i = 0; i < min(5, rows * cols); i++) {
        printf("%.4f ", matrix[i]);
    }
    printf("\n");
}

// Modify main function signature to accept parameters
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

    // Problem dimensions
    //const int batch_size = 1;
    //const int num_heads = 8;
    //const int seq_len = 512;
    //const int head_dim = 64;
    //const int d_model = head_dim * num_heads;
    
    printf("\nProblem Configuration:\n");
    printf("Batch size: %d\n", batch_size);
    printf("Number of attention heads: %d\n", num_heads);
    printf("Sequence length: %d\n", seq_len);
    printf("Head dimension: %d\n", head_dim);
    printf("Model dimension: %d\n", d_model);
    
    // Calculate sizes
    size_t input_size = batch_size * seq_len * d_model * sizeof(float);
    size_t qkv_size = batch_size * num_heads * seq_len * head_dim * sizeof(float);
    size_t score_size = batch_size * num_heads * seq_len * seq_len * sizeof(float);
    
    // Host memory allocation
    float *h_input = (float*)malloc(input_size);
    float *h_W_Q = (float*)malloc(d_model * d_model * sizeof(float));
    float *h_W_K = (float*)malloc(d_model * d_model * sizeof(float));
    float *h_W_V = (float*)malloc(d_model * d_model * sizeof(float));
    float *h_output = (float*)malloc(input_size);
    
    // Device memory allocation
    float *d_input, *d_W_Q, *d_W_K, *d_W_V;
    float *d_Q, *d_K, *d_V, *d_scores, *d_attn_output;
    
    CUDA_CHECK(cudaMalloc(&d_input, input_size));
    CUDA_CHECK(cudaMalloc(&d_W_Q, d_model * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W_K, d_model * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_W_V, d_model * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Q, qkv_size));
    CUDA_CHECK(cudaMalloc(&d_K, qkv_size));
    CUDA_CHECK(cudaMalloc(&d_V, qkv_size));
    CUDA_CHECK(cudaMalloc(&d_scores, score_size));
    CUDA_CHECK(cudaMalloc(&d_attn_output, qkv_size));
    
    // Initialize random values
    int blockSize = 256;
    int numBlocks = (seq_len * d_model + blockSize - 1) / blockSize;
    
    initialize_random_matrix<<<numBlocks, blockSize>>>(d_input, seq_len, d_model, 1234ULL);
    initialize_random_matrix<<<numBlocks, blockSize>>>(d_W_Q, d_model, d_model, 1235ULL);
    initialize_random_matrix<<<numBlocks, blockSize>>>(d_W_K, d_model, d_model, 1236ULL);
    initialize_random_matrix<<<numBlocks, blockSize>>>(d_W_V, d_model, d_model, 1237ULL);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEvent_t proj_start, proj_stop;
    cudaEvent_t attn_start, attn_stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventCreate(&proj_start));
    CUDA_CHECK(cudaEventCreate(&proj_stop));
    CUDA_CHECK(cudaEventCreate(&attn_start));
    CUDA_CHECK(cudaEventCreate(&attn_stop));
    
    // Start timing
    CUDA_CHECK(cudaEventRecord(start));
    
    // Linear projections timing
    CUDA_CHECK(cudaEventRecord(proj_start));
    
    dim3 projBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 projGrid((d_model + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                  (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < num_heads; h++) {
            float* Q_head = d_Q + (b * num_heads + h) * seq_len * head_dim;
            float* K_head = d_K + (b * num_heads + h) * seq_len * head_dim;
            float* V_head = d_V + (b * num_heads + h) * seq_len * head_dim;
            
            matmul<<<projGrid, projBlock>>>(d_input, d_W_Q, Q_head, seq_len, d_model, head_dim);
            matmul<<<projGrid, projBlock>>>(d_input, d_W_K, K_head, seq_len, d_model, head_dim);
            matmul<<<projGrid, projBlock>>>(d_input, d_W_V, V_head, seq_len, d_model, head_dim);
        }
    }
    
    CUDA_CHECK(cudaEventRecord(proj_stop));
    
    // Attention computation timing
    CUDA_CHECK(cudaEventRecord(attn_start));
    
    dim3 attnBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 attnGrid((seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                  (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < num_heads; h++) {
            float* Q_head = d_Q + (b * num_heads + h) * seq_len * head_dim;
            float* K_head = d_K + (b * num_heads + h) * seq_len * head_dim;
            float* V_head = d_V + (b * num_heads + h) * seq_len * head_dim;
            float* scores_head = d_scores + (b * num_heads + h) * seq_len * seq_len;
            float* output_head = d_attn_output + (b * num_heads + h) * seq_len * head_dim;
            
            // Compute attention scores
            attention_scores<<<attnGrid, attnBlock>>>(Q_head, K_head, scores_head, 
                                                    seq_len, head_dim);
            
            // Apply softmax
            softmax<<<seq_len, min(seq_len, MAX_THREADS)>>>(scores_head, scores_head, seq_len);
            
            // Multiply with values
            matmul<<<projGrid, projBlock>>>(scores_head, V_head, output_head, 
                                          seq_len, seq_len, head_dim);
        }
    }
    
    CUDA_CHECK(cudaEventRecord(attn_stop));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    // Calculate timing results
    float total_time = 0;
    float proj_time = 0;
    float attn_time = 0;
    CUDA_CHECK(cudaEventElapsedTime(&total_time, start, stop));
    CUDA_CHECK(cudaEventElapsedTime(&proj_time, proj_start, proj_stop));
    CUDA_CHECK(cudaEventElapsedTime(&attn_time, attn_start, attn_stop));
    
    printf("\nTiming Results:\n");
    printf("Linear Projections time: %.3f ms\n", proj_time);
    printf("Attention Computation time: %.3f ms\n", attn_time);
    printf("Total execution time: %.3f ms\n", total_time);
    
    // Copy results back for verification
    CUDA_CHECK(cudaMemcpy(h_output, d_attn_output, qkv_size, cudaMemcpyDeviceToHost));
    
    // Print a few output values for verification
    printf("\nOutput verification (first few values):\n");
    for (int i = 0; i < 5; i++) {
        printf("%.4f ", h_output[i]);
    }
    printf("\n");
    
    // Print some intermediate values for debugging
    float *h_Q = (float*)malloc(qkv_size);
    CUDA_CHECK(cudaMemcpy(h_Q, d_Q, qkv_size, cudaMemcpyDeviceToHost));
    printf("\nQ matrix verification (first few values):\n");
    for (int i = 0; i < 5; i++) {
        printf("%.4f ", h_Q[i]);
    }
    printf("\n");
    
    // Cleanup
    free(h_input);
    free(h_W_Q);
    free(h_W_K);
    free(h_W_V);
    free(h_output);
    free(h_Q);
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_W_Q));
    CUDA_CHECK(cudaFree(d_W_K));
    CUDA_CHECK(cudaFree(d_W_V));
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_scores));
    CUDA_CHECK(cudaFree(d_attn_output));
    
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(proj_start));
    CUDA_CHECK(cudaEventDestroy(proj_stop));
    CUDA_CHECK(cudaEventDestroy(attn_start));
    CUDA_CHECK(cudaEventDestroy(attn_stop));
    
    return 0;
}