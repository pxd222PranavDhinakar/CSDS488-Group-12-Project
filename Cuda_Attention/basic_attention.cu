// basic_attention.cu
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h> 
#include <curand.h> s
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
    int seq_len = 4;  // Sequence length
    int d_model = 8;  // Model dimension
    int d_k = 8;      // Dimension of Q, K, V (often d_model / num_heads, but we'll keep it simple)
    
    // Allocate host memory
    float *input_seq = (float*)malloc(seq_len * d_model * sizeof(float));
    float *W_Q = (float*)malloc(d_model * d_k * sizeof(float));
    float *W_K = (float*)malloc(d_model * d_k * sizeof(float));
    float *W_V = (float*)malloc(d_model * d_k * sizeof(float));
    float *Q = (float*)malloc(seq_len * d_k * sizeof(float));
    float *K = (float*)malloc(seq_len * d_k * sizeof(float));
    float *V = (float*)malloc(seq_len * d_k * sizeof(float));
    float *output = (float*)malloc(seq_len * d_k * sizeof(float));
    
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

    // Perform linear projections
    dim3 projBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 projGrid((d_k + BLOCK_SIZE - 1) / BLOCK_SIZE, (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    linearProjection<<<projGrid, projBlock>>>(d_input_seq, d_W_Q, d_Q, seq_len, d_model, d_k);
    linearProjection<<<projGrid, projBlock>>>(d_input_seq, d_W_K, d_K, seq_len, d_model, d_k);
    linearProjection<<<projGrid, projBlock>>>(d_input_seq, d_W_V, d_V, seq_len, d_model, d_k);

    // Compute attention
    computeAttention(d_Q, d_K, d_V, d_output, seq_len, d_k);

    // Copy results back to host
    cudaMemcpy(input_seq, d_input_seq, seq_len * d_model * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Q, d_Q, seq_len * d_k * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(K, d_K, seq_len * d_k * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(V, d_V, seq_len * d_k * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(output, d_output, seq_len * d_k * sizeof(float), cudaMemcpyDeviceToHost);

    // Print results (you can use the printMatrix function from before)
    printf("Input Sequence:\n");
    printMatrix("Input", input_seq, seq_len, d_model);
    printf("Q Matrix:\n");
    printMatrix("Q", Q, seq_len, d_k);
    printf("K Matrix:\n");
    printMatrix("K", K, seq_len, d_k);
    printf("V Matrix:\n");
    printMatrix("V", V, seq_len, d_k);
    printf("Output Matrix:\n");
    printMatrix("Output", output, seq_len, d_k);

    // Free memory
    free(input_seq); free(W_Q); free(W_K); free(W_V); free(Q); free(K); free(V); free(output);
    cudaFree(d_input_seq); cudaFree(d_W_Q); cudaFree(d_W_K); cudaFree(d_W_V);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_output);

    return 0;
}