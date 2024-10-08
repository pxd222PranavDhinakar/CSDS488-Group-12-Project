# Language Model Implementations

This repository contains from-scratch PyTorch implementations of key components in modern language models, including the attention mechanism and a fully architected GPT model.

## Project Structure

```
.
├── Attention
│   └── Attention.ipynb
├── GPT
│   ├── GPT.ipynb
│   └── input.txt
├── Cuda_Attention
│   ├── basic_attention.cu
│   └── Makefile
└── display_contents.sh
```

## Contents

1. **Attention Mechanism**
   - Location: `Attention/Attention.ipynb`
   - Description: This Jupyter notebook contains a detailed implementation of the attention mechanism used in language models.

2. **GPT Model**
   - Location: `GPT/GPT.ipynb`
   - Description: This Jupyter notebook provides a complete implementation of a GPT (Generative Pre-trained Transformer) model architecture.
   - Input: `GPT/input.txt` - This file likely contains sample text for testing or training the GPT model.
2. **CUDA Attention Implementation**
   - Location: `Cuda_Attention/basic_attention.cu`
   - Description: This is a from scratch implementation of the computations needed to calculate an Attention map, given a token matrix.

4. **Utility Script**
   - `display_contents.sh`: A shell script to display the contents of the project directory.

## Getting Started

1. Ensure you have PyTorch installed in your Python environment.
2. Clone this repository to your local machine.
3. Navigate to the project directory.
4. Open the Jupyter notebooks in the `Attention` and `GPT` folders to explore the implementations.

## Usage

- To understand the attention mechanism, start with `Attention/Attention.ipynb`.
- To explore the full GPT model, open `GPT/GPT.ipynb`.
- You can use the provided `input.txt` file in the GPT folder for testing the model, or replace it with your own text data.
- The main work of this project is found in the `Cuda_Attention` folder where we implement the attention mechanism in Cuda.

# Basic Attention Cuda Kernel

## Environment Construction
```bash
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1
```

## GPU Node Allocation
```bash
srun --partition=markov_gpu --gres=gpu:1 --pty bash
```

## Building and Running

Navigate to the `Cuda_Attention` folder, there will be a `Makefile` in the same directory as your `basic_attention.cu` file.

1. Your directory structure should now look like this:
   ```
   [pxd222@classt08 Attention]$ ls
   basic_attention.cu  Makefile
   ```

2. To build the program, run:
   ```
   make
   ```

3. To build and run the program in one command, use:
   ```
   make run
   ```

4. To clean up (remove the compiled executable), use:
   ```
   make clean
   ```

This Makefile provides a more flexible and standard way to build and run your CUDA program. It's easily extendable if you need to add more source files or change compilation options in the future.

Once the script is compiled you can run it using the following command
```bash
./basic_attention
```


# CUDA Attention Mechanism Output Analysis

## Input Sequence

```markdown
Input (4x8):
0.145468 0.820181 0.550399 0.294830 0.914733 0.868979 0.321921 0.782857 
0.011302 0.285450 0.781606 0.233840 0.679064 0.282442 0.629903 0.121223 
0.433255 0.383079 0.513567 0.298722 0.416607 0.034491 0.049395 0.046656 
0.616587 0.648044 0.868518 0.401159 0.063146 0.497170 0.680894 0.935035 
```

- Dimensions: 4x8 (seq_len x d_model)
- Contains random values between 0 and 1
- Represents the initial encoded token sequence

## Transformed Matrices

### Q Matrix (Query)

```markdown
Q (4x8):
3.117584 2.386504 2.576346 2.423232 2.926047 2.808740 2.009905 2.901865 
1.836229 1.388084 1.614373 1.277430 1.663288 1.851346 1.283993 2.336187 
1.208117 1.075262 0.989760 0.955381 1.236640 1.325096 1.295062 1.433438 
2.412041 2.538443 2.541083 2.107105 3.044547 2.752587 1.913300 2.799440 
```

- Dimensions: 4x8 (seq_len x d_model)
- Values are transformed and scaled up compared to the input
- Represents the query transformation in the attention mechanism

### K Matrix (Key)

```markdown
K (4x8):
1.971101 3.234277 2.976773 2.266640 2.320701 2.273600 2.258418 1.857677 
1.322393 2.149621 1.667928 1.579631 1.341867 1.326408 1.661282 1.088120 
1.106765 1.834499 1.300379 0.884700 1.249388 1.300293 1.133756 0.919280 
1.671280 2.833535 2.706891 2.363745 2.466719 2.446401 3.070130 1.507630 
```

- Dimensions: 4x8 (seq_len x d_model)
- Values are transformed and scaled up compared to the input
- Represents the key transformation in the attention mechanism

### V Matrix (Value)

```markdown
V (4x8):
3.022253 2.729255 2.220316 2.510578 2.677460 2.897735 2.422355 2.413175 
2.026588 1.617990 1.311722 1.269090 1.675888 1.743236 1.736210 1.557071 
1.166417 0.944016 0.963545 1.087488 1.501791 1.481416 1.147066 1.008968 
2.856917 2.388366 2.406079 2.113227 2.754015 2.560301 3.044812 1.927372 
```

- Dimensions: 4x8 (seq_len x d_model)
- Values are transformed and scaled up compared to the input
- Represents the value transformation in the attention mechanism

## Output Matrix

```markdown
Output (4x8):
1.751328 1.408140 1.212213 1.230671 1.637505 1.675776 1.551623 1.384323 
1.758491 1.429290 1.249830 1.286225 1.688355 1.722806 1.570837 1.394722 
1.835845 1.504872 1.322277 1.358584 1.759902 1.792993 1.647459 1.446445 
1.762090 1.418793 1.220872 1.240326 1.646031 1.685167 1.560194 1.392288 
```

- Dimensions: 4x8 (seq_len x d_model)
- Represents the result of applying attention to the transformed Q, K, and V matrices

## Observations and Analysis:

1. **Linear Transformations**: The Q, K, and V matrices show the result of applying learned linear transformations to the input sequence. This is evident from the increased scale and changed distribution of values compared to the input.

2. **Distinct Transformations**: Q, K, and V matrices have different patterns of values, indicating that each serves a different role in the attention mechanism.

3. **Attention Computation**: The output matrix values are within a smaller range compared to Q, K, and V, suggesting the softmax operation's normalizing effect in the attention computation.

4. **Contextual Information**: The output rows are similar but not identical, indicating that each token in the sequence is attending to different parts of the input, but with some shared context.

5. **Dimensionality Preservation**: All matrices maintain the 4x8 shape, preserving the sequence length and model dimension throughout the process.

6. **Value Range**: Unlike the previous output, the transformed matrices and output are not bounded between 0 and 1, reflecting the effect of learned weight matrices and the full attention computation.

7. **Positional Trends**: There's a noticeable trend in the output where values in the middle columns (4-6) tend to be slightly higher, suggesting these dimensions might capture important features across the sequence.

This output demonstrates the full pipeline of the attention mechanism, from input transformation to the final attended output, providing a clear view of how information is processed and contextualized in the model.

# Mathematical Flow of Attention Mechanism

## 1. Input Representation

We start with an encoded sentence matrix $X \in \mathbb{R}^{s \times d}$, where:
- $s$ is the sequence length
- $d$ is the model dimension

In the script, this is represented by `input_seq` with dimensions `seq_len × d_model`.

## 2. Linear Projections to Q, K, and V

We project the input $X$ into Query (Q), Key (K), and Value (V) matrices using weight matrices $W_Q$, $W_K$, and $W_V$:

$$Q = XW_Q$$
$$K = XW_K$$
$$V = XW_V$$

Where $W_Q, W_K, W_V \in \mathbb{R}^{d \times d_k}$, and $d_k$ is the dimension of the query/key space.

In the script, this is done by the `linearProjection` kernel:

```cpp
__global__ void linearProjection(float* input, float* weight, float* output, int seq_len, int d_model, int d_k)
```

Mathematically, for each element:

$$(XW)_{ij} = \sum_{k=1}^d X_{ik} W_{kj}$$

## 3. Attention Calculation

The attention mechanism is computed as follows:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Let's break this down step by step:

### 3.1 Computing $QK^T$

First, we compute the dot product of Q and K^T:

$$(QK^T)_{ij} = \sum_{k=1}^{d_k} Q_{ik} K_{jk}$$

In the script, this is the first part of the `attention` kernel:

```cpp
float sum = 0.0f;
for (int i = 0; i < d_model; i++) {
    sum += Q[row * d_model + i] * K[i * seq_len + col];
}
```

### 3.2 Scaling

We scale the result by $\frac{1}{\sqrt{d_k}}$:

$$S = \frac{QK^T}{\sqrt{d_k}}$$

This is done in the script immediately after the dot product:

```cpp
QK[row * seq_len + col] = sum / sqrtf((float)d_model);
```

### 3.3 Softmax

We apply the softmax function to each row of S:

$$\text{softmax}(S_i)_j = \frac{e^{S_{ij}}}{\sum_{k=1}^s e^{S_{ik}}}$$

This is implemented in the `softmax` device function:

```cpp
__device__ void softmax(float* input, float* output, int N)
```

### 3.4 Multiplication with V

Finally, we multiply the softmax output with V:

$$A = \text{softmax}(S)V$$

Where $A_{ij} = \sum_{k=1}^s \text{softmax}(S_i)_k V_{kj}$

This is the last part of the `attention` kernel:

```cpp
float sum = 0.0f;
for (int i = 0; i < seq_len; i++) {
    sum += softmax_out[row * seq_len + i] * V[i * d_model + col];
}
output[row * d_model + col] = sum;
```

## 4. Output

The final output A is the attention matrix, where each row is a weighted sum of the values, with the weights determined by the compatibility of the query with the corresponding keys.

In the script, this final output is stored in the `output` matrix and has dimensions `seq_len × d_k`.

## Summary

The mathematical flow can be summarized as:

1. $X \in \mathbb{R}^{s \times d}$ (input)
2. $Q = XW_Q$, $K = XW_K$, $V = XW_V$ where $Q, K, V \in \mathbb{R}^{s \times d_k}$
3. $S = \frac{QK^T}{\sqrt{d_k}}$ where $S \in \mathbb{R}^{s \times s}$
4. $P = \text{softmax}(S)$ where $P \in \mathbb{R}^{s \times s}$
5. $A = PV$ where $A \in \mathbb{R}^{s \times d_k}$ (output)

This process allows the model to focus on different parts of the input sequence when producing each element of the output, which is a key feature of transformer models.


# CUDA Kernel Design for Attention Mechanism

## General CUDA Concepts

Before diving into specific kernels, let's review some key CUDA concepts:

- **Threads**: The smallest unit of parallel execution.
- **Blocks**: Groups of threads that can cooperate and share memory.
- **Grid**: The overall structure of blocks for a kernel launch.
- **Shared Memory**: Fast, on-chip memory shared by all threads in a block.
- **Global Memory**: Slower, off-chip memory accessible by all threads.

## 1. Matrix Initialization Kernel

```cpp
__global__ void initializeRandomMatrix(float* matrix, int rows, int cols, unsigned long long seed)
```

### Threading Layout:
- 1D grid of blocks, 1D blocks of threads.
- Each thread initializes multiple elements if necessary.

### Execution Strategy:
1. Calculate global thread ID: `idx = blockIdx.x * blockDim.x + threadIdx.x`
2. Use stride to handle cases where there are more elements than threads:
   ```cpp
   for (int i = idx; i < rows * cols; i += stride) {
       // Initialize element
   }
   ```
3. Use cuRAND to generate random numbers independently for each thread.

### Efficiency Considerations:
- Coalesced memory access pattern for writing to global memory.
- Independent random number generation avoids synchronization overhead.

## 2. Linear Projection Kernel

```cpp
__global__ void linearProjection(float* input, float* weight, float* output, int seq_len, int d_model, int d_k)
```

### Threading Layout:
- 2D grid of blocks, 2D blocks of threads.
- Each thread computes one element of the output matrix.

### Execution Strategy:
1. Calculate row and column indices:
   ```cpp
   int row = blockIdx.y * blockDim.y + threadIdx.y;
   int col = blockIdx.x * blockDim.x + threadIdx.x;
   ```
2. Each thread performs dot product for its assigned element:
   ```cpp
   for (int i = 0; i < d_model; i++) {
       sum += input[row * d_model + i] * weight[i * d_k + col];
   }
   ```

### Efficiency Considerations:
- 2D grid matches the 2D nature of matrix multiplication.
- Potential for shared memory optimization (not implemented in current version).
- Coalesced memory access for reading input and writing output.

## 3. Attention Kernel

```cpp
__global__ void attention(float* Q, float* K, float* V, float* output, int seq_len, int d_model)
```

### Threading Layout:
- 2D grid of blocks, 2D blocks of threads.
- Each thread computes elements for QK^T, softmax, and final output.

### Execution Strategy:
1. Compute QK^T:
   ```cpp
   if (row < seq_len && col < seq_len) {
       for (int i = 0; i < d_model; i++) {
           sum += Q[row * d_model + i] * K[i * seq_len + col];
       }
       QK[row * seq_len + col] = sum / sqrtf((float)d_model);
   }
   ```
2. Apply softmax (row-wise):
   ```cpp
   if (row < seq_len) {
       softmax(&QK[row * seq_len], &softmax_out[row * seq_len], seq_len);
   }
   ```
3. Compute final output:
   ```cpp
   if (row < seq_len && col < d_model) {
       for (int i = 0; i < seq_len; i++) {
           sum += softmax_out[row * seq_len + i] * V[i * d_model + col];
       }
       output[row * d_model + col] = sum;
   }
   ```

### Efficiency Considerations:
- Uses shared memory for intermediate results (QK and softmax output).
- Synchronization (`__syncthreads()`) ensures all threads complete each step before proceeding.
- Potential for further optimization by tiling and reducing global memory access.

## 4. Softmax Device Function

```cpp
__device__ void softmax(float* input, float* output, int N)
```

### Execution Strategy:
1. Find maximum value in the input array.
2. Compute exponentials and sum.
3. Normalize by dividing each element by the sum.

### Efficiency Considerations:
- Inline device function for better performance.
- Reduces numerical instability by subtracting the maximum value before exponentiation.

## Overall Execution Flow

1. Initialize matrices using `initializeRandomMatrix` kernel.
2. Perform linear projections for Q, K, and V using `linearProjection` kernel.
3. Compute attention using the `attention` kernel.

Each step is executed with an appropriate grid and block configuration to maximize parallelism and efficiency.

## Potential Optimizations

1. Use shared memory in `linearProjection` to reduce global memory access.
2. Implement tiling in the `attention` kernel to handle larger sequences.
3. Explore using Tensor Cores for matrix multiplications if available.
4. Implement multi-head attention for increased parallelism.
5. Use CUDA streams for concurrent kernel execution where possible.

These CUDA kernels are designed to parallelize the attention mechanism computations efficiently. Each kernel utilizes the GPU's massive parallelism by distributing work across many threads, with considerations for memory access patterns and shared resource utilization.

