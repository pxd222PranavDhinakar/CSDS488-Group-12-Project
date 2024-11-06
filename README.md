# Latest Build and Run Instructions
## Access Cluster
To access the Markov HPC cluster, use the following SSH command:
```bash
ssh -X [case_id]@markov.case.edu
```

## Clone Repository
Clone our repository inside of your HPC SSH instance.
```bash
git clone https://github.com/pxd222PranavDhinakar/CSDS488-Group-12-Project.git
```
## Allocating GPU Nodes

1. **Allocate a GPU Node**:
   - Use `srun` to allocate a GPU node and ensure `DISPLAY` is passed correctly:
   - Get rid of the `--x11` tag if you do not have x11 forwarding set up.

```sh
srun --x11 -p markov_gpu --gres=gpu:1 --mem=8gb --pty /bin/bash
```

## Build Environment

```bash
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1
```

## Code Setup

1. **Navigate to the Refactor Directory**:
```bash
cd Refactor
```

2. **Directory Structure**:
```
Refactor/
├── Basic/
│   ├── Makefile
│   └── basic_attention.cu
├── Flash/
│   ├── Makefile
│   └── flash_attention.cu
├── display_contents.sh
└── test_attention.sh
```

3. **Compile Basic Attention**:
```bash
cd Basic
make clean # (Optional) For getting rid of precompiled versions
make
```

4. **Compile Flash Attention**:
```bash
cd ../Flash
make clean # (Optional) For getting rid of precompiled versions
make
```

## Running the Implementations

Both implementations accept command-line parameters in the following format:
```bash
./[implementation]_attention <batch_size> <num_heads> <seq_len> <head_dim>
```

Example usage:
```bash
# For Basic Attention
cd Basic
./basic_attention 32 8 512 64

# For Flash Attention
cd ../Flash
./flash_attention 32 8 512 64
```

Parameters:
- `batch_size`: Number of sequences to process in parallel
- `num_heads`: Number of attention heads
- `seq_len`: Length of input sequences
- `head_dim`: Dimension of each attention head

## Automated Testing

To run automated tests across multiple configurations:
```bash
cd ..  # Make sure you're in the Refactor directory
chmod +x test_attention.sh
./test_attention.sh
```

The test script will:
- Compile both implementations
- Run multiple configurations
- Generate performance comparisons
- Save results in the `test_results` directory

## Output Files

After running the test script, you'll find the following in the `test_results` directory:
- `basic_results.txt`: Timing results for basic attention
- `flash_results.txt`: Timing results for flash attention
- `comparison.png`: Visual comparison plot
- `summary.txt`: Statistical summary of the results