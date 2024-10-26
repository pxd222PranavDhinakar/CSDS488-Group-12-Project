# Compiler
NVCC = nvcc

# Compiler flags - all on one line
NVCC_FLAGS = -O3 -arch=sm_75 -std=c++14 --use_fast_math -lineinfo -diag-suppress 20044

# Additional flags for specific features - all on one line
CUDA_FLAGS = -DCUDA_HAS_FP16=1 -DCUDA_HAS_FP16_ATOMIC_ADD=1 -lcudart -lcublas

# Build target
TARGET = flash_attention

# Source files
SRC = flash_attention.cu

# Output binary
BIN = $(TARGET)

all: $(BIN)

$(BIN): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(CUDA_FLAGS) -o $@ $<

run: $(BIN)
	./$(BIN)

clean:
	rm -f $(BIN)

.PHONY: all run clean