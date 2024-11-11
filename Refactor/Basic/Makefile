# Compiler
NVCC = nvcc

# Compiler flags
# Using -O3 for maximum optimization
# arch=sm_60 supports Pascal architecture and newer
# -use_fast_math enables fast math operations
NVCC_FLAGS = -O3 -arch=sm_60 -use_fast_math

# Additional flags for debug build
DEBUG_FLAGS = -g -G -DDEBUG

# Build target
TARGET = basic_attention

# Source file
SRC = basic_attention.cu

# Build both release and debug versions
all: $(TARGET) $(TARGET)_debug

# Release build
$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

# Debug build
$(TARGET)_debug: $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(DEBUG_FLAGS) -o $@ $<

# Run with default parameters
run: $(TARGET)
	./$(TARGET) 32 8 512 64

# Run debug version
run_debug: $(TARGET)_debug
	./$(TARGET)_debug 32 8 512 64

# Clean built files
clean:
	rm -f $(TARGET) $(TARGET)_debug

# Phony targets
.PHONY: all run run_debug clean