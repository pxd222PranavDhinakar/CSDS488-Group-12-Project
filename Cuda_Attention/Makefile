# Compiler
NVCC = nvcc

# Compiler flags
NVCC_FLAGS = -O3

# Build target
TARGET = basic_attention

all: $(TARGET)

$(TARGET): basic_attention.cu
        $(NVCC) $(NVCC_FLAGS) -o $@ $<

run: $(TARGET)
        ./$(TARGET)

clean:
        rm -f $(TARGET)

.PHONY: all run clean
