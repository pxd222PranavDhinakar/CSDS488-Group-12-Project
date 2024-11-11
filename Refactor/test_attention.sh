#!/bin/bash

# Directory setup
BASIC_DIR="Basic"
FLASH_DIR="Flash"
OUTPUT_DIR="test_results"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Test configurations
CONFIGS=(
    "16 4 256 32"    # Small configuration
    "32 8 512 64"    # Medium configuration
    "64 12 1024 128" # Large configuration
)

# Function to run tests for a specific implementation
run_tests() {
    local impl_dir=$1
    local impl_name=$2
    local output_file="$OUTPUT_DIR/${impl_name}_results.txt"
    
    echo "Running tests for $impl_name implementation..."
    echo "Results for $impl_name implementation - $(date)" > $output_file
    echo "batch_size,num_heads,seq_len,head_dim,proj_time,attn_time,total_time" >> $output_file
    
    cd $impl_dir
    make clean && make
    
    for config in "${CONFIGS[@]}"; do
        echo "Testing configuration: $config"
        read -r batch_size num_heads seq_len head_dim <<< "$config"
        
        # Run the test and extract timing information
        output=$(./${impl_name}_attention $batch_size $num_heads $seq_len $head_dim)
        
        # Extract timing information using grep and awk
        proj_time=$(echo "$output" | grep "Linear Projections time" | awk '{print $4}')
        attn_time=$(echo "$output" | grep -E "(Attention|Flash Attention) time" | awk '{print $4}')
        total_time=$(echo "$output" | grep "Total execution time" | awk '{print $4}')
        
        # Save results
        echo "$batch_size,$num_heads,$seq_len,$head_dim,$proj_time,$attn_time,$total_time" >> ../$output_file
    done
    cd ..
}

# Run tests for both implementations
run_tests $BASIC_DIR "basic"
run_tests $FLASH_DIR "flash"

# Compare results
echo "Generating comparison report..."
python3 - <<END
import pandas as pd
import matplotlib.pyplot as plt

# Read results
basic_df = pd.read_csv('$OUTPUT_DIR/basic_results.txt', skiprows=1)
flash_df = pd.read_csv('$OUTPUT_DIR/flash_results.txt', skiprows=1)

# Create comparison plot
plt.figure(figsize=(12, 6))
x = range(len(basic_df))
width = 0.35

plt.bar([i - width/2 for i in x], basic_df['total_time'], width, label='Basic Attention')
plt.bar([i + width/2 for i in x], flash_df['total_time'], width, label='Flash Attention')

plt.xlabel('Configuration')
plt.ylabel('Total Time (ms)')
plt.title('Performance Comparison: Basic vs Flash Attention')
plt.legend()

configs = [f'{r.batch_size}/{r.num_heads}/{r.seq_len}/{r.head_dim}' for _, r in basic_df.iterrows()]
plt.xticks(x, configs, rotation=45)

plt.tight_layout()
plt.savefig('$OUTPUT_DIR/comparison.png')

# Generate summary statistics
with open('$OUTPUT_DIR/summary.txt', 'w') as f:
    f.write('Performance Summary\n\n')
    f.write('Average speedup: {:.2f}x\n'.format(
        basic_df['total_time'].mean() / flash_df['total_time'].mean()))
    f.write('\nDetailed Statistics:\n')
    f.write('\nBasic Attention:\n')
    f.write(basic_df.describe().to_string())
    f.write('\n\nFlash Attention:\n')
    f.write(flash_df.describe().to_string())
END

echo "Testing complete! Results are in the $OUTPUT_DIR directory"
