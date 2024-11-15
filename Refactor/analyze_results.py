def read_results(filename):
    data = []
    with open(filename, 'r') as f:
        # Skip header lines
        next(f)  # Skip timestamp
        next(f)  # Skip column headers
        for line in f:
            if line.strip():
                values = line.strip().split(',')
                data.append({
                    'config': f"{values[0]}/{values[1]}/{values[2]}/{values[3]}",
                    'proj_time': float(values[4]) if values[4] else 0,
                    'attn_time': float(values[5]) if values[5] else 0,
                    'total_time': float(values[6]) if values[6] else 0
                })
    return data

def main():
    # Read results
    basic_data = read_results('test_results/basic_results.txt')
    flash_data = read_results('test_results/flash_results.txt')

    # Generate text-based comparison
    with open('test_results/summary.txt', 'w') as f:
        f.write('Performance Summary\n\n')
        f.write('Configuration Comparisons:\n')
        f.write('-' * 80 + '\n')
        f.write(f"{'Config':20} {'Basic (ms)':15} {'Flash (ms)':15} {'Speedup':10}\n")
        f.write('-' * 80 + '\n')
        
        total_speedup = 0
        valid_configs = 0
        
        for basic, flash in zip(basic_data, flash_data):
            if basic['total_time'] > 0 and flash['total_time'] > 0:
                speedup = basic['total_time'] / flash['total_time']
                f.write(f"{basic['config']:20} {basic['total_time']:15.2f} "
                       f"{flash['total_time']:15.2f} {speedup:10.2f}x\n")
                total_speedup += speedup
                valid_configs += 1
        
        if valid_configs > 0:
            avg_speedup = total_speedup / valid_configs
            f.write('-' * 80 + '\n')
            f.write(f"\nAverage Speedup: {avg_speedup:.2f}x\n")

        # Add detailed timing breakdowns
        f.write('\nDetailed Timing Analysis:\n')
        f.write('-' * 80 + '\n')
        f.write('Basic Implementation:\n')
        for data in basic_data:
            f.write(f"\nConfig: {data['config']}\n")
            f.write(f"  Projection Time: {data['proj_time']:.2f} ms\n")
            f.write(f"  Attention Time:  {data['attn_time']:.2f} ms\n")
            f.write(f"  Total Time:      {data['total_time']:.2f} ms\n")

        f.write('\nFlash Implementation:\n')
        for data in flash_data:
            f.write(f"\nConfig: {data['config']}\n")
            f.write(f"  Projection Time: {data['proj_time']:.2f} ms\n")
            f.write(f"  Attention Time:  {data['attn_time']:.2f} ms\n")
            f.write(f"  Total Time:      {data['total_time']:.2f} ms\n")

if __name__ == '__main__':
    main()
