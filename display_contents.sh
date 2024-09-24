#!/bin/bash

# Function to capture output
capture_output() {
    echo "Directory Structure:"
    tree -L 2

    echo -e "\nFile Contents:"

    find . -type f \( -name "*.c" -o -name "*.h" -o -name "*.asm" -o -name "Makefile" \) | while read file; do
        echo -e "\n--- $file ---"
        echo "Location: $(dirname "$file")"
        echo "Contents:"
        cat "$file"
        echo -e "--- End of $file ---\n"
    done
}

# Capture the output and copy it to clipboard
capture_output | tee >(pbcopy)

echo "Output has been copied to clipboard. You can now paste it anywhere."
