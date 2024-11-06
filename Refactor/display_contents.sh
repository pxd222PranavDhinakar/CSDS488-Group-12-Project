#!/bin/bash

# Function to show directory structure (excluding compiled files)
show_directory_structure() {
    echo "Directory Structure:"
    echo "."
    find . -mindepth 1 -type d | sort | sed -e 's/[^-][^\/]*\//  /g' -e 's/^/  /'
    
    # List files in each directory, excluding compiled files
    for dir in $(find . -type d | sort); do
        if [ -n "$(ls -A $dir)" ]; then
            echo
            echo "  $dir:"
            # List only source files and makefiles
            ls -1 "$dir" | grep -E '\.(c|h|asm|cu|cpp|hpp|cc|s|S|Makefile)$' | sed 's/^/    /'
        fi
    done
}

# Function to capture output
capture_output() {
    show_directory_structure

    echo -e "\nFile Contents:"

    # Use find with -type f to get only source files and makefiles
    find . -type f \( \
        -name "*.c" -o \
        -name "*.h" -o \
        -name "*.asm" -o \
        -name "*.cu" -o \
        -name "*.cpp" -o \
        -name "*.hpp" -o \
        -name "*.cc" -o \
        -name "Makefile" \
    \) -not -path "*/\.*" \
    -not -name "*.o" \
    -not -name "*.so" \
    -not -name "*.a" \
    -not -name "*.exe" \
    -not -name "basic_attention" \
    -not -name "flash_attention" \
    -print0 | sort -z | while IFS= read -r -d '' file; do
        echo -e "\n--- $file ---"
        echo "Location: $(dirname "$file")"
        echo "Contents:"
        cat "$file"
        echo -e "--- End of $file ---\n"
    done
}

# Display the output
capture_output

echo "Script execution complete."