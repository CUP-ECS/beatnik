#!/bin/bash

# Script to format all .cpp and .hpp files in the fury library
# Usage: Run from build directory with: ../clangformat

# Get the script's directory (should be the project root when run as ../clangformat)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Formatting C++ files in ${SCRIPT_DIR}..."

# Find and format all .cpp and .hpp files in the SCRIPT_DIR directory,
# skipping any directory whose name contains "build".
find "${SCRIPT_DIR}" \
    -type d -name '*build*' -prune -o \
    -type f \( -name "*.cpp" -o -name "*.hpp" \) -print0 | \
while IFS= read -r -d '' file; do
    echo "Formatting: ${file}"
    clang-format -i "${file}"
done

echo "Done formatting!"
