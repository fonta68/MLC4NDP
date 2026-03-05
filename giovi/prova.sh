#!/bin/bash

# Usage: ./myscript.sh [arg1] [arg2]

# If $1 is empty/unset, default to "defaultValue1"
arg1="${1:-defaultValue1}"

# If $2 is empty/unset, default to "defaultValue2"
arg2="${2:-defaultValue2}"

echo "Argument 1 is: $arg1"
echo "Argument 2 is: $arg2"

