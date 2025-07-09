#!/bin/bash
set -e  # Exit the script if any statement returns a non-true return value

# Set workspace directory from env or default
WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"

# Call Python handler
call_python_handler() {
    echo "Calling Python handler.py..."
    python $WORKSPACE_DIR/handler.py
}

# ---------------------------------------------------------------------------- #
#                               Main Program                                   #
# ---------------------------------------------------------------------------- #

echo "Pod Started"

call_python_handler

echo "Start script(s) finished"

sleep infinity
