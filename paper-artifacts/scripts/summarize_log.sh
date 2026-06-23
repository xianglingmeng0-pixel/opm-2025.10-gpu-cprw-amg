#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 LOG_FILE [LOG_FILE ...]"
    exit 1
fi

for log in "$@"; do
    echo "==== ${log}"
    if [ ! -f "$log" ]; then
        echo "missing"
        continue
    fi
    grep -E "Number of MPI processes:|Threads per MPI process:|Number of timesteps:|Simulation time:|Assembly time:|Linear solve time:|Linear setup:|Props/update time:|Overall Newton Iterations:|Overall Linear Iterations:" "$log" | tail -12
done
