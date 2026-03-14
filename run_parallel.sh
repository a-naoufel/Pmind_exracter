#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 START END NUM_JOBS"
    echo "Example: $0 0 999 10"
    exit 1
fi

START="$1"
END="$2"
JOBS="$3"

if ! [[ "$START" =~ ^[0-9]+$ ]]; then
    echo "Error: START must be an integer"
    exit 1
fi

if ! [[ "$END" =~ ^[0-9]+$ ]]; then
    echo "Error: END must be an integer"
    exit 1
fi

if ! [[ "$JOBS" =~ ^[0-9]+$ ]] || [ "$JOBS" -le 0 ]; then
    echo "Error: NUM_JOBS must be a positive integer"
    exit 1
fi

if [ "$START" -gt "$END" ]; then
    echo "Error: START must be <= END"
    exit 1
fi

TOTAL=$((END - START + 1))

if [ "$JOBS" -gt "$TOTAL" ]; then
    JOBS="$TOTAL"
fi

BASE_CHUNK=$((TOTAL / JOBS))
REMAINDER=$((TOTAL % JOBS))

current_start=$START
pids=()

for ((i=0; i<JOBS; i++)); do
    chunk_size=$BASE_CHUNK
    if [ "$i" -lt "$REMAINDER" ]; then
        chunk_size=$((chunk_size + 1))
    fi

    chunk_end=$((current_start + chunk_size - 1))

    start_fmt=$(printf "X%03d" "$current_start")
    end_fmt=$(printf "X%03d" "$chunk_end")

    echo "Starting job $((i+1))/$JOBS: python3 genai.py $start_fmt $end_fmt"
    python3 genai.py "$start_fmt" "$end_fmt" &

    pids+=($!)
    current_start=$((chunk_end + 1))
done

failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        failed=1
    fi
done

if [ "$failed" -ne 0 ]; then
    echo "Some jobs failed."
    exit 1
fi

echo "All jobs finished successfully."