#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/


echo "Starting server"
python3 server_new.py &
sleep 3  # Sleep for 3s to give the server enough time to start

# Start clients with given IDs
for i in `seq 0 1 2`; do
    echo "Starting client $i"
    python3 client_new.py ${i} &
done


# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait

# Test VFL system
python3 compute_test_metrics.py
