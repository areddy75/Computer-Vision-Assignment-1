#!/bin/bash
# Wait for any running queue to finish, then run another queue (keeps GPU jobs sequential).
cd "$(dirname "$0")/.."
while pgrep -f "run_queue.sh" > /dev/null; do sleep 20; done
exec scripts/run_queue.sh "$@"
