#!/usr/bin/env bash
# Health check for every Slurm job we've launched, in one call.
#
# Exists because Slurm failures are silent: a job that OOMs after 40 seconds
# disappears from squeue exactly like one that finished successfully, so polling
# `squeue` tells you nothing about whether work is actually happening. Three jobs
# died unnoticed (OOM, CUDA init error, wall-clock timeout) while squeue looked fine.
#
# Reports, per job: final state, elapsed vs requested wall time, max RSS, and the
# last error-ish line from its log. Run it a couple of minutes after any submit --
# OOM and CUDA-init failures surface almost immediately.
set -uo pipefail
D=/insomnia001/depts/edu/users/au2327/wearable

echo "=== QUEUE NOW ==="
squeue -u au2327 -o "%.10i %.13j %.7P %.2t %.8M %.10L %R" 2>/dev/null || echo "(none)"

echo
echo "=== LAST 10 JOBS (sacct) ==="
sacct -u au2327 --starttime now-1days \
  --format=JobID%12,JobName%14,State%14,Elapsed,Timelimit,MaxRSS,ExitCode \
  -X 2>/dev/null | head -14

echo
echo "=== LOG TAILS (errors only) ==="
for f in "$D"/logs_*.txt; do
  [ -f "$f" ] || continue
  # Only surface logs that look unhealthy, so a clean run stays quiet.
  hit=$(tr -d '\r' < "$f" | grep -iE "error|Traceback|CUDA|OutOfMemory|CANCELLED|TIME LIMIT|Killed" \
        | grep -viE "expandable_segments|error_|no error" | tail -2)
  if [ -n "$hit" ]; then
    echo "--- $(basename "$f")"
    echo "$hit" | sed 's/^/    /'
  fi
done

echo
echo "=== PROGRESS MARKERS ==="
for f in "$D"/logs_*.txt; do
  [ -f "$f" ] || continue
  last=$(tr -d '\r' < "$f" | grep -E "^  [0-9]+/|step [0-9]+/|retrieved for|done:|saved adapter" | tail -1)
  [ -n "$last" ] && printf "  %-26s %s\n" "$(basename "$f")" "$last"
done
