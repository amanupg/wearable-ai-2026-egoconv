#!/usr/bin/env bash
# Wait until CUDA actually initialises before running the payload.
#
# Four jobs died with "CUDA unknown error ... Setting the available devices to
# be zero" within 8-51s of starting. It is not node-specific (ins082 hosts both
# failures and a healthy job) -- it is an allocation race: the job lands on a
# node whose GPU is not yet released, CUDA sees no device, and the process dies
# instantly. Retrying inside the allocation costs seconds and rescues the job.
D=/insomnia001/depts/edu/users/au2327/wearable
for i in $(seq 1 30); do
  if "$D/.venv/bin/python" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() and torch.cuda.device_count()>0 else 1)" 2>/dev/null; then
    echo "cuda ready after ${i} attempt(s)"
    exec "$@"
  fi
  echo "cuda not ready (attempt $i), sleeping 20s"
  sleep 20
done
echo "CUDA never became available on $(hostname) after 10 min"
exit 1
