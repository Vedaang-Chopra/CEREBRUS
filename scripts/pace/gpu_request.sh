#!/bin/bash
# Get the best available GPU job on PACE-ICE for a workload. Run on a PACE login or compute node.
#
#   gpu_request.sh status                              free GPUs per type + your queue
#   gpu_request.sh plan   <profile> [options]          estimated start of every candidate (submits nothing)
#   gpu_request.sh submit <profile> [options]          submit every candidate as a "race": the first job that
#                                                      starts cancels the others (same job name)
# Profiles (candidates in preference order; a candidate = gpu_type:count):
#   varloop        our 26B looped VLM, inference (one copy ~51 GB)     h200:1 h100:1 h100:2 a100:2 l40s:2 a40:2
#   varloop-train  post-training the looped VLM                         h200:2 h100:4 h200:1 h100:2
#   medium         7-14B models: full fine-tune / GRPO practice         h200:1 h100:1 h100:2 a100:2 l40s:2
#   small          <= 3B models: LoRA / GRPO / distillation practice   l40s:1 a40:1 a100:1 rtx_6000:1 h100:1 v100:1
# Options:
#   --gpus "h100:1,l40s:2"   override the candidate list
#   --blackwell              also allow rtx_pro_6000_blackwell (96 GB, sm_120; check your kernels first)
#   --time 16:00:00          wall time (partition max 16 h)
#   --cpus 8 --mem 64        per GPU (smaller requests start sooner)
#   --cmd "python x.py"      what the job runs (default: hold the GPU so you can ssh in)
#   --name tag               job name (default: profile name)
# A held job writes its node to ~/scratch/logs/gpu_<name>.node. From the Mac:
#   ssh -J pace <node>      (or add "Host atl1-*  ProxyJump pace" to your ssh config)
set -euo pipefail

PARTITIONS="ice-gpu,coc-gpu,ice-bw-gpu"
LOG_DIR="$HOME/scratch/logs"
declare -A PROFILES=(
  [varloop]="h200:1 h100:1 h100:2 a100:2 l40s:2 a40:2"
  [varloop-train]="h200:2 h100:4 h200:1 h100:2"
  [medium]="h200:1 h100:1 h100:2 a100:2 l40s:2"
  [small]="l40s:1 a40:1 a100:1 rtx_6000:1 h100:1 v100:1"
)

status() {
  echo "GPU type | total | in use"
  # one line per node (a node in several partitions is listed once per partition, hence sort -u on the node name)
  sinfo -N -h -p "$PARTITIONS" -O "NodeList:30,Gres:45,GresUsed:60" | sort -u | awk '
    { split($2, t, ":"); type = t[2]; total = t[3] + 0
      split($3, u, ":"); used = u[3] + 0
      tot[type] += total; use[type] += used }
    END { for (k in tot) printf "%-26s %5d %8d\n", k, tot[k], use[k] }' | sort
  echo; echo "Free GPUs can still be reserved for higher-priority jobs; 'plan' gives the scheduler's estimate."
  echo; squeue -u "$USER" -o "%.10i %.12j %.9T %.28b %.10L %.20S %R"
}

candidates() {  # profile -> candidate list
  local list="${GPUS_OVERRIDE:-${PROFILES[$1]:-}}"
  [[ -n "$list" ]] || { echo "unknown profile '$1' (choose: ${!PROFILES[*]})" >&2; exit 1; }
  [[ "$BLACKWELL" == 1 ]] && list="$list rtx_pro_6000_blackwell:1"
  echo "${list//,/ }"
}

sbatch_args() {  # candidate -> sbatch resource options
  local type="${1%%:*}" count="${1##*:}"
  echo "-p $PARTITIONS --gres=gpu:$type:$count -c $((CPUS * count)) --mem=$((MEM * count))G -t $TIME"
}

plan() {
  for c in $(candidates "$1"); do
    local when
    when=$(sbatch --test-only $(sbatch_args "$c") --wrap=true 2>&1 | grep -oE "start at [^ ]+|error.*" || true)
    printf "%-30s %s\n" "$c" "${when:-no estimate}"
  done
}

job_script() {  # the body every racing job runs
  cat <<EOF
#!/bin/bash
# Race: the first sibling to start cancels the pending ones; a later starter exits.
me=\$SLURM_JOB_ID
first=\$(squeue -u "\$USER" -h -n "$NAME" -t R -o "%i" | sort -n | head -1)
if [[ "\$first" != "\$me" ]]; then echo "sibling \$first already running; exiting"; exit 0; fi
squeue -u "\$USER" -h -n "$NAME" -t PD -o "%i" | xargs -r scancel
mkdir -p "$LOG_DIR"; hostname > "$LOG_DIR/gpu_$NAME.node"
echo "job \$me on \$(hostname) | \$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | paste -sd ';')"
${CMD:-sleep infinity}
EOF
}

submit() {
  mkdir -p "$LOG_DIR"
  local script; script=$(mktemp "$LOG_DIR/gpu_${NAME}_XXXX.sh"); job_script > "$script"
  for c in $(candidates "$1"); do
    sbatch $(sbatch_args "$c") -J "$NAME" -o "$LOG_DIR/gpu_${NAME}_%j.log" "$script" | sed "s/^/$c: /"
  done
  echo "job script: $script | node file once one starts: $LOG_DIR/gpu_$NAME.node"
}

ACTION="${1:-status}"; shift || true
PROFILE="${1:-}"; [[ $# -gt 0 ]] && shift
GPUS_OVERRIDE=""; BLACKWELL=0; TIME="16:00:00"; CPUS=8; MEM=64; CMD=""; NAME="${PROFILE:-gpu}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus) GPUS_OVERRIDE="$2"; shift 2 ;;
    --blackwell) BLACKWELL=1; shift ;;
    --time) TIME="$2"; shift 2 ;;
    --cpus) CPUS="$2"; shift 2 ;;
    --mem) MEM="$2"; shift 2 ;;
    --cmd) CMD="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 1 ;;
  esac
done

case "$ACTION" in
  status) status ;;
  plan) plan "$PROFILE" ;;
  submit) submit "$PROFILE" ;;
  *) echo "usage: gpu_request.sh status | plan <profile> | submit <profile> [options]" >&2; exit 1 ;;
esac
