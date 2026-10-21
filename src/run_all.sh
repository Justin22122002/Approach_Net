#!/usr/bin/env bash
# Runs the full training + evaluation pipeline from the src directory:
#   1. next-state FT-Transformer + ResNet101          (independent)
#   2. FT-Transformer + ResNet101                     -> results timestamp saved
#   3. constraint FT-Transformer + ResNet101          <- --pretrained-dir from 2.
#   4. ResNet101-GNN + FT-Transformer-GNN             -> results timestamp saved
#   5. constraint ResNet101-GNN + FT-Transformer-GNN  <- --pretrained-dir from 4.
#   6. rollouts of all models                         <- timestamps from 1.-5.
#   7. trajectory evaluation of all rollouts          <- timestamps from 1.-5.
#
# usage:  ./run_all_trainings.sh [--test] [--python python3] [--skip-eval] [--with-rnav] [--eval-only <pipeline_runs.csv>]
#   --test        dry run: every script only prints its resolved params and returns a dummy timestamp (no training)
#   --skip-eval   only training (steps 1-5)
#   --with-rnav   also run the RNAV optimizer rollout in step 6 (model independent, slow)
#   --eval-only   skip training, take the timestamps from an existing pipeline_runs_*.csv (steps 6-7)
set -euo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SRC_DIR"

PY="$(command -v python3 || command -v python)"
TEST_ARGS=()
SKIP_EVAL=0
WITH_RNAV=0
EVAL_ONLY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --test)      TEST_ARGS=(--test); shift ;;
    --python)    PY="$2"; shift 2 ;;
    --skip-eval) SKIP_EVAL=1; shift ;;
    --with-rnav) WITH_RNAV=1; shift ;;
    --eval-only) EVAL_ONLY="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

mkdir -p ml/results
LOG="ml/results/pipeline_runs_$(date +%Y-%m-%d_%H-%M-%S)${TEST_ARGS:+_test}.csv"

LAST_RUN=""

# run_step <module> [extra args...]  -> sets LAST_RUN to the results timestamp of that run
run_step() {
  local module="$1"; shift
  local result_file
  result_file="$(mktemp)"

  # pick --pretrained-dir out of the extra args (for the log)
  local pretrained="-"
  local args=("$@")
  for ((i = 0; i < ${#args[@]}; i++)); do
    if [[ "${args[i]}" == "--pretrained-dir" ]]; then
      pretrained="${args[i+1]}"
    fi
  done

  echo
  echo ">>> $PY -m $module $* ${TEST_ARGS[*]:-}"

  "$PY" -m "$module" --result-file "$result_file" "${TEST_ARGS[@]}" "$@"

  LAST_RUN="$(cat "$result_file")"
  rm -f "$result_file"

  if [[ -z "$LAST_RUN" ]]; then
    echo "ERROR: $module did not report a results folder" >&2
    exit 1
  fi
  echo "<<< $module  pretrained: $pretrained  ->  results: $LAST_RUN"
  echo "$STEP,$module,$pretrained,$LAST_RUN" >> "$LOG"
}

# run_eval <module> [extra args...]  -> no results timestamp, plots are only saved (--no-show)
run_eval() {
  local module="$1"; shift
  echo
  echo ">>> $PY -m $module $* --no-show ${TEST_ARGS[*]:-}"
  "$PY" -m "$module" --no-show "${TEST_ARGS[@]}" "$@"
  echo "<<< $module done"
}

# lookup <module>  -> results timestamp of that module from the --eval-only csv
lookup() {
  local ts
  ts="$(awk -F, -v m="$1" '$2 == m { ts = $4 } END { print ts }' "$EVAL_ONLY")"
  if [[ -z "$ts" ]]; then
    echo "ERROR: $1 not found in $EVAL_ONLY" >&2
    exit 1
  fi
  echo "$ts"
}

if [[ -z "$EVAL_ONLY" ]]; then
  echo "step,module,pretrained_timestamp,results_timestamp" > "$LOG"

  # 1. next-state models
  STEP=1
  run_step ml.train_ft_transformer_next_state_all_airports
  FT_NEXT_RUN="$LAST_RUN"

  run_step ml.train_resnet_101_next_state_all_airports
  RESNET_NEXT_RUN="$LAST_RUN"

  # 2. base models
  STEP=2
  run_step ml.train_ft_transformer_all_airports
  FT_RUN="$LAST_RUN"

  run_step ml.train_resnet_101_all_airports
  RESNET_RUN="$LAST_RUN"

  # 3. constraint models (warm start from 2.)
  STEP=3
  run_step ml.train_constraint_ft_transformer_cached_all_airports --pretrained-dir "$FT_RUN"
  FT_CON_RUN="$LAST_RUN"

  run_step ml.train_constraint_resnet_101_cached_all_airports --pretrained-dir "$RESNET_RUN"
  RESNET_CON_RUN="$LAST_RUN"

  # 4. GNN models
  STEP=4
  run_step ml.train_resnet_101_gnn_all_airports
  RESNET_GNN_RUN="$LAST_RUN"

  run_step ml.train_ft_transformer_gnn_all_airports
  FT_GNN_RUN="$LAST_RUN"

  # 5. constraint GNN models (warm start from 4.)
  STEP=5
  run_step ml.train_constraint_resnet_101_gnn_all_airports --pretrained-dir "$RESNET_GNN_RUN"
  RESNET_GNN_CON_RUN="$LAST_RUN"

  run_step ml.train_constraint_ft_transformer_gnn_all_airports --pretrained-dir "$FT_GNN_RUN"
  FT_GNN_CON_RUN="$LAST_RUN"

  echo
  echo "training finished, runs logged in $LOG:"
  column -s, -t < "$LOG" 2>/dev/null || cat "$LOG"
else
  echo "eval only, timestamps from $EVAL_ONLY"
  FT_NEXT_RUN="$(lookup ml.train_ft_transformer_next_state_all_airports)"
  RESNET_NEXT_RUN="$(lookup ml.train_resnet_101_next_state_all_airports)"
  FT_RUN="$(lookup ml.train_ft_transformer_all_airports)"
  RESNET_RUN="$(lookup ml.train_resnet_101_all_airports)"
  FT_CON_RUN="$(lookup ml.train_constraint_ft_transformer_cached_all_airports)"
  RESNET_CON_RUN="$(lookup ml.train_constraint_resnet_101_cached_all_airports)"
  RESNET_GNN_RUN="$(lookup ml.train_resnet_101_gnn_all_airports)"
  FT_GNN_RUN="$(lookup ml.train_ft_transformer_gnn_all_airports)"
  RESNET_GNN_CON_RUN="$(lookup ml.train_constraint_resnet_101_gnn_all_airports)"
  FT_GNN_CON_RUN="$(lookup ml.train_constraint_ft_transformer_gnn_all_airports)"
fi

if [[ "$SKIP_EVAL" -eq 1 ]]; then
  echo "evaluation skipped (--skip-eval)"
  exit 0
fi

# 6. rollouts
run_eval evaluation.main_data_ft_transformer_next_state            --run "$FT_NEXT_RUN"
run_eval evaluation.main_data_ft_transformer_next_state_constraint_cvxpy --run "$FT_NEXT_RUN"
run_eval evaluation.main_data_resnet_101_next_state                 --run "$RESNET_NEXT_RUN"
run_eval evaluation.main_data_resnet_101_next_state_constraint_cvxpy --run "$RESNET_NEXT_RUN"

run_eval evaluation.main_data_ft_transformer_model                  --run "$FT_RUN"
run_eval evaluation.main_data_ft_transformer_model_constraint_cvxpy --run "$FT_RUN" --constraint-run "$FT_CON_RUN"
run_eval evaluation.main_data_resnet_101_model                      --run "$RESNET_RUN"
run_eval evaluation.main_data_resnet_101_model_constraint_cvxpy     --run "$RESNET_RUN" --constraint-run "$RESNET_CON_RUN"

run_eval evaluation.main_data_resnet_101_gnn_model_and_constraint_model     --run "$RESNET_GNN_RUN" --constraint-run "$RESNET_GNN_CON_RUN"
run_eval evaluation.main_data_ft_transformer_gnn_model_and_constraint_model --run "$FT_GNN_RUN" --constraint-run "$FT_GNN_CON_RUN"

if [[ "$WITH_RNAV" -eq 1 ]]; then
  run_eval evaluation.main_data_optimizer_rnav
fi

# 7. trajectory evaluation
run_eval evaluation.trajectory_evaluator_all \
  --ft-action "$FT_RUN" --ft-action-constraint "$FT_CON_RUN" \
  --resnet-action "$RESNET_RUN" --resnet-action-constraint "$RESNET_CON_RUN" \
  --ft-next-state "$FT_NEXT_RUN" --resnet-next-state "$RESNET_NEXT_RUN" \
  --ft-gnn "$FT_GNN_RUN" --ft-gnn-constraint "$FT_GNN_CON_RUN" \
  --resnet-gnn "$RESNET_GNN_RUN" --resnet-gnn-constraint "$RESNET_GNN_CON_RUN"

echo
echo "pipeline finished"
[[ -f "$LOG" ]] && echo "runs: $LOG"
echo "summary: data/summary_experiments.csv"