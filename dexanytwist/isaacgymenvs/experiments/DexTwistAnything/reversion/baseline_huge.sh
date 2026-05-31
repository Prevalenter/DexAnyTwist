#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACGYM_ENVS_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPO_ROOT="$(cd "$ISAACGYM_ENVS_DIR/../.." && pwd)"

export PYTHONPATH="$REPO_ROOT/dexanytwist${PYTHONPATH:+:$PYTHONPATH}"

CUDA_DEVICE="${CUDA_DEVICE:-cuda:0}"
calculate_cls_metric=False
dataset_name="dataset_all"
num_envs=16384
headless=True
force_render=False
test=False
is_dr=True
seed=100
model_path=""

mode="${1:-}"
case "$mode" in
    --test_metric)
        shift
        if [ $# -lt 1 ]; then
            echo "error: --test_metric requires a checkpoint path" >&2
            exit 1
        fi
        echo "Metric test mode enabled on ${CUDA_DEVICE}"
        model_path="$1"
        if [[ "$model_path" != /* ]]; then
            model_path="$(realpath -m "$model_path")"
        fi
        shift
        num_envs=16384
        headless=True
        force_render=False
        test=True
        is_dr=True
        calculate_cls_metric=True
        dataset_name="train_all"
        ;;
    "")
        ;;
    *)
        if [[ "$mode" == --* ]]; then
            echo "error: unknown mode $mode" >&2
            exit 1
        fi
        ;;
esac

cd "$ISAACGYM_ENVS_DIR"

cmd=(
    conda run --no-capture-output -n isaac_lx python train.py
    task=dexanytwist/DexAnyTwist
    train=dexanytwist/DexAnyTwistPPO
    task.env.asymmetric_observations=True
    task.env.asym_state_fingertip_state=True
    task.env.asym_state_dr_params=False
    "task.dataset_name=${dataset_name}"
    "headless=${headless}"
    "force_render=${force_render}"
    task.task.schedule=None
    "task.task.randomize=${is_dr}"
    "task.env.calculate_cls_metric=${calculate_cls_metric}"
    "num_envs=${num_envs}"
    "sim_device=${CUDA_DEVICE}"
    "rl_device=${CUDA_DEVICE}"
    "test=${test}"
    "seed=${seed}"
    train.params.config.save_best_after=50
    "train.params.network.mlp.units=[4096]"
    train.params.network.rnn.units=1024
)

if [ -n "$model_path" ]; then
    cmd+=("checkpoint=${model_path}")
fi

cmd+=("$@")

exec "${cmd[@]}"
