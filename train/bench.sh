#!/bin/bash
# bench.sh <num_envs> <iters> <tag>: resume H, stage I config, print per-iteration timing
export WANDB_MODE=disabled
T=/mnt/d/cc_work/duck-train-learn/targeted
cd ~/duck/xgoduck_rl || exit 1
RB=logs/rsl_rl/xgoduck_real/2026-11-27_16-11-55_scratch/
SYNCFIX=${SYNCFIX:-} SCR_STAGE=I SIM_ITERS=${SIM_ITERS:-} SIM_LS=${SIM_LS:-} .venv/bin/python "$T/duck_scratch.py" --env.scene.num-envs $1 --agent.max-iterations $2 --agent.logger tensorboard \
  --agent.resume True --agent.load-run "$(basename $RB)" --agent.load-checkpoint model_2999.pt > $T/bench_$3.log 2>&1
RUN=$(ls -dt logs/rsl_rl/xgoduck_real/*/ | head -1); rm -rf "$RUN"
grep -a "Steps per second\|Collection time\|Learning time" $T/bench_$3.log | tail -$((3*($2-5))) | awk '/Steps/{s+=$4;n++} /Collection/{c+=$3} /Learning time/{l+=$3} END{printf "TAG '$3' envs '$1' steps/s %.0f coll %.2f learn %.2f (n=%d)\n",s/n,c/n,l/n,n}'
grep -a -i "out of memory\|Traceback" $T/bench_$3.log | head -3
