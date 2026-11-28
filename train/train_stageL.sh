#!/bin/bash
# 11-27 stage L: resume J, fall penalty -1000 (-20/fall), foot_slip -1.0, 800 its. Solver iters 4/ls 10 (bench 11-28: +38% steps/s, H acceptance unchanged within ~4%).
export WANDB_MODE=disabled
T=/mnt/d/cc_work/duck-train-learn/targeted
cd ~/duck/xgoduck_rl || exit 1
true
RB=logs/rsl_rl/xgoduck_real/2026-11-28_01-57-37_scratch/
CK=$(ls -t ${RB}model_*.pt | head -1); echo "resume $CK"
SIM_ITERS=4 SIM_LS=10 FALLL=1 SCR_STAGE=J .venv/bin/python "$T/duck_scratch.py" --env.scene.num-envs 4096 --agent.max-iterations 800 --agent.logger tensorboard \
  --agent.resume True --agent.load-run "$(basename $RB)" --agent.load-checkpoint "$(basename $CK)" || exit 1
RUN=$(ls -dt logs/rsl_rl/xgoduck_real/*/ | head -1); CK=$(ls -t "$RUN"model_*.pt | head -1); NAME=$(basename "$RUN")
.venv/bin/python scripts/export.py Mjlab-Velocity-Flat-XgoDuck --checkpoint-file "$CK" --onnx-file "$RUN$NAME.onnx" || exit 1
cp "$RUN$NAME.onnx" /mnt/d/cc_work/duck-train-learn/my-models/xgoduck_walk_speedL_$NAME.onnx
echo "== DONE $NAME"
