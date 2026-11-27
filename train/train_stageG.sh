#!/bin/bash
# 11-27 stage G: resume E, cadence reward + short swing window, 1000 its.
export WANDB_MODE=disabled
T=/mnt/d/cc_work/duck-train-learn/targeted
cd ~/duck/xgoduck_rl || exit 1
true
RB=logs/rsl_rl/xgoduck_real/2026-11-26_23-12-37_scratch/
CK=$(ls -t ${RB}model_*.pt | head -1); echo "resume $CK"
SCR_STAGE=G .venv/bin/python "$T/duck_scratch.py" --env.scene.num-envs 4096 --agent.max-iterations 1000 --agent.logger tensorboard \
  --agent.resume True --agent.load-run "$(basename $RB)" --agent.load-checkpoint "$(basename $CK)" || exit 1
RUN=$(ls -dt logs/rsl_rl/xgoduck_real/*/ | head -1); CK=$(ls -t "$RUN"model_*.pt | head -1); NAME=$(basename "$RUN")
.venv/bin/python scripts/export.py Mjlab-Velocity-Flat-XgoDuck --checkpoint-file "$CK" --onnx-file "$RUN$NAME.onnx" || exit 1
cp "$RUN$NAME.onnx" /mnt/d/cc_work/duck-train-learn/my-models/xgoduck_walk_gaitG_$NAME.onnx
echo "== DONE $NAME"
