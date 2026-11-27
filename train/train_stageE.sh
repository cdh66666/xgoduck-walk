#!/bin/bash
# Wait for stage B (train_scratch2.sh) to finish, then stage C: resume latest B checkpoint, 1000 its, turning.
export WANDB_MODE=disabled
T=/mnt/d/cc_work/duck-train-learn/targeted
cd ~/duck/xgoduck_rl || exit 1
true
RB=logs/rsl_rl/xgoduck_real/2026-11-26_18-52-37_scratch/
CK=$(ls -t ${RB}model_*.pt | head -1); echo "resume $CK"
SCR_STAGE=E .venv/bin/python "$T/duck_scratch.py" --env.scene.num-envs 4096 --agent.max-iterations 600 --agent.logger tensorboard \
  --agent.resume True --agent.load-run "$(basename $RB)" --agent.load-checkpoint "$(basename $CK)" || exit 1
RUN=$(ls -dt logs/rsl_rl/xgoduck_real/*/ | head -1); CK=$(ls -t "$RUN"model_*.pt | head -1); NAME=$(basename "$RUN")
.venv/bin/python scripts/export.py Mjlab-Velocity-Flat-XgoDuck --checkpoint-file "$CK" --onnx-file "$RUN$NAME.onnx" || exit 1
cp "$RUN$NAME.onnx" /mnt/d/cc_work/duck-train-learn/my-models/xgoduck_walk_turnE_$NAME.onnx
echo "== DONE $NAME"
