#!/bin/bash
# 11-27 stage H: from scratch, G rewards + stronger cadence (w15, swing 0.08-0.22), 3000 its.
export WANDB_MODE=disabled
T=/mnt/d/cc_work/duck-train-learn/targeted
cd ~/duck/xgoduck_rl || exit 1
SCR_STAGE=H .venv/bin/python "$T/duck_scratch.py" --env.scene.num-envs 4096 --agent.max-iterations 3000 --agent.logger tensorboard || exit 1
RUN=$(ls -dt logs/rsl_rl/xgoduck_real/*/ | head -1); CK=$(ls -t "$RUN"model_*.pt | head -1); NAME=$(basename "$RUN")
.venv/bin/python scripts/export.py Mjlab-Velocity-Flat-XgoDuck --checkpoint-file "$CK" --onnx-file "$RUN$NAME.onnx" || exit 1
cp "$RUN$NAME.onnx" /mnt/d/cc_work/duck-train-learn/my-models/xgoduck_walk_gaitH_$NAME.onnx
echo "== DONE $NAME"
