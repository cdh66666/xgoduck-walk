#!/bin/bash
# usage: train_scratch.sh <iters>
export WANDB_MODE=disabled
T=/mnt/d/cc_work/duck-train-learn/targeted
cd ~/duck/xgoduck_rl || exit 1
.venv/bin/python "$T/duck_scratch.py" --env.scene.num-envs 4096 --agent.max-iterations "$1" --agent.logger tensorboard || exit 1
RUN=$(ls -dt logs/rsl_rl/xgoduck_real/*/ | head -1); CK=$(ls -t "$RUN"model_*.pt | head -1); NAME=$(basename "$RUN")
echo "RUN $NAME CK $(basename $CK)"
.venv/bin/python scripts/export.py Mjlab-Velocity-Flat-XgoDuck --checkpoint-file "$CK" --onnx-file "$RUN$NAME.onnx" || exit 1
OUT=/mnt/d/cc_work/duck-train-learn/my-models/xgoduck_walk_scratch_$NAME.onnx; cp "$RUN$NAME.onnx" "$OUT"
.venv/bin/python "$T/accept_speed.py" "scratch=$OUT" > "$T/accept_scratch.log" 2>&1
echo "== DONE: $OUT"; grep -a "^RESULT" "$T/accept_scratch.log"
