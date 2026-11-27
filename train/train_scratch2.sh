#!/bin/bash
# Stage A from scratch (1000 it), then stage B resume (1500 it). Snapshots every 250 its by snap_loop.
export WANDB_MODE=disabled
T=/mnt/d/cc_work/duck-train-learn/targeted
cd ~/duck/xgoduck_rl || exit 1
SCR_STAGE=A .venv/bin/python "$T/duck_scratch.py" --env.scene.num-envs 4096 --agent.max-iterations 1000 --agent.logger tensorboard || exit 1
RA=$(ls -dt logs/rsl_rl/xgoduck_real/*/ | head -1); echo "STAGE_A $RA"
SCR_STAGE=B .venv/bin/python "$T/duck_scratch.py" --env.scene.num-envs 4096 --agent.max-iterations 1500 --agent.logger tensorboard \
  --agent.resume True --agent.load-run "$(basename $RA)" --agent.load-checkpoint "$(ls -t ${RA}model_*.pt | head -1 | xargs basename)" || exit 1
RUN=$(ls -dt logs/rsl_rl/xgoduck_real/*/ | head -1); CK=$(ls -t "$RUN"model_*.pt | head -1); NAME=$(basename "$RUN")
echo "STAGE_B $RUN $CK"
.venv/bin/python scripts/export.py Mjlab-Velocity-Flat-XgoDuck --checkpoint-file "$CK" --onnx-file "$RUN$NAME.onnx" || exit 1
cp "$RUN$NAME.onnx" /mnt/d/cc_work/duck-train-learn/my-models/xgoduck_walk_scratch2_$NAME.onnx
echo "== DONE"
