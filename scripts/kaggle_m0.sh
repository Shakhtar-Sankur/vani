#!/bin/bash
# vani M0 on Kaggle (GPU T4 x2, Internet on). In a notebook cell:
#   !cd /tmp && rm -rf vani && git clone -q --depth 1 https://github.com/Shakhtar-Sankur/vani && bash vani/scripts/kaggle_m0.sh
# About an hour: OpenAI's models on GPU 0, Meta's and the Bengali setting on GPU 1, then an
# English control. QUICK=1 for a 10-minute check on 24 recordings.
set -eo pipefail
cd "$(dirname "$0")/.."
echo "== vani M0: vani $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
pip install -q "transformers==5.19.0" soundfile sentencepiece 2>&1 | grep -v -i "dependency resolver\|incompatible\|^$" | tail -2 || true
python -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__)"
LIMIT=$( [ -n "$QUICK" ] && echo "--limit 24" )
OUT=runs/m0
python -c "from vani import fleurs; list(fleurs.clips('or_in', limit=1))" >/dev/null  # download once, before both GPUs read it
CUDA_VISIBLE_DEVICES=0 python scripts/eval_m0.py --out $OUT $LIMIT --systems whisper-large-v3,whisper-large-v3-turbo,whisper-small > runs/gpu0.log 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 python scripts/eval_m0.py --out $OUT $LIMIT --systems mms-1b-all,seamless-m4t-v2-large,whisper-large-v3-as-bengali > runs/gpu1.log 2>&1 &
P1=$!
wait $P0 || { echo "GPU 0 failed:"; grep -E "Error|error|Traceback" runs/gpu0.log | tail -5; }
wait $P1 || { echo "GPU 1 failed:"; grep -E "Error|error|Traceback" runs/gpu1.log | tail -5; }
grep -h '^{' runs/gpu0.log runs/gpu1.log
echo "== English control: Whisper on 120 FLEURS English recordings"
CUDA_VISIBLE_DEVICES=0 python scripts/eval_m0.py --out $OUT --lang en_us --limit 120 --systems whisper-large-v3,whisper-small 2>&1 | grep '^{'
[ -d /kaggle/working ] && mkdir -p /kaggle/working/vani-m0 && cp -r $OUT /kaggle/working/vani-m0/
echo "== summary"
python scripts/summarize_m0.py $OUT
echo "== done: copy from '== vani M0' to here and send it back"
