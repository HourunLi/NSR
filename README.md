#环境配置
conda env create \
  -n nsr \
  -f ".../NSR/manifests/environment-no-builds.yml"

conda activate nsr


#模型下载
mkdir -p ".../NSR/cache/huggingface"

mkdir -p ".../NSR/models/Qwen2.5-Math-7B"

HF_HOME=".../NSR/cache/huggingface" \
hf download \
  Qwen/Qwen2.5-Math-7B \
  --local-dir ".../NSR/models/Qwen2.5-Math-7B"


#修改参数适配当前代码
python - <<'PY'
from pathlib import Path
import json

path = Path(
    ".../NSR/"
    "models/Qwen2.5-Math-7B/config.json"
)

config = json.loads(path.read_text(encoding="utf-8"))
old_value = config.get("max_position_embeddings")

if old_value is None or old_value < 10240:
    config["max_position_embeddings"] = 32768
    path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

print(
    "max_position_embeddings:",
    old_value,
    "->",
    config["max_position_embeddings"],
)
PY


#提交任务
cd ".../NSR/NSR-main"

unset PYTORCH_CUDA_ALLOC_CONF

NSR_ROOT=".../NSR" \
RAY_ADDRESS=auto \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -u -m baseline_4gpu.launch run \
  --name DAPO7B-baseline-author32-run1 \
  --n-gpus 8
