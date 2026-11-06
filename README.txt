#环境配置（与参考环境 manifests/pip-freeze.txt 一致：Python 3.10、torch 2.6.0、vllm 0.8.3、ray 2.43.0、flash-attn 2.7.4.post1）
#environment-no-builds.yml 的 pip 部分装不上（torch==2.6.0+cu124 不在 PyPI、flash-attn 需要源码编译），改用 requirements-reference.txt
#torch 2.7 的 flash-attn 预编译包都需要 glibc 2.32，ModelArts 机器更旧；torch 2.6 这一版只需要 glibc 2.14
conda create -y -n nsr-ref -c conda-forge python=3.10

conda activate nsr-ref

python -m pip install --no-deps -r "/data/lihourun/NSR/manifests/requirements-reference.txt"
#也可以直接修复已有的 Python 3.11 环境 nsr：先 pip uninstall -y flash-attn，再在 nsr 里执行上面这条 pip install（会把 torch/vllm/ray/xformers 等改回参考版本）

python -c "import torch, flash_attn, flash_attn_2_cuda, vllm, ray, grpc; from grpc._cython import cygrpc; print(torch.__version__, flash_attn.__version__, vllm.__version__, ray.__version__)"


#模型下载
mkdir -p "/data/lihourun/NSR/cache/huggingface"

mkdir -p "/data/lihourun/NSR/models/Qwen2.5-Math-7B"

HF_HOME="/data/lihourun/NSR/cache/huggingface" \
hf download \
  Qwen/Qwen2.5-Math-7B \
  --local-dir "/data/lihourun/NSR/models/Qwen2.5-Math-7B"


#修改参数适配当前代码
python - <<'PY'
from pathlib import Path
import json

path = Path(
    "/data/lihourun/NSR/"
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
cd "/data/lihourun/NSR/NSR-main"

unset PYTORCH_CUDA_ALLOC_CONF

unset RAY_ADDRESS

#单机 8 卡：--nnodes 1（不加则是作者的 4 机 x 8 卡 = 32 卡布局，需要事先起好多机 Ray 集群）
NSR_ROOT="/data/lihourun/NSR" \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -u -m baseline_4gpu.launch run \
  --name DAPO7B-baseline-author32-run1 \
  --n-gpus 8 \
  --nnodes 1
