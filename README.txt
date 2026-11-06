#环境配置（与参考环境 manifests/pip-freeze.txt 一致：Python 3.10、torch 2.6.0、vllm 0.8.3、ray 2.43.0、flash-attn 2.7.4.post1）
#environment-no-builds.yml 的 pip 部分装不上（torch==2.6.0+cu124 不在 PyPI、flash-attn 需要源码编译），改用 requirements-reference.txt
#torch 2.7 的 flash-attn 预编译包都需要 glibc 2.32，ModelArts 机器更旧；torch 2.6 这一版只需要 glibc 2.14
conda create -y -n nsr-ref -c conda-forge python=3.10

conda activate nsr-ref

#pip 一律经 pip-env-only.sh 运行，训练启动器也用同一规则：只看得到本环境内安装的包，ModelArts 的 modelarts.pth 加入的只读目录 /modelarts/tools/visualization/tensorboard（自带 Markdown、grpcio 等）及继承的 PYTHONPATH 全部隐藏；缺什么直接报错，不借用外部版本
#也可以直接修复已有的 Python 3.11 环境 nsr：先 pip uninstall -y flash-attn，再在 nsr 里执行下面两步（会把 torch/vllm/ray/xformers 等改回参考版本）
bash /data/lihourun/NSR/manifests/pip-env-only.sh install --no-deps -r "/data/lihourun/NSR/manifests/requirements-reference.txt"

#flash-attn 不在 PyPI 镜像上，单独从 GitHub 装（按 Python 版本自动选 cp310/cp311）
bash /data/lihourun/NSR/manifests/pip-env-only.sh install --no-deps -r "/data/lihourun/NSR/manifests/requirements-flash-attn.txt"

#若上一步报 SSL/证书错误：用 curl 或从别处下载同名 wheel，校验 sha256 后装本地文件（以 Python 3.11 为例；3.10 把 cp311 换成 cp310，哈希见 requirements-flash-attn.txt）
mkdir -p "/data/lihourun/NSR/wheels" && cd "/data/lihourun/NSR/wheels"
curl -fL -o "flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl" "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1%2Bcu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
echo "58853b28a5a926cae14402bfd8d4d93a45ebf8f9e79533f37ab09d0d77a99c05  flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl" | sha256sum -c
bash /data/lihourun/NSR/manifests/pip-env-only.sh install --no-deps "./flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"

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
