# 四卡 Qwen2.5-Math-7B / DAPO baseline

这是一次新的、最多 300 个外层训练 step 的 baseline 运行，不是 smoke test。NSR 关闭；保持论文明确披露的训练规模，不按 4/32 缩小全局 batch。代码调用原来的 `recipe.dapo.main_dapo`，不另写训练算法，不改原始模型、Parquet、评分器或旧启动脚本。

范围：DAPO-Math-17k 训练、AIME-2024 验证。**不是论文全部实验的完整复现**：论文还评测 AIME25、AMC、GPQA、MMLU-Pro，并要求至少三次独立重复。这里先跑一次，不自动把你的计算费用乘三。

## 1. 参数依据和不能伪称“完全一致”的地方

论文指用户提供的 16 页 PDF《Stabilizing RLVR via Stochastic Recovery of Near-Boundary Signals》。作者脚本指 `recipe/nsr/NSR_delta_0.1_7b.sh`；其 NSR 开关在本方案关闭。

| 项目 | 本方案 | 依据 |
|---|---|---|
| 模型、数据、回答长度 | Qwen2.5-Math-7B、DAPO-Math-17k、8192 token | 论文 B.1/B.2 |
| 算法 | DAPO / GRPO 优势估计，NSR=False | 论文 baseline 描述、作者实现 |
| 全局 batch / mini-batch | 512 / 32 | B.3 |
| 每题训练采样数 | 16 | 作者 7B 脚本 |
| vLLM GPU显存比例 | 0.60 | 四卡H20稳定性适配；不是论文算法参数 |
| KL、entropy regularization | 均关闭 | B.3 |
| clip low / high | 0.2 / 0.28 | B.3 |
| dynamic sampling、token-mean、overlong shaping | 开启 | B.3；具体字段沿用作者脚本 |
| 学习率 | 从第一步起固定 1e-6，warmup=0 | 按 B.3 “fixed” 字面选择，见下文 |
| 外层训练 step | 300 | 附录 C 的 300×16，结合代码 mini-batch 更新循环与图10 |
| AIME 每题采样、温度、top-p | 32、1.0、0.7 | B.5 |
| 验证 / 保存间隔 | 5 / 20 step，含初始验证、最终验证和保存 | 作者脚本 |
| GPU / 节点 | 4 / 1 | 用户硬件适配；作者脚本默认 8 / 4，不是论文对硬件的明确声明 |

必须披露的选择：

- **学习率细节不完全公开**：论文写固定 1e-6，但没有明确说明 warmup；作者脚本设 warmup=10。这里选 0 是“从头固定”的字面方案，不能证明作者实际实验没有 warmup。若追求作者脚本轨迹而非这个字面方案，应把 `protocol.py` 中 `lr_warmup_steps` 改回 10，并在实验记录中声明。
- **300×16 的实现解释**：此版代码每个外层 step 将训练批划分为 512/32=16 个 mini-batch，并分别执行更新。不要再凭论文的 “accumulation” 字样叠加一个额外的 16 倍参数。`trainer.total_epochs=3` 只是允许数据迭代器供应足够的批次，`total_training_steps=300` 才是停止上限；它不表示三次独立实验。动态筛选失败仍可能提前报错，脚本不保证无条件完成 300 步。
- **AIME 重复采样修正**：你的验证文件是 960 行，即 30 题×每题重复32次，因此 `val_kwargs.n=1`，不再乘一次32。预检查要求这个文件结构；如更换为30行文件，不能继续用这个配置。
- **提示词兼容修正**：用户提供的 tokenizer 自动 system 要求 boxed，而数据与 `math_dapo.py` 要求 `Answer:`。自定义数据集在内存里补一条明确的 `Answer:` system，保留原题和评分器。论文没有公开这条 system 的逐字内容，这不是“作者原提示词”。已有其他 system 时直接报错，不覆盖。
- **上下文长度**：模型配置必须支持至少 2048+8192=10240。预检查会读取并记录你已修改过的本地模型配置，但不会自动改 config.json / RoPE。能通过长度检查不等于论文证明了某一种上下文扩展方案。
- **其他未披露值**：生成 batch1536、prompt2048、overlong4096、weight_decay0.1、SP2/TP1、offload、eager 等来自作者脚本/仓库默认值。保留它们，不冒称每一项都见于论文。数据 shuffle seed1、vLLM seed0 是此版实现的默认值，不是作者公开的独立实验种子；不能保证逐位可复现。
- **OOM修正**：第一版曾沿用作者脚本的 `gpu_memory_utilization=0.8`。实际四卡任务在完成约20个外层step后，于 `vLLM wake_up(tags=["kv_cache"])` 重新映射KV cache时OOM；因此本版改为 `0.60`。它不改变DAPO损失、batch、采样数、clip阈值或训练步数，但会减少KV cache并可能降低吞吐。它也不能防止其他用户在任务运行后重新占用同一GPU。
- **输出管理**：日志改 console；保留最近两个完整 actor checkpoint。这是运行管理选择，不影响训练目标；框架会自动删除本次新实验目录内更旧的 checkpoint，不动旧实验目录。

## 2. 先确认空间和旧任务

完整保存包含 FSDP 权重、Adam 状态、其他恢复信息和 HF 导出。7.62B 模型在不同精度下，一个完整 checkpoint 可能约 100–120 GiB；实际以文件为准。保留两个并留工作空间，本启动器采用 **300 GiB 可用空间的保守门槛**，这不是论文规定，也不是最低理论需求。

还会要求运行时所在文件系统至少有20 GiB且至少6%可用，避免继续依赖调高 Ray 满盘阈值。若输出盘另有位置，可设置 `BASELINE_RUNS=/你的可写目录/paper-runs`，但运行时/缓存目录仍需要足够空间。

```bash
df -h /home/lihourun/hsc/NSR-work
nvidia-smi
```

**你此前剩余十几 GB 时不满足这个保存方案。** 预检查允许旧训练继续运行，但新训练拒绝占用已有任务的 GPU，不会自动结束任何进程。不要先停掉已经跑20h的任务再检查空间。

新参数不能热更新到旧 Python 进程。切换意味着从本地初始 Qwen 模型重新训练；旧任务如果一直 `save_freq=-1`，结束后未保存的训练进度无法恢复。先保留旧日志，再决定是否结束旧任务。

## 3. 上传（只需一次）

把 `baseline_2or4gpu_unsafe_overrides_v5.zip` 上传到服务器 `/home/lihourun/hsc/NSR-work/`。不需要重装环境或重下模型。

```bash
source /home/lihourun/miniconda3/etc/profile.d/conda.sh
conda activate nsr
export NSR_ROOT=/home/lihourun/hsc/NSR-work
cd "$NSR_ROOT/NSR-main"

# 如果没有安装过旧包：
test ! -e baseline_4gpu && python -m zipfile -e "$NSR_ROOT/baseline_2or4gpu_unsafe_overrides_v5.zip" .

# 如果已经安装过前一版，先备份，再用新包更新；不要执行上一行：
# cp -a baseline_4gpu "baseline_4gpu.before_oomsafe.$(date +%Y%m%d-%H%M%S)"
# python -m zipfile -e "$NSR_ROOT/baseline_2or4gpu_unsafe_overrides_v5.zip" .
python -m unittest baseline_4gpu.test_protocol -v
```

解压后的文件夹位置应为 `$NSR_ROOT/NSR-main/baseline_4gpu/`，不是再嵌套一层 NSR-main。

## 4. 预检查（可以在旧任务仍运行时执行）

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -u -m baseline_4gpu.launch check --name DAPO7B-paper4-v2
```

两卡运行时必须同时只暴露两张卡并显式指定数量，例如物理GPU 2、3：

```bash
export CUDA_VISIBLE_DEVICES=2,3
python -u -m baseline_4gpu.launch check \
  --name DAPO7B-paper2-oomsafe-v1 \
  --n-gpus 2 \
  --skip-disk-check
```

两卡只改变 `trainer.n_gpus_per_node=2`、`N_GPUS=2` 和可见GPU数量校验。全局batch512、mini-batch32、rollout.n16、SP2、TP1及300步均保持不变。因此每张卡承担更多工作，墙钟时间通常显著长于四卡；不要为了加速检查而把论文batch等参数减半。

它检查本地模型文件、上下文长度、实际数据集适配方法、验证重复次数、评分器、你服务器安装的 Hydra 配置接口和空间。不加载7B权重到GPU，不启动Ray训练。只读取一条训练样本及验证文件，不扫描整个训练集的每条提示词；模型分片检查是存在性/非空检查，不是15GB文件的完整性校验。

应看到 `AIME ... 960 samples`、`Hydra configuration: OK`。`BLOCKED` 表示还不能启动，先解决其原因。GPU busy 在 check 模式仅提示；run 模式会阻止重复占用。

如果报模型 context 小于10240、配置键不存在、已有不同system、磁盘不足等，保留完整输出并反馈；不要去掉检查强行启动。

如用户明确决定承担磁盘风险，可在 `check` 和 `run` 两处都添加 `--skip-disk-check`。它只忽略本启动器的300GiB/运行盘余量门槛，不关闭Ray自身的95%满盘保护，不关闭checkpoint写入，也不会忽略GPU占用。运行清单会记录该选项和启动时剩余空间：

```bash
python -u -m baseline_4gpu.launch check \
  --name DAPO7B-paper4-oomsafe-v1 \
  --skip-disk-check
```

当前仅约10.6GiB可用时，小于本地14.19GiB模型权重本身；第20步完整checkpoint以及最终HF导出很可能失败。跳过检查不是对空间充足的证明。

如还要允许使用已有进程/显存分配的GPU，在 `check` 和 `run` 中同时添加 `--skip-gpu-check`。它不会释放、预留或隔离任何显存，也不会结束其他进程；运行清单会记录启动时繁忙的GPU。即使使用该参数，当前账号已有另一个DAPO/Ray训练进程时仍会阻止启动。

## 5. 确认切换后，结束自己的旧任务

先找出自己旧任务的实际 Python driver，而不是根据过期 PID 直接结束：

```bash
ps -u "$(id -un)" -o pid,ppid,etime,args | grep -E 'recipe.dapo.main_dapo|DAPO_baseline_7b.sh'
```

核对命令行确实属于要结束的旧 baseline 后，用 `kill -TERM 实际的Python_driver_PID`。这里的中文是占位符，不能原样执行。不要用 `pkill python` 或全机批量杀进程命令。再看 `nvidia-smi`；若旧 Ray worker 不退出，先检查它们的用户和命令行，不要盲目杀其它任务。

GPU 0、1、2、3 都释放且允许你使用后，再执行下一节。只看到GPU空闲并不代替实验室/集群的使用许可。

## 6. 启动新的完整单次 baseline

```bash
cd "$NSR_ROOT/NSR-main"
export CUDA_VISIBLE_DEVICES=0,1,2,3
mkdir -p "$NSR_ROOT/logs"
LAUNCH_LOG="$NSR_ROOT/logs/paper4-launch-$(date +%Y%m%d-%H%M%S).log"
nohup python -u -m baseline_4gpu.launch run --name DAPO7B-paper4-v2 > "$LAUNCH_LOG" 2>&1 &
echo "启动器 PID=$!"
echo "启动检查日志=$LAUNCH_LOG"
tail -n 80 -f "$LAUNCH_LOG"
```

明确跳过磁盘检查时，启动行改为：

```bash
nohup python -u -m baseline_4gpu.launch run \
  --name DAPO7B-paper4-oomsafe-v1 \
  --skip-disk-check \
  --skip-gpu-check \
  > "$LAUNCH_LOG" 2>&1 &
```

两卡且同时显式忽略磁盘/GPU占用检查时：

```bash
export CUDA_VISIBLE_DEVICES=2,3
nohup python -u -m baseline_4gpu.launch run \
  --name DAPO7B-paper2-oomsafe-v1 \
  --n-gpus 2 \
  --skip-disk-check \
  --skip-gpu-check \
  > "$LAUNCH_LOG" 2>&1 &
```

启动器检查通过后，打印真正训练日志的位置。Ctrl+C 只退出当前的 tail，不会结束 nohup 训练。

```bash
tail -n 100 -f "$NSR_ROOT/paper-runs/DAPO7B-paper4-v2/train.log"
```

使用 `BASELINE_RUNS` 时相应调整路径。此启动器只接受新实验目录；同名目录已存在时拒绝覆盖，也不会偷偷 resume。重跑先保存失败日志，再换一个明确的新 `--name`。

日志开头应出现：NSR=False、total_training_steps=300、n_gpus_per_node=4、val_kwargs.n=1、gpu_memory_utilization=0.6、学习率warmup=0。先初始验证，之后才开始训练；这不是卡住的充分证据。

确认进度请看日志中的实际 `step:`、`actor/lr`、reward/acc 与验证结果，不要仅根据 tqdm 或 GPU 利用率推断完成百分比。动态采样会额外消耗生成批次，4张H20也不能直接按768/4小时预测结束时间。

## 7. 结果在哪里

新实验目录中：

- `train.log`：训练输出。
- `resolved.yaml`：启动时解析的完整配置。
- `manifest.json`：显式参数、软件版本、关键源文件哈希、实际system示例、本地模型配置。
- `launcher_snapshot/`：本次启动器副本。
- `validation/0.jsonl`、`5.jsonl`……`300.jsonl`：每次验证960条生成及评分。
- `checkpoints/global_step_300/actor/`：最终训练状态。
- `checkpoints/global_step_300/actor/huggingface/`：最终HF模型导出。

每20步保存，末步300也保存；只保留最近两个 actor checkpoint，因此不能期待事后比较所有历史模型。

```bash
python -m baseline_4gpu.report "$NSR_ROOT/paper-runs/DAPO7B-paper4-v2/validation/300.jsonl"
```

报告按每题32个样本计算 Pass@1 和组合估计 `1-C(n-c,16)/C(n,16)`，再对30题平均。论文没有披露具体估计器实现；这里明确采用常见无放回估计，不把此版框架 bootstrap 的 `best@16/mean` 自动当成相同指标。报告使用原评分器的 `acc`，不是带长度惩罚的 shaped reward。

## 验证边界

本包已做纯Python单元测试和本地源代码接口静态检查。交付端没有你的Linux/H20环境，未完成GPU端端到端训练测试；服务器的 `check` 会进一步验证真实依赖和Hydra组合，但它也不等于完整训练成功。真实模型训练中的OOM、通信超时、动态采样无有效组等仍需根据新日志诊断。
