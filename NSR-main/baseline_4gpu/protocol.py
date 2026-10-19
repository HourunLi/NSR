"""Paper-aligned choices; see README.md for provenance and unresolved details."""
import copy
import json
from collections import Counter
from pathlib import Path

SYSTEM_PROMPT = (
    "Please reason step by step. The last line of your response must be "
    "Answer: <final answer>. Replace <final answer> with the actual answer. "
    "Do not wrap the final answer in LaTeX commands."
)


def with_answer_system(messages):
    """Keep the source question intact; never silently replace an existing system."""
    result = copy.deepcopy(list(messages))
    if not result or not any(m.get("role") == "user" for m in result):
        raise ValueError("Expected a non-empty conversation with a user message")
    systems = [m for m in result if m.get("role") == "system"]
    if systems:
        if len(systems) != 1 or result[0] != systems[0] or systems[0].get("content") != SYSTEM_PROMPT:
            raise ValueError("Unexpected existing system prompt; inspect instead of overwriting")
        return result
    return [{"role": "system", "content": SYSTEM_PROMPT}] + result


def prompt_key(messages):
    return json.dumps(messages, ensure_ascii=False, sort_keys=True)


def validate_aime_rows(rows):
    counts = Counter(prompt_key(row["prompt"]) for row in rows)
    if len(rows) != 960 or len(counts) != 30 or set(counts.values()) != {32}:
        raise ValueError(
            "This protocol requires exactly 30 distinct AIME prompts, each repeated 32 times; "
            f"got {len(rows)} rows, {len(counts)} prompts, repetition histogram {dict(Counter(counts.values()))}"
        )
    for row in rows:
        source = row["data_source"]
        if source != "math_dapo" and not source.startswith("aime"):
            raise ValueError(f"Unexpected validation scoring route: {source}")
    return counts


def settings(root: Path, run_dir: Path, n_gpus: int = 4):
    """Explicit overrides; no hidden repeated runs and no unrecorded fast-mode flags."""
    if n_gpus not in (2, 4, 8):
        raise ValueError("This launcher supports 2/4 GPUs locally, or 8 GPUs per node across 4 nodes")
    return {
        # Paper B.1/B.2/B.3; prompt length/truncation/gen batch from author script.
        "data.train_files": str(root / "data/dapo-math-17k.parquet"),
        "data.val_files": str(root / "data/aime-2024.parquet"),
        "data.prompt_key": "prompt",
        "data.truncation": "left",
        "data.max_prompt_length": 2048,
        "data.max_response_length": 8192,
        "data.gen_batch_size": 1536,
        "data.train_batch_size": 512,
        "data.filter_overlong_prompts": False,
        "++data.cache_dir": str(root / "cache/verl/rlhf"),
        "data.custom_cls.path": "pkg://baseline_4gpu.dataset",
        "data.custom_cls.name": "AnswerFormatDataset",
        "++data.seed": 1,  # Existing sampler default; not an author-published experiment seed.
        "actor_rollout_ref.model.path": str(root / "models/Qwen2.5-Math-7B"),
        "actor_rollout_ref.model.use_remove_padding": True,
        "actor_rollout_ref.model.enable_gradient_checkpointing": True,
        "actor_rollout_ref.model.enable_activation_offload": True,
        "actor_rollout_ref.actor.NSR": False,
        "actor_rollout_ref.actor.nsr_delta": 0.1,  # Inactive when NSR=False.
        "actor_rollout_ref.actor.use_random_adv": False,
        "actor_rollout_ref.actor.clip_ratio_low": 0.2,
        "actor_rollout_ref.actor.clip_ratio_high": 0.28,
        "actor_rollout_ref.actor.clip_ratio_c": 10.0,
        "actor_rollout_ref.actor.use_kl_loss": False,
        "actor_rollout_ref.actor.kl_loss_coef": 0.0,
        "actor_rollout_ref.actor.entropy_coeff": 0.0,
        "actor_rollout_ref.actor.loss_agg_mode": "token-mean",
        "actor_rollout_ref.actor.policy_loss.loss_mode": "vanilla",
        "actor_rollout_ref.actor.ppo_mini_batch_size": 32,
        "actor_rollout_ref.actor.ppo_epochs": 1,
        "actor_rollout_ref.actor.optim.lr": 1e-6,
        "actor_rollout_ref.actor.optim.lr_warmup_steps": 0,
        "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio": 0.0,
        "actor_rollout_ref.actor.optim.warmup_style": "constant",
        "actor_rollout_ref.actor.optim.weight_decay": 0.1,
        "actor_rollout_ref.actor.grad_clip": 1.0,
        "actor_rollout_ref.actor.use_dynamic_bsz": True,
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": 6144,
        "actor_rollout_ref.actor.ulysses_sequence_parallel_size": 2,
        "actor_rollout_ref.actor.fsdp_config.fsdp_size": -1,
        "actor_rollout_ref.actor.fsdp_config.param_offload": True,
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload": True,
        "actor_rollout_ref.actor.checkpoint.save_contents": ["hf_model"],
        "actor_rollout_ref.actor.checkpoint.load_contents": [],
        "actor_rollout_ref.rollout.name": "vllm",
        "actor_rollout_ref.rollout.n": 16,
        "actor_rollout_ref.rollout.tensor_model_parallel_size": 1,
        # Four-card H20 safety adaptation after a real OOM while remapping vLLM KV cache.
        # This is an execution-memory setting, not a paper-published algorithm hyperparameter.
        "actor_rollout_ref.rollout.gpu_memory_utilization": 0.50,
        "actor_rollout_ref.rollout.enforce_eager": True,  # Inherited repo default; no risky engine change.
        "actor_rollout_ref.rollout.enable_chunked_prefill": True,
        "actor_rollout_ref.rollout.max_num_batched_tokens": 10240,
        "actor_rollout_ref.rollout.temperature": 1.0,
        "actor_rollout_ref.rollout.top_p": 1.0,
        "actor_rollout_ref.rollout.top_k": -1,
        "++actor_rollout_ref.rollout.seed": 0,  # Existing engine default, explicitly recorded.
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz": True,
        "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu": 6144,
        "actor_rollout_ref.rollout.val_kwargs.temperature": 1.0,
        "actor_rollout_ref.rollout.val_kwargs.top_p": 0.7,
        "actor_rollout_ref.rollout.val_kwargs.top_k": -1,
        "actor_rollout_ref.rollout.val_kwargs.do_sample": True,
        # 960 source rows already contain 30 x 32 repetitions. Paper B.5: k=32.
        "actor_rollout_ref.rollout.val_kwargs.n": 1,
        "actor_rollout_ref.ref.log_prob_use_dynamic_bsz": True,
        "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu": 6144,
        "actor_rollout_ref.ref.ulysses_sequence_parallel_size": 2,
        "actor_rollout_ref.ref.fsdp_config.param_offload": True,
        "algorithm.adv_estimator": "grpo",
        "algorithm.use_kl_in_reward": False,
        "algorithm.kl_ctrl.kl_coef": 0.0,
        "algorithm.filter_groups.enable": True,
        "algorithm.filter_groups.metric": "acc",
        "algorithm.filter_groups.max_num_gen_batches": 10,
        "reward_model.reward_manager": "dapo",
        "reward_model.overlong_buffer.enable": True,
        "reward_model.overlong_buffer.len": 4096,
        "reward_model.overlong_buffer.penalty_factor": 1.0,
        "trainer.logger": ["console"],
        "trainer.project_name": "DAPO-paper-aligned",
        "trainer.experiment_name": run_dir.name,
        "trainer.n_gpus_per_node": n_gpus,
        "trainer.nnodes": 4 if n_gpus == 8 else 1,
        "trainer.val_before_train": True,
        "trainer.test_freq": 5,
        "trainer.save_freq": 300,
        "trainer.max_actor_ckpt_to_keep": 1,
        # At most 300*10=3000 generation batches; three 1166-batch passes suffice.
        "trainer.total_epochs": 3,
        "trainer.total_training_steps": 300,
        "trainer.resume_mode": "disable",
        "trainer.default_local_dir": str(run_dir / "checkpoints"),
        "trainer.rollout_data_dir": None,
        "trainer.validation_data_dir": str(run_dir / "validation"),
        "hydra.run.dir": str(run_dir / "hydra"),
    }


def hydra_args(root, run_dir, n_gpus=4):
    return [
        f"{key}={json.dumps(value, ensure_ascii=False)}"
        for key, value in settings(root, run_dir, n_gpus=n_gpus).items()
    ]
