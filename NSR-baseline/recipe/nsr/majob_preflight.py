"""Resolve a majob Hydra profile and check its inputs before training."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq
from hydra import compose, initialize_config_dir
from hydra.errors import HydraException
from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException


def check_inputs(config):
    model_path = Path(config.actor_rollout_ref.model.path)
    train_file = Path(config.data.train_files)
    val_file = Path(config.data.val_files)
    print(
        f"Model: {config.majob.model_id} ({model_path})\n"
        f"Train data: {train_file}\nAIME data: {val_file}",
        file=sys.stderr,
    )

    model = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    overrides = OmegaConf.to_container(config.actor_rollout_ref.model.override_config, resolve=True)
    effective_model = {**model, **overrides}
    if (
        model.get("model_type") != "qwen2"
        or model.get("hidden_size") != config.majob.model_hidden_size
        or effective_model.get("max_position_embeddings") != 4096
        or float(effective_model.get("rope_theta", 0)) != 10000.0
        or effective_model.get("rope_scaling") is not None
    ):
        raise ValueError(f"The model at {model_path} must match the native {config.majob.model_id} profile.")
    changes = {key: value for key, value in overrides.items() if model.get(key) != value}
    if changes:
        print(f"Applying model config overrides for this run: {changes}", file=sys.stderr)

    prompt = config.data.max_prompt_length
    response = config.data.max_response_length
    buffer = config.reward_model.overlong_buffer.len
    context = effective_model["max_position_embeddings"]
    if (
        type(prompt) is not int or type(response) is not int
        or prompt < 1 or response <= buffer or prompt + response > context
    ):
        raise ValueError(
            f"Need integer prompt > 0, response > overlong buffer ({buffer}), "
            f"and prompt + response <= model context ({context})."
        )
    rollout = config.actor_rollout_ref.rollout
    max_batched_tokens = rollout.max_num_batched_tokens
    if max_batched_tokens is None:
        max_batched_tokens = prompt + response
    if rollout.enable_chunked_prefill and max_batched_tokens < (rollout.max_model_len or prompt + response):
        raise ValueError("With chunked prefill, max_num_batched_tokens must cover the full sequence length.")

    # Opening the training file checks access without scanning its contents.
    with train_file.open("rb"):
        pass
    rows = pq.ParquetFile(val_file).metadata.num_rows
    if rows != config.majob.validation_rows:
        raise ValueError(f"{val_file} must contain {config.majob.validation_rows} rows, found {rows}.")
    return max_batched_tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile")
    parser.add_argument("--check", action="store_true", help="Check inputs without starting CUDA.")
    args = parser.parse_args()
    config_dir = Path(__file__).resolve().parents[1] / "dapo" / "config"
    try:
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            config = compose(config_name="dapo_trainer", overrides=[f"majob={args.profile}"])
        print(f"Profile: {config_dir / 'majob' / (args.profile + '.yaml')}", file=sys.stderr)
        max_batched_tokens = check_inputs(config)
    except (OSError, ValueError, HydraException, OmegaConfBaseException) as error:
        raise SystemExit(f"ERROR: {error}") from None

    if args.check:
        print("Preflight passed.", file=sys.stderr)
        return

    actor = config.actor_rollout_ref.actor
    rollout = config.actor_rollout_ref.rollout
    ref = config.actor_rollout_ref.ref
    try:
        gpu_count = subprocess.check_output(
            [
                sys.executable, "verl/utils/cuda_diagnostics.py",
                str(actor.ppo_mini_batch_size * rollout.n),
                str(config.data.train_batch_size * rollout.n),
                str(config.data.gen_batch_size * rollout.n),
                str(actor.ulysses_sequence_parallel_size),
                str(ref.ulysses_sequence_parallel_size),
                str(rollout.tensor_model_parallel_size),
            ],
            text=True,
        ).strip()
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from None
    print(config.majob.experiment_prefix, config.trainer.total_training_steps, max_batched_tokens, gpu_count)


if __name__ == "__main__":
    main()
