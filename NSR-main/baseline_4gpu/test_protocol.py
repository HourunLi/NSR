import json
from pathlib import Path
import tempfile
import unittest

from .protocol import SYSTEM_PROMPT, dataset_path, hydra_args, settings, validate_aime_rows, with_answer_system
from .report import pass_at_k, summarize


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        (self.root / "data").mkdir()
        for filename in ("dapo-math-17k.parquet", "aime-2024.parquet"):
            (self.root / "data" / filename).touch()

    def test_dataset_layouts_reach_hydra(self):
        names = {"data.train_files": ("DAPO-Math-17k", "dapo-math-17k.parquet"),
                 "data.val_files": ("AIME-2024", "aime-2024.parquet")}
        expected = {key: str(self.root / "data" / filename)
                    for key, (_, filename) in names.items()}
        for layout in ("flat", "mixed", "nested"):
            if layout != "flat":
                key = "data.val_files" if layout == "mixed" else "data.train_files"
                folder, filename = names[key]
                target = self.root / "data" / folder / "data" / filename
                target.parent.mkdir(parents=True)
                Path(expected[key]).rename(target)
                (target.parent.parent / "info.json").write_text("{}")
                expected[key] = str(target)
            with self.subTest(layout=layout):
                cfg = settings(self.root, self.root / "run")
                args = hydra_args(self.root, self.root / "run")
                for key, path in expected.items():
                    self.assertEqual(cfg[key], path)
                    self.assertIn(f"{key}={json.dumps(path)}", args)

    def test_missing_and_ambiguous_dataset_rejected(self):
        filename = "aime-2024.parquet"
        flat = self.root / "data" / filename
        flat.unlink()
        with self.assertRaises(FileNotFoundError):
            dataset_path(self.root, filename)
        for folder in ("AIME-2024", "backup"):
            nested = self.root / "data" / folder / "data" / filename
            nested.parent.mkdir(parents=True)
            nested.touch()
        with self.assertRaises(ValueError):
            dataset_path(self.root, filename)
        nested.unlink()
        flat.touch()
        with self.assertRaises(ValueError):
            dataset_path(self.root, filename)

    def test_prompt_is_explicit_without_changing_question(self):
        source = [{"role": "user", "content": "Problem. Answer:"}]
        result = with_answer_system(source)
        self.assertEqual(result[0], {"role": "system", "content": SYSTEM_PROMPT})
        self.assertEqual(result[1:], source)
        self.assertEqual(len(source), 1)
        result[1]["content"] = "changed"
        self.assertEqual(source[0]["content"], "Problem. Answer:")

    def test_idempotent_and_conflicting_system_rejected(self):
        messages = with_answer_system([{"role": "user", "content": "a"}])
        self.assertEqual(with_answer_system(messages), messages)
        with self.assertRaises(ValueError):
            with_answer_system([{"role": "system", "content": "boxed"}, {"role": "user", "content": "a"}])

    def test_repetition_validation(self):
        rows = [{"prompt": [{"role": "user", "content": str(i)}], "data_source": "aime2024"}
                for i in range(30) for _ in range(32)]
        self.assertEqual(len(validate_aime_rows(rows)), 30)
        with self.assertRaises(ValueError):
            validate_aime_rows(rows[:-1])

    def test_core_parameters(self):
        root = self.root
        cfg = settings(root, root / "paper-runs/test")
        expected = {"trainer.n_gpus_per_node": 4, "trainer.nnodes": 1,
                    "trainer.total_training_steps": 300, "data.train_batch_size": 512,
                    "actor_rollout_ref.actor.ppo_mini_batch_size": 32,
                    "actor_rollout_ref.rollout.n": 16,
                    "actor_rollout_ref.rollout.gpu_memory_utilization": 0.50,
                    "actor_rollout_ref.rollout.val_kwargs.n": 1,
                    "actor_rollout_ref.actor.NSR": False,
                    "actor_rollout_ref.actor.optim.lr": 1e-6,
                    "actor_rollout_ref.actor.optim.lr_warmup_steps": 0,
                    "data.max_response_length": 8192,
                    "trainer.save_freq": 300}
        for key, value in expected.items():
            self.assertEqual(cfg[key], value, key)
        self.assertEqual(512 // 32, 16)
        self.assertEqual(len(hydra_args(root, root / "test")), len(cfg))

    def test_two_gpu_changes_only_gpu_count(self):
        root = self.root
        run = root / "paper-runs/test"
        four = settings(root, run, n_gpus=4)
        two = settings(root, run, n_gpus=2)
        changed = {key for key in four if four[key] != two[key]}
        self.assertEqual(changed, {"trainer.n_gpus_per_node"})
        self.assertEqual(two["trainer.n_gpus_per_node"], 2)
        self.assertEqual(two["data.train_batch_size"], 512)
        self.assertEqual(two["actor_rollout_ref.actor.ppo_mini_batch_size"], 32)
        self.assertEqual(two["actor_rollout_ref.rollout.n"], 16)
        self.assertEqual(two["actor_rollout_ref.actor.ulysses_sequence_parallel_size"], 2)

    def test_invalid_gpu_count_rejected(self):
        with self.assertRaises(ValueError):
            settings(Path("/tmp/root"), Path("/tmp/run"), n_gpus=3)

    def test_pass_estimator(self):
        self.assertEqual(pass_at_k(32, 0, 16), 0)
        self.assertEqual(pass_at_k(32, 32, 16), 1)
        self.assertEqual(pass_at_k(32, 1, 16), 0.5)
        self.assertAlmostEqual(pass_at_k(32, 8, 1), 0.25)

    def test_report_from_saved_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "300.jsonl"
            rows = [{"input": str(q), "step": 300, "acc": i == 0,
                     "pred": "540" if i == 0 else "[INVALID]"}
                    for q in range(30) for i in range(32)]
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            result = summarize(path)
            self.assertAlmostEqual(result["pass_at_1"], 1/32)
            self.assertAlmostEqual(result["pass_at_16"], 0.5)
            self.assertAlmostEqual(result["invalid_fraction"], 31/32)
            path.write_text(json.dumps(rows[0]), encoding="utf-8")
            with self.assertRaises(ValueError):
                summarize(path)


if __name__ == "__main__":
    unittest.main()
