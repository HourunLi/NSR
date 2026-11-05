import json
from pathlib import Path
import tempfile
import unittest

import contextlib
import io
import os
from unittest import mock

from . import launch
from .launch import cluster_blockers, describe_existing_run, existing_run_plan, mount_fstype, temp_dirs
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
        with self.assertRaises(ValueError):
            settings(self.root, self.root / "run", n_gpus=4, nnodes=4)

    def test_eight_gpu_layouts(self):
        run = self.root / "paper-runs/test"
        four = settings(self.root, run, n_gpus=4)
        author = settings(self.root, run, n_gpus=8)
        single = settings(self.root, run, n_gpus=8, nnodes=1)
        self.assertEqual((author["trainer.n_gpus_per_node"], author["trainer.nnodes"]), (8, 4))
        self.assertEqual((single["trainer.n_gpus_per_node"], single["trainer.nnodes"]), (8, 1))
        self.assertEqual({key for key in four if four[key] != single[key]}, {"trainer.n_gpus_per_node"})
        self.assertIn("trainer.nnodes=1", hydra_args(self.root, run, n_gpus=8, nnodes=1))

    def test_cluster_check_matches_verl_pool(self):
        def nodes(*gpus, available=None):
            return {"nodes": [{"host": f"10.0.0.{i}", "alive": True, "gpus": g} for i, g in enumerate(gpus)],
                    "available_gpus": sum(gpus) if available is None else available}
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cluster_blockers(nodes(8, 8, 8, 8), 8, 4), [])
            self.assertEqual(len(cluster_blockers(nodes(8), 8, 4)), 2)
            self.assertEqual(len(cluster_blockers(nodes(8, 8, 8, 4), 8, 4)), 2)
            self.assertEqual(len(cluster_blockers(nodes(8, 8, 8, 8, available=24), 8, 4)), 1)
            dead = nodes(8, 8, 8, 8)
            dead["nodes"][0]["alive"] = False
            self.assertEqual(len(cluster_blockers(dead, 8, 4)), 1)

    def test_existing_run_is_described_not_modified(self):
        run = self.root / "paper-runs/old"
        (run / "checkpoints/global_step_20").mkdir(parents=True)
        (run / "train.log").write_text("\n".join(f"line {i}" for i in range(100)) + "\nValueError: boom\n")
        before = sorted(str(path) for path in run.rglob("*"))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            describe_existing_run(run)
        self.assertIn("global_step_20", out.getvalue())
        self.assertIn("ValueError: boom", out.getvalue())
        self.assertNotIn("line 50", out.getvalue())
        self.assertEqual(sorted(str(path) for path in run.rglob("*")), before)

    def test_mount_fstype_longest_prefix(self):
        mounts = "\n".join([
            "overlay / overlay rw 0 0",
            "tmpfs /dev/shm tmpfs rw 0 0",
            "obsfs /data/lihourun fuse.obsfs rw 0 0",
            "/dev/sdb /data/lihourun2 xfs rw 0 0",
            "srv:/x /mnt/with\\040space nfs4 rw 0 0",
        ])
        self.assertEqual(mount_fstype("/data/lihourun/rp/abc", mounts), "fuse.obsfs")
        self.assertEqual(mount_fstype("/data/lihourun", mounts), "fuse.obsfs")
        self.assertEqual(mount_fstype("/data/lihourun2/rp", mounts), "xfs")
        self.assertEqual(mount_fstype("/data/other", mounts), "overlay")
        self.assertEqual(mount_fstype("/mnt/with space/a", mounts), "nfs4")
        self.assertFalse(launch.socket_capable("fuse.obsfs"))
        self.assertFalse(launch.socket_capable("nfs4"))
        self.assertTrue(launch.socket_capable("ext4"))

    def test_temp_dirs_leave_object_store_mounts(self):
        root = Path("/data/lihourun/NSR")
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("BASELINE_LOCAL_TMP", None)
            with mock.patch.object(launch, "mount_fstype", return_value="xfs"):
                tmp, ray_tmp, note = temp_dirs(root, "run")
                self.assertEqual((tmp.parent, ray_tmp.parent.parent, note), (root.parent, root.parent, None))
            with mock.patch.object(launch, "mount_fstype", return_value="fuse.obsfs"):
                tmp, ray_tmp, note = temp_dirs(root, "run")
                self.assertEqual((tmp, ray_tmp.parent), (Path("/tmp/tp"), Path("/tmp/rp")))
                self.assertIn("fuse.obsfs", note)
                self.assertLessEqual(len(str(ray_tmp)), launch.MAX_RAY_TMPDIR_CHARS)
            os.environ["BASELINE_LOCAL_TMP"] = "/cache"
            self.assertEqual(temp_dirs(root, "run")[1].parent, Path("/cache/rp"))

    def test_training_env_drops_inherited_pythonpath(self):
        repo = Path("/data/lihourun/NSR/NSR-main")
        inherited = {"PYTHONPATH": "/modelarts/tools/visualization/tensorboard", "RAY_ADDRESS": "auto"}
        with mock.patch.dict(os.environ, inherited), mock.patch.object(launch, "mount_fstype", return_value="xfs"):
            env = launch.environment(Path("/data/lihourun/NSR"), repo, "run", 8, 1)
            self.assertEqual(env["PYTHONPATH"], str(repo))
            self.assertNotIn("RAY_ADDRESS", env)
            self.assertEqual(env["NNODES"], "1")
            self.assertEqual(env["PYTHONNOUSERSITE"], "1")
            self.assertEqual(launch.environment(Path("/data/lihourun/NSR"), repo, "run", 8, 4)["RAY_ADDRESS"], "auto")

    def test_grpc_diagnosis(self):
        shadow = "/modelarts/tools/visualization/tensorboard"
        base = {"grpc": shadow + "/grpc/__init__.py", "entry": shadow, "prefix": "/env",
                "python": "/env/bin/python", "sources": ["/env/lib/python3.11/site-packages/modelarts.pth"]}
        self.assertEqual(launch.grpc_diagnosis(dict(base, grpcio="1.83.0"), True), [])
        missing = launch.grpc_diagnosis(dict(base, grpcio=None), False)[0]
        self.assertIn("/env/bin/python -m pip install grpcio", missing)
        self.assertIn("modelarts.pth", missing)
        shadowed = launch.grpc_diagnosis(dict(base, grpcio="1.83.0"), False)[0]
        self.assertIn(shadow, shadowed)
        self.assertIn("modelarts.pth", shadowed)
        self.assertEqual(len(launch.grpc_diagnosis(None, False)), 1)

    def test_pathfix_drops_listed_entries_only(self):
        import subprocess
        import sys
        shadow = self.root / "modelarts"
        (shadow / "grpc").mkdir(parents=True)
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(launch.PATHFIX_DIR), str(shadow), str(self.root)]),
                   BASELINE_DROP_SYS_PATH=str(shadow))
        out = subprocess.run([sys.executable, "-c", "import json, sys; print(json.dumps(sys.path))"],
                             env=env, capture_output=True, text=True, check=True).stdout
        paths = json.loads(out)
        self.assertNotIn(str(shadow), paths)
        self.assertIn(str(self.root), paths)

    def test_existing_run_overwrite_policy(self):
        runs = self.root / "paper-runs"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(existing_run_plan(runs / "new", overwrite=False), (False, None))
            failed = runs / "failed"
            failed.mkdir(parents=True)
            (failed / "train.log").write_text("Traceback\n")
            self.assertEqual(existing_run_plan(failed, overwrite=False), (True, None))
            done = runs / "done"
            (done / "checkpoints/global_step_300").mkdir(parents=True)
            replace, blocker = existing_run_plan(done, overwrite=False)
            self.assertFalse(replace)
            self.assertIn("--overwrite", blocker)
            self.assertEqual(existing_run_plan(done, overwrite=True), (True, None))
            (runs / "file").write_text("x")
            self.assertFalse(existing_run_plan(runs / "file", overwrite=True)[0])

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
