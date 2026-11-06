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
            self.assertEqual(env["PYTHONPATH"], os.pathsep.join([str(repo), str(launch.PATHFIX_DIR)]))
            self.assertEqual(env["BASELINE_ENV_ONLY"], "1")
            self.assertNotIn("RAY_ADDRESS", env)
            self.assertEqual(env["NNODES"], "1")
            self.assertEqual(env["PYTHONNOUSERSITE"], "1")
            self.assertEqual(launch.environment(Path("/data/lihourun/NSR"), repo, "run", 8, 4)["RAY_ADDRESS"], "auto")

    def test_env_diagnosis(self):
        shadow = "/modelarts/tools/visualization/tensorboard"
        broken = {"grpc": shadow + "/grpc/__init__.py", "grpc_entry": shadow,
                  "grpc_sources": ["/env/lib/python3.11/site-packages/modelarts.pth"],
                  "grpc_error": "ImportError: cannot import name 'cygrpc'", "grpc_in_env": False,
                  "grpcio": "1.80.0", "grpcio_dir": shadow, "dashboard_full": True, "dashboard_failures": [],
                  "flash_attn": "2.7.4.post1", "flash_error": None, "python_tag": "cp311", "glibc": "2.35",
                  "torch": {"version": "2.7.0+cu126", "cuda": "12.6", "cxx11abi": True},
                  "prefix": "/env", "python": "/env/bin/python"}
        healthy = dict(broken, grpc="/env/lib/python3.11/site-packages/grpc/__init__.py", grpc_sources=[],
                       grpc_entry="/env/lib/python3.11/site-packages", grpc_error=None, grpc_in_env=True)
        self.assertEqual(launch.env_diagnosis(healthy), [])
        # ModelArts case: the only grpc (and its grpcio metadata) lives outside the env.
        missing = launch.env_diagnosis(dict(broken, grpc=None, grpc_error="ModuleNotFoundError: grpc"))[0]
        self.assertIn("pip-env-only.sh install --no-deps grpcio==1.83.0", missing)
        self.assertIn("has no grpc", missing)
        broken_in_env = launch.env_diagnosis(dict(broken, grpc_in_env=True))[0]
        self.assertIn("--force-reinstall grpcio==1.83.0", broken_in_env)
        # Minimal Ray (no dashboard deps) does not need grpc at all.
        absent = dict(broken, grpc=None, grpc_entry=None, grpc_sources=[], grpcio=None, grpcio_dir=None,
                      grpc_error="ModuleNotFoundError: No module named 'grpc'")
        self.assertEqual(launch.env_diagnosis(dict(absent, dashboard_full=False)), [])
        self.assertIn("pip-env-only.sh install", launch.env_diagnosis(absent)[0])
        failing = dict(healthy, dashboard_failures=["ray.dashboard.modules.x: ImportError: boom"])
        self.assertIn("ray.dashboard.modules.x", launch.env_diagnosis(failing)[0])
        self.assertEqual(len(launch.env_diagnosis(None)), 1)
        no_flash = dict(healthy, flash_attn=None, flash_error="ModuleNotFoundError: No module named 'flash_attn'")
        message = launch.env_diagnosis(no_flash)[0]
        self.assertIn("releases/download/v2.7.4.post1/flash_attn-2.7.4.post1%2Bcu12torch2.7cxx11abiTRUE-"
                      "cp311-cp311-linux_x86_64.whl", message)
        old_abi = dict(no_flash, torch={"version": "2.6.0+cu124", "cuda": "12.4", "cxx11abi": False},
                       python_tag="cp310")
        self.assertIn("cu12torch2.6cxx11abiFALSE-cp310-cp310", launch.env_diagnosis(old_abi)[0])
        self.assertEqual(len(launch.env_diagnosis(dict(broken, flash_attn=None, flash_error="x"))), 2)
        # ModelArts node: torch 2.7 wheels need glibc 2.32, so point to the torch 2.6 reference env instead.
        old_glibc = launch.env_diagnosis(dict(no_flash, glibc="2.31"))[0]
        self.assertIn("requirements-reference.txt", old_glibc)
        self.assertNotIn("releases/download", old_glibc)
        self.assertIn("releases/download", launch.env_diagnosis(dict(old_abi, glibc="2.31"))[0])

    def test_env_only_hides_startup_pth_entries(self):
        """A .pth in the env adds an outside directory (as modelarts.pth does); env-only mode hides it."""
        import subprocess
        import sys
        import venv
        env_dir, outside, extra = self.root / "env", self.root / "modelarts", self.root / "on_pythonpath"
        try:
            venv.EnvBuilder(with_pip=False).create(env_dir)
        except Exception as exc:  # pragma: no cover - depends on the interpreter build
            self.skipTest(f"cannot create a venv: {exc}")
        python = env_dir / "bin" / "python"
        site_dir = Path(subprocess.run([str(python), "-c", "import site; print(site.getsitepackages()[0])"],
                                       capture_output=True, text=True, check=True).stdout.strip())
        (outside / "grpc").mkdir(parents=True)
        extra.mkdir()
        (site_dir / "modelarts.pth").write_text(str(outside) + "\n")
        show = ("import json, sys, sitecustomize; "
                "print(json.dumps([sys.path, getattr(sitecustomize, 'BASELINE_HIDDEN', None)]))")
        base = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "BASELINE_ENV_ONLY")}
        base["PYTHONPATH"] = os.pathsep.join([str(launch.PATHFIX_DIR), str(extra)])
        def run(**extra_env):
            out = subprocess.run([str(python), "-c", show], env=dict(base, **extra_env),
                                 capture_output=True, text=True, check=True).stdout
            return json.loads(out)
        paths, hidden = run()
        self.assertIn(str(outside), paths)          # default: the .pth entry is visible
        paths, hidden = run(BASELINE_ENV_ONLY="1")
        self.assertNotIn(str(outside), paths)       # env-only: hidden
        self.assertEqual(hidden, [str(outside)])
        self.assertIn(str(extra), paths)            # PYTHONPATH entries stay
        self.assertIn(str(site_dir), paths)         # the env's own site-packages stays

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
