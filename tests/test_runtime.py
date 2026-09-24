from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "qwen_image_runtime_under_test",
    PROJECT_ROOT / "runtime.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Could not load runtime.py for testing.")
runtime = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runtime
SPEC.loader.exec_module(runtime)


class ExternalRuntimeTests(unittest.TestCase):
    def test_external_llama_server_overrides_missing_bundled_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / (
                "llama-server.exe" if os.name == "nt" else "llama-server"
            )
            executable.write_bytes(b"test")
            executable.chmod(0o755)
            with patch.dict(
                os.environ,
                {runtime.EXTERNAL_LLAMA_SERVER_ENV: str(executable)},
                clear=False,
            ):
                spec = runtime.load_runtime_spec(root=Path(directory))

        self.assertEqual(spec.executable, executable.resolve())
        self.assertEqual(spec.library_dirs, ())
        self.assertEqual(
            spec.backend,
            f"external ({runtime.EXTERNAL_LLAMA_SERVER_ENV})",
        )

    def test_external_llama_server_must_exist(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing-qwen-image-llama-server"
            with patch.dict(
                os.environ,
                {runtime.EXTERNAL_LLAMA_SERVER_ENV: str(missing)},
                clear=False,
            ):
                with self.assertRaisesRegex(RuntimeError, "does not point to"):
                    runtime.load_runtime_spec()

    def test_empty_library_dirs_leave_environment_unchanged(self):
        spec = runtime.RuntimeSpec(
            executable=Path("/opt/llama.cpp/bin/llama-server"),
            library_dirs=(),
            platform_name="linux",
            backend="external",
            n_gpu_layers="auto",
            fit=True,
            fit_target_mib=1536,
            flash_attention="auto",
        )
        original = {"PATH": "original-path", "LD_LIBRARY_PATH": "original-libs"}
        self.assertEqual(runtime.build_runtime_environment(spec, original), original)


class ServerLifecycleTests(unittest.TestCase):
    def spec(self, executable):
        return runtime.RuntimeSpec(executable, (), "windows", "external", "auto", True, 1536, "auto")

    def test_text_server_omits_projector_and_image_server_includes_it(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "llama-server.exe"
            executable.touch()
            for projector in (None, Path(directory) / "projector.gguf"):
                server = runtime.LlamaServer(Path(directory) / "model.gguf", projector,
                                             runtime_spec=self.spec(executable))
                process = Mock()
                process.poll.return_value = None
                with patch.object(runtime.subprocess, "Popen", return_value=process) as popen, \
                        patch.object(server, "_request", return_value={}):
                    try:
                        server.start()
                        args = popen.call_args.args[0]
                        self.assertEqual("--mmproj" in args, projector is not None)
                        if projector:
                            self.assertEqual(args[args.index("--mmproj") + 1], str(projector))
                        self.assertEqual(args[args.index("--alias") + 1], "qwen-image-2.1-pe")
                    finally:
                        server.stop()

    def test_reuse_and_switch_between_text_and_image_models(self):
        manager = runtime.LlamaServerManager()
        model = Path("text.gguf")
        first, second = Mock(), Mock()
        first.is_running = second.is_running = True
        with patch.object(runtime, "load_runtime_spec", return_value=self.spec(Path("llama-server.exe"))), \
                patch.object(runtime, "LlamaServer", side_effect=[first, second]):
            self.assertEqual(manager.acquire(model, None), (first, True))
            self.assertEqual(manager.acquire(model, None), (first, False))
            self.assertEqual(manager.acquire(Path("image.gguf"), Path("mmproj.gguf")), (second, True))
            first.stop.assert_called_once()
            manager.release()
            second.stop.assert_called_once()

    def test_chat_maps_seed_and_preserves_thinking_controls(self):
        server = runtime.LlamaServer(Path("model.gguf"), None,
                                    runtime_spec=self.spec(Path("llama-server.exe")))
        with patch.object(server, "_request", return_value={
                "choices": [{"message": {"content": "result"}}], "usage": {}}) as request:
            server.chat([], seed=2**64 - 1, max_tokens=8192, temperature=0.7,
                        top_p=0.8, top_k=20, min_p=0, presence_penalty=1.5,
                        repetition_penalty=1, think_mode=True, reasoning_effort="xhigh")
            payload = request.call_args.args[2]
            self.assertEqual(payload["seed"], (2**64 - 1) % 0xFFFFFFFF)
            self.assertTrue(payload["chat_template_kwargs"]["enable_thinking"])
            self.assertEqual(payload["reasoning_effort"], "xhigh")


if __name__ == "__main__":
    unittest.main()
