from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("image_models_under_test", ROOT / "model_files.py")
models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(models)


class ModelTests(unittest.TestCase):
    def test_all_models_relative_paths_and_cache_exclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("T2I/pe_t2i_heretic-Q4_K_M.gguf", "I2I/pe_i2i_heretic-Q6_K.gguf",
                         "other/pe_t2i_heretic-Q4_K_M.gguf", "unrelated.gguf", "nested/Other.GGUF",
                         "nested/mmproj-other.GGUF", "notes.txt",
                         ".cache/pe_t2i_heretic-Q8_0.gguf", "I2I/pe_i2i_heretic.mmproj-bf16.gguf"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            self.assertEqual(models.model_options(root),
                             ["I2I/pe_i2i_heretic-Q6_K.gguf", "nested/Other.GGUF",
                              "other/pe_t2i_heretic-Q4_K_M.gguf",
                              "T2I/pe_t2i_heretic-Q4_K_M.gguf", "unrelated.gguf"])
            self.assertEqual(models.projector_options(root),
                             ["", "I2I/pe_i2i_heretic.mmproj-bf16.gguf", "nested/mmproj-other.GGUF"])
            self.assertEqual(models.resolve_gguf(root, "nested/Other.GGUF"), root / "nested/Other.GGUF")
            self.assertEqual(models.resolve_gguf(root, "pe_i2i_heretic-Q6_K.gguf"),
                             root / "I2I/pe_i2i_heretic-Q6_K.gguf")
            with self.assertRaisesRegex(FileNotFoundError, "Ambiguous"):
                models.resolve_gguf(root, "pe_t2i_heretic-Q4_K_M.gguf")

    def test_unified_selector_only_lists_installed_q8_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = ["Qwen-Image-2.1-PE-T2I-Heretic-GGUF/pe_t2i_heretic-Q8_0.gguf",
                     "Qwen-Image-2.1-PE-I2I-Heretic-GGUF/pe_i2i_heretic-Q8_0.gguf"]
            for name in names:
                path = root / name
                path.parent.mkdir()
                path.touch()
            projector = root / Path(names[1]).parent / models.DEFAULT_MMPROJ
            projector.touch()
            self.assertEqual(set(models.model_options(root)), set(names))
            self.assertEqual(models.projector_options(root), ["", projector.relative_to(root).as_posix()])
            (root / names[1]).unlink()
            self.assertEqual(models.model_options(root), [names[0]])

    def test_path_escape_and_missing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            outer = Path(directory)
            root = outer / "LLM"
            root.mkdir()
            (outer / "outside.gguf").touch()
            for name in ("../outside.gguf", str(outer / "outside.gguf"), "missing.gguf"):
                with self.assertRaises(FileNotFoundError):
                    models.resolve_gguf(root, name)
            self.assertEqual(models.model_options(root, "T2I"), [models.DEFAULT_MODELS["T2I"]])


if __name__ == "__main__":
    unittest.main()
