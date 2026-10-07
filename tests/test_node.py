"""Exercise node execution without requiring a GPU or importing ComfyUI."""
from __future__ import annotations

import importlib.util
import asyncio
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "image_prompt_test_package"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package


class Field:
    def __init__(self, name=None, **kwargs):
        self.name = name
        self.__dict__.update(kwargs)


class NodeOutput:
    def __init__(self, *values):
        self.result = values


io = types.SimpleNamespace(ComfyNode=object, Schema=Field, NodeOutput=NodeOutput,
                           **{name: types.SimpleNamespace(Input=Field, Output=Field)
                              for name in ("Image", "String", "Combo", "Boolean", "Int")})
management = types.ModuleType("comfy.model_management")
management.soft_empty_cache = Mock()
management.unload_all_models = Mock()
comfy = types.ModuleType("comfy")
comfy.model_management = management
api = types.ModuleType("comfy_api.latest")
api.io = io
api.ComfyExtension = object
folder_paths = types.ModuleType("folder_paths")
folder_paths.models_dir = "unused"
media = types.ModuleType(PACKAGE + ".media")
media.image_content = lambda image: {"type": "image_url", "image_url": {"url": image.label}}
media.text_content = lambda text: {"type": "text", "text": text}
with patch.dict(sys.modules, {"comfy": comfy, "comfy.model_management": management,
                            "comfy_api": types.ModuleType("comfy_api"), "comfy_api.latest": api,
                            "folder_paths": folder_paths, PACKAGE + ".media": media}):
    spec = importlib.util.spec_from_file_location(PACKAGE + ".node", ROOT / "node.py")
    node = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = node
    spec.loader.exec_module(node)


def image(label="image", batch=1):
    return types.SimpleNamespace(shape=(batch, 16, 16, 3), label=label)


def response(mode="T2I"):
    value = {"rewritten_prompt": "A precise image prompt.", "wh_ratio": "1:1"}
    if mode == "I2I":
        value["ratio_follow"] = ""
    return json.dumps(value), {"total_tokens": 42}


class NodeTests(unittest.TestCase):
    def test_package_entrypoint_loads_new_node(self):
        spec = importlib.util.spec_from_file_location(PACKAGE, ROOT / "__init__.py",
                                                     submodule_search_locations=[str(ROOT)])
        entrypoint = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {PACKAGE: entrypoint, PACKAGE + ".node": node}):
            spec.loader.exec_module(entrypoint)
        extension = asyncio.run(entrypoint.comfy_entrypoint())
        self.assertEqual(asyncio.run(extension.get_node_list()), [node.QwenImagePrompt])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in (*node.DEFAULT_MODELS.values(), node.DEFAULT_MMPROJ):
            (self.root / name).touch()
        self.server = Mock()
        self.server.chat.return_value = response()
        self.manager = Mock()
        self.manager.acquire.return_value = self.server, True
        for target, value in (("MODEL_DIR", self.root), ("SERVER_MANAGER", self.manager)):
            patcher = patch.object(node, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_schema_has_ten_fixed_optional_images_and_original_outputs(self):
        schema = node.QwenImagePrompt.define_schema()
        images = [field for field in schema.inputs if (field.name or "").startswith("reference_image_")]
        self.assertEqual(len(images), 10)
        self.assertTrue(all(field.optional for field in images))
        self.assertEqual([o.name for o in schema.outputs], ["h3_prompt", "selected_skill", "detected_mode"])
        for removed in ("model_mode", "t2i_model", "i2i_model"):
            self.assertNotIn(removed, [f.name for f in schema.inputs])
        self.assertEqual(next(f for f in schema.inputs if f.name == "mode").options, ["T2I", "I2I"])
        self.assertEqual(schema.node_id, "QwenImagePromptSimple")
        model_field = next(f for f in schema.inputs if f.name == "model")
        self.assertEqual(set(model_field.options), set(node.DEFAULT_MODELS.values()))
        self.assertIn(model_field.default, model_field.options)
        self.assertEqual(next(f for f in schema.inputs if f.name == "skill").options, ["qwen-image-prompt"])
        self.assertEqual(next(f for f in schema.inputs if f.name == "mmproj").default, "")

    def test_t2i_without_images_loads_only_t2i_without_projector(self):
        (self.root / node.DEFAULT_MMPROJ).unlink()
        (self.root / node.DEFAULT_MODELS["I2I"]).unlink()
        result = node.QwenImagePrompt.execute("A bicycle", reference_image_3=None)
        self.assertEqual(result.result, ("A precise image prompt.", "qwen-image-prompt", "T2I"))
        self.assertIsNone(self.manager.acquire.call_args.args[1])
        self.manager.release.assert_called_once()

    def test_sparse_images_are_forwarded_in_socket_order(self):
        self.server.chat.return_value = response("I2I")
        result = node.QwenImagePrompt.execute("Combine these", mode="I2I", model=node.DEFAULT_MODELS["I2I"], mmproj=node.DEFAULT_MMPROJ, reference_image_2=image("second"),
                                             reference_image_10=image("tenth"), force_unload_model=False)
        self.assertEqual(result.result[-1], "I2I")
        content = self.server.chat.call_args.args[0][1]["content"]
        self.assertEqual([item["image_url"]["url"] for item in content if item["type"] == "image_url"],
                         ["second", "tenth"])
        self.assertEqual([item["text"] for item in content if item["type"] == "text"],
                         ["<image1>", "<image2>", "Combine these"])
        self.assertEqual(self.manager.acquire.call_args.args[1], self.root / node.DEFAULT_MMPROJ)
        self.manager.release.assert_not_called()

    def test_all_ten_images_forwarded(self):
        self.server.chat.return_value = response("I2I")
        node.QwenImagePrompt.execute("Combine", mode="I2I", model=node.DEFAULT_MODELS["I2I"], mmproj=node.DEFAULT_MMPROJ, **{f"reference_image_{i}": image(str(i)) for i in range(1, 11)})
        content = self.server.chat.call_args.args[0][1]["content"]
        self.assertEqual(sum(part["type"] == "image_url" for part in content), 10)

    def test_t2i_ignores_stale_projector_selection(self):
        node.QwenImagePrompt.execute("A bicycle", mmproj="missing.gguf")
        self.assertIsNone(self.manager.acquire.call_args.args[1])

    def test_explicit_i2i_projector(self):
        projector = self.root / "I2I" / node.DEFAULT_MMPROJ
        projector.parent.mkdir()
        projector.touch()
        self.server.chat.return_value = response("I2I")
        node.QwenImagePrompt.execute("Edit", mode="I2I", model=node.DEFAULT_MODELS["I2I"], reference_image_1=image(),
                                     mmproj="I2I/" + node.DEFAULT_MMPROJ)
        self.assertEqual(self.manager.acquire.call_args.args[1], projector)

    def test_t2i_never_sends_connected_images_or_projector(self):
        result = node.QwenImagePrompt.execute("A bicycle", reference_image_1=image(),
                                             mmproj=node.DEFAULT_MMPROJ)
        self.assertEqual(result.result[-1], "T2I")
        self.assertIsNone(self.manager.acquire.call_args.args[1])
        content = self.server.chat.call_args.args[0][1]["content"]
        self.assertTrue(all(item["type"] == "text" for item in content))

    def test_i2i_requires_images_and_projector(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            node.QwenImagePrompt.execute("Edit", mode="I2I", model=node.DEFAULT_MODELS["I2I"])
        with self.assertRaisesRegex(ValueError, "Select the I2I mmproj"):
            node.QwenImagePrompt.execute("Edit", mode="I2I", model=node.DEFAULT_MODELS["I2I"], reference_image_1=image())
        with self.assertRaisesRegex(ValueError, "not a model file"):
            node.QwenImagePrompt.execute("Edit", mode="I2I", model=node.DEFAULT_MODELS["I2I"],
                                         reference_image_1=image(), mmproj=node.DEFAULT_MODELS["I2I"])
        self.manager.acquire.assert_not_called()

    def test_other_model_loads_with_and_without_reference_images(self):
        model = self.root / "Other" / "model-Q5_K_M.gguf"
        model.parent.mkdir()
        model.touch()
        projector = model.parent / "mmproj-other-f16.gguf"
        projector.touch()
        schema = node.QwenImagePrompt.define_schema()
        self.assertIn("Other/model-Q5_K_M.gguf", next(f for f in schema.inputs if f.name == "model").options)
        self.assertIn("Other/mmproj-other-f16.gguf", next(f for f in schema.inputs if f.name == "mmproj").options)
        result = node.QwenImagePrompt.execute("Describe", model="Other/model-Q5_K_M.gguf")
        self.assertEqual(result.result[-1], "T2I")
        self.manager.acquire.assert_called_with(model, None)
        with self.assertRaisesRegex(ValueError, "Select the I2I mmproj"):
            node.QwenImagePrompt.execute("Edit", mode="I2I", model="Other/model-Q5_K_M.gguf", reference_image_1=image())
        self.server.chat.return_value = response("I2I")
        result = node.QwenImagePrompt.execute("Edit", mode="I2I", model="Other/model-Q5_K_M.gguf",
                                             mmproj="Other/mmproj-other-f16.gguf", reference_image_1=image("reference"))
        self.assertEqual(result.result[-1], "I2I")
        self.manager.acquire.assert_called_with(model, projector)
        content = self.server.chat.call_args.args[0][1]["content"]
        self.assertTrue(any(part.get("image_url", {}).get("url") == "reference" for part in content))

    def test_auto_and_unknown_selections_rejected(self):
        for selection in ("auto", "unknown"):
            with self.assertRaisesRegex(ValueError, "Unknown model mode"):
                node.QwenImagePrompt.execute("A bicycle", mode=selection)
            with self.assertRaisesRegex(ValueError, "Unknown skill"):
                node.QwenImagePrompt.execute("A bicycle", skill=selection)
        self.manager.acquire.assert_not_called()

    def test_selected_skill_is_used_without_router(self):
        with patch.object(node, "SKILL_NAMES", ("custom-style",)), patch.object(node, "system_prompt", return_value="Custom instructions") as build:
            result = node.QwenImagePrompt.execute("A bicycle", skill="custom-style")
        self.assertEqual(result.result[1], "custom-style")
        self.assertEqual(build.call_args.args[0], "custom-style")
        self.server.chat.assert_called_once()

    def test_mode_overrides_model_filename_and_connected_images(self):
        result = node.QwenImagePrompt.execute("A bicycle", mode="T2I", model=node.DEFAULT_MODELS["I2I"], reference_image_1=image())
        self.assertEqual(result.result[-1], "T2I")
        self.manager.acquire.assert_called_with(self.root / node.DEFAULT_MODELS["I2I"], None)

    def test_more_than_ten_images_rejected(self):
        with self.assertRaisesRegex(ValueError, "At most 10"):
            node._validate_reference_images([image() for _ in range(11)])

    def test_batch_rejected_before_acquiring_server(self):
        with self.assertRaisesRegex(ValueError, "exactly one image"):
            node.QwenImagePrompt.execute("Edit", mode="I2I", model=node.DEFAULT_MODELS["I2I"], reference_image_1=image(batch=2))
        self.manager.acquire.assert_not_called()
        self.manager.release.assert_called_once()

    def test_repairs_invalid_json_once(self):
        self.server.chat.side_effect = [("invalid", {}), response()]
        self.assertEqual(node.QwenImagePrompt.execute("A bicycle").result[0], "A precise image prompt.")
        self.assertEqual(self.server.chat.call_count, 2)

    def test_failure_always_unloads_even_when_reuse_enabled(self):
        self.server.chat.return_value = ("invalid", {})
        with self.assertRaises(ValueError):
            node.QwenImagePrompt.execute("A bicycle", force_unload_model=False)
        self.manager.release.assert_called_once()

    def test_missing_projector_fails_before_server(self):
        (self.root / node.DEFAULT_MMPROJ).unlink()
        with self.assertRaises(FileNotFoundError):
            node.QwenImagePrompt.execute("Edit", mode="I2I", model=node.DEFAULT_MODELS["I2I"], mmproj=node.DEFAULT_MMPROJ, reference_image_1=image())
        self.manager.acquire.assert_not_called()


if __name__ == "__main__":
    unittest.main()
