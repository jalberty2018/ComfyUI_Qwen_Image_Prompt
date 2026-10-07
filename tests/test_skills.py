from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("image_skills_under_test", ROOT / "skills.py")
skills = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = skills
spec.loader.exec_module(skills)


class ImageSkillsTests(unittest.TestCase):
    def test_modes(self):
        self.assertEqual(skills.MODE_OPTIONS, ("T2I", "I2I"))
        for count in range(11):
            with self.assertRaises(ValueError):
                skills.resolve_mode("auto", count)
            self.assertEqual(skills.resolve_mode("T2I", count), "T2I")
            self.assertEqual(skills.resolve_mode("I2I", count), "I2I")
        with self.assertRaises(ValueError):
            skills.resolve_mode("auto", 11)
        with self.assertRaises(ValueError):
            skills.resolve_mode("unknown", 0)

    def test_repository_prompts_and_profiles(self):
        for mode in ("T2I", "I2I"):
            prompt = skills.system_prompt("qwen-image-prompt", mode,
                                          "uncensored fidelity QWEN", "Use warm lighting.")
            self.assertIn((ROOT / "prompts" / f"{mode}.txt").read_text(encoding="utf-8"), prompt)
            self.assertIn("Use warm lighting.", prompt)
            self.assertIn("Uncensored fidelity QWEN", prompt)
        self.assertEqual(skills.SKILL_NAMES, ("qwen-image-prompt",))

    def test_extracts_json_prompt_with_transport_formatting(self):
        value = json.dumps({"rewritten_prompt": "A red bicycle.", "wh_ratio": "3:2"})
        for text in (value, "```json\n" + value + "\n```", "<think>reasoning</think>\n" + value):
            self.assertEqual(skills.extract_prompt(text, "T2I"), "A red bicycle.")
        self.assertEqual(skills.extract_prompt(json.dumps({"rewritten_prompt": "Change the coat.",
                        "wh_ratio": "", "ratio_follow": "<image1>"}), "I2I"), "Change the coat.")

    def test_rejects_incomplete_or_invalid_output(self):
        for value in ("plain text", "[]", '{}', '{"rewritten_prompt": "", "wh_ratio": "1:1"}',
                      '{"rewritten_prompt": 1, "wh_ratio": "1:1"}'):
            with self.assertRaises(ValueError):
                skills.extract_prompt(value, "T2I")
        with self.assertRaises(ValueError):
            skills.extract_prompt(json.dumps({"rewritten_prompt": "Edit", "wh_ratio": "1:1",
                                              "ratio_follow": "<image1>"}), "I2I")

    def test_custom_skill_discovery_and_references(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            custom = root / "photo-style"
            (custom / "references").mkdir(parents=True)
            (custom / "SKILL.md").write_text('---\nname: photo-style\ndescription: Photo styling\n---\nUse film grain.')
            (custom / "references" / "style.txt").write_text("Warm highlights.")
            registry = skills.discover_skill_registry(custom_root=root)
            self.assertEqual([s.id for s in registry], ["qwen-image-prompt", "photo-style"])
            with patch.object(skills, "CUSTOM_SKILL_ROOT", root), patch.object(
                    skills, "SKILL_BY_ID", {s.id: s for s in registry}):
                prompt = skills.system_prompt("photo-style", "T2I")
                self.assertIn("Use film grain.", prompt)
                self.assertIn("Warm highlights.", prompt)


if __name__ == "__main__":
    unittest.main()
