# Custom image prompt Skills

Place each prompt-only Skill in its own subdirectory with a `SKILL.md`, then restart ComfyUI. Optional YAML front matter supports `name`, `description`, `display-name` and `version`; `meta.yaml` also supports tags. IDs use lowercase letters, numbers, periods, underscores and hyphens and cannot be `auto`. Duplicate IDs cannot override bundled Skills.

Markdown and text files under `references/` are loaded as additional guidance. No scripts or tools are executed. Use Skills for image style or editing instructions. The selected model's JSON contract and reference-image facts take precedence over Skill instructions. The node extracts `rewritten_prompt` into its first output.

`skill=auto` uses the bundled `qwen-image-prompt` Skill when it is the only one. With custom Skills installed, a short routing request selects a Skill from their descriptions.
