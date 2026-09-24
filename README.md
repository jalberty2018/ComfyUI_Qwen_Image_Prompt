# ComfyUI Qwen Image 2.1 Prompt

Local prompt enhancement for Qwen-Image 2.1 using the two Heretic GGUF prompt rewriters and an external llama.cpp **b11115** executable. Add **Qwen Image 2.1 Prompt (Local)** to your workflow.

## Installation

From your ComfyUI directory:

```sh
git clone https://github.com/jalberty2018/ComfyUI_Qwen_Image_Prompt.git custom_nodes/ComfyUI_Qwen_Image_Prompt
```

Download the models and configure llama.cpp as described below, then restart ComfyUI.

## Modes and inputs

The mode is automatic; there is no `model_mode` selector:

- Empty reference-image inputs: T2I, with no mmproj passed to llama.cpp.
- One to ten reference images: I2I, with the matching mmproj.
- `skill` contains only `qwen-image-prompt`; no automatic skill routing is used.
- Leave `mmproj` empty for T2I. For I2I, select the projector or leave it empty to discover the matching file beside the I2I model automatically.

There are exactly ten optional IMAGE inputs, `reference_image_1` through `reference_image_10`. Disconnected inputs and inputs returning `None` are skipped. Each supplied tensor must hold one image, not a batch. Nonempty inputs are numbered consecutively as `<image1>`, `<image2>`, etc., in socket order. Refer to these consecutive numbers in your request. Video and audio inputs are not supported.

The three STRING outputs retain their existing names and order: `h3_prompt`, `selected_skill`, `detected_mode`. The historical first name is retained only for output compatibility; its value is the extracted `rewritten_prompt` for Qwen Image. The third output is `T2I` or `I2I`. Aspect-ratio metadata is parsed internally and is not exposed as an additional output.

This node uses a distinct `QwenImagePromptLocal` ID so it can coexist with the original node. Replace the old node in existing workflows and reconnect the three outputs; the old video widgets and serialized widget positions are not compatible. After upgrading from a version with `model_mode`, recreate the node to avoid shifted saved widget values.

## Models

Place models anywhere below `ComfyUI/models/LLM`. The two model-file selectors show only the corresponding model family's Q4_K_M, Q6_K and Q8_0 variants, using relative paths to distinguish duplicate filenames. Only the model for the selected mode must be installed.

| Mode | Repository | Default file |
|---|---|---|
| T2I | [PE-T2I-Heretic-GGUF](https://huggingface.co/pottokao/Qwen-Image-2.1-PE-T2I-Heretic-GGUF) | `pe_t2i_heretic-Q4_K_M.gguf` |
| I2I | [PE-I2I-Heretic-GGUF](https://huggingface.co/pottokao/Qwen-Image-2.1-PE-I2I-Heretic-GGUF) | `pe_i2i_heretic-Q4_K_M.gguf` |

For I2I images, place `pe_i2i_heretic.mmproj-bf16.gguf` beside the I2I model. The node first checks that directory, then searches `models/LLM` for one unambiguous matching projector.

From the ComfyUI directory:

```sh
hf download pottokao/Qwen-Image-2.1-PE-T2I-Heretic-GGUF pe_t2i_heretic-Q4_K_M.gguf --local-dir models/LLM/Qwen-Image-2.1-PE-T2I-Heretic-GGUF
hf download pottokao/Qwen-Image-2.1-PE-I2I-Heretic-GGUF pe_i2i_heretic-Q4_K_M.gguf pe_i2i_heretic.mmproj-bf16.gguf --local-dir models/LLM/Qwen-Image-2.1-PE-I2I-Heretic-GGUF
```

The repository-specific system prompts are bundled in `prompts/T2I.txt` and `prompts/I2I.txt`, with their respective licenses. These model instructions define the JSON response format. The node extracts the prompt and retries once to repair malformed output.

## External llama.cpp b11115

Set the executable path **before starting ComfyUI**. This points to a local executable, not an HTTP URL or an already running server.

Linux:

```sh
export QWEN_IMAGE_LLAMA_SERVER=/opt/llama.cpp/bin/llama-server
```

PowerShell:

```powershell
$env:QWEN_IMAGE_LLAMA_SERVER = 'C:\llama.cpp\b11115\llama-server.exe'
```

Keep the executable's runtime libraries available through the external installation's normal library search paths. The node starts its own process on a free loopback port. It does not install or replace the external executable. Without this environment variable, the existing `runtime_config.json` loader and `python install_runtime.py` fallback remain available; the installer is pinned to b11115.

Seed mapping, thinking/reasoning controls, sampling presets, token budget, model reuse and force unload are retained. `force_unload_model` defaults to true; errors always stop the node-owned server and clear the cache. Switching model, projector or runtime causes a reload.

`prompt_profile` offers `standard` and `uncensored fidelity QWEN`. Additional system instructions are retained. The node uses only the bundled `qwen-image-prompt` skill.

## Checks

```sh
python -m unittest discover -s tests -v
```

The tests use mocked ComfyUI and llama.cpp interfaces to check routing, image forwarding, output parsing, model selection and cleanup. A live GPU inference check requires the GGUF files, ComfyUI and the external executable.
