from __future__ import annotations

import atexit
import gc
import logging
import threading
import time
from pathlib import Path

import comfy.model_management
import folder_paths
from comfy_api.latest import ComfyExtension, io

from .media import image_content, text_content
from .model_files import DEFAULT_MODELS, DEFAULT_MMPROJ, model_options, resolve_gguf
from .runtime import LlamaServerManager
from .skills import (
    MODE_OPTIONS, PROMPT_PROFILES, PROMPT_PROFILE_STANDARD, SKILL_NAMES,
    DEFAULT_SKILL_ID, resolve_mode, router_prompt, parse_skill_selection,
    system_prompt, extract_prompt,
)

MODEL_DIR = Path(folder_paths.models_dir) / "LLM"
INFERENCE_LOCK = threading.Lock()
SERVER_MANAGER = LlamaServerManager()
LOGGER = logging.getLogger("ComfyUI.QwenImagePrompt")
atexit.register(SERVER_MANAGER.release)


def _release_node_resources():
    SERVER_MANAGER.release()
    gc.collect()
    comfy.model_management.soft_empty_cache(force=True)


def _validate_reference_images(images):
    if len(images) > 10:
        raise ValueError(f"At most 10 reference images are supported; got {len(images)}")
    for index, image in enumerate(images, 1):
        if len(image.shape) != 4 or int(image.shape[0]) != 1:
            raise ValueError(f"Reference image {index} must contain exactly one image (BHWC)")
        if int(image.shape[-1]) not in (3, 4) or min(image.shape[1:3]) < 1:
            raise ValueError(f"Reference image {index} must be a non-empty RGB or RGBA image")


def _sampling_settings(think_mode):
    return dict(
        temperature=1.0 if think_mode else 0.7,
        top_p=0.95 if think_mode else 0.8,
        top_k=20, min_p=0.0,
        presence_penalty=0.0 if think_mode else 1.5,
        repetition_penalty=1.0,
    )


def _user_content(prompt, images):
    content = []
    for index, image in enumerate(images, 1):
        content.extend([text_content(f"<image{index}>"), image_content(image)])
    content.append(text_content(prompt))
    return content


class QwenImagePrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="QwenImagePromptLocal",
            display_name="Qwen Image 2.1 Prompt (Local)",
            category="😺dzNodes/Qwen_Image_Prompt",
            description="Qwen-Image 2.1 T2I / I2I prompt enhancer using llama.cpp b11115.",
            inputs=[
                io.String.Input("prompt", multiline=True, dynamic_prompts=True,
                                default="Describe the image or the desired edit."),
                io.Combo.Input("model_mode", options=list(MODE_OPTIONS), default="auto",
                               tooltip="Auto: T2I without images, I2I with images. T2I ignores connected images."),
                io.Combo.Input("skill", options=["auto", *SKILL_NAMES], default="auto"),
                io.Combo.Input("t2i_model", options=model_options(MODEL_DIR, "T2I")),
                io.Combo.Input("i2i_model", options=model_options(MODEL_DIR, "I2I")),
                io.Boolean.Input("think_mode", default=False),
                io.Combo.Input("reasoning_effort", options=["low", "medium", "xhigh"], default="medium",
                               tooltip="Applied only in thinking mode."),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, step=1,
                             control_after_generate=True),
                io.Int.Input("max_tokens", default=8192, min=256, max=8192, step=128),
                io.Boolean.Input("force_unload_model", default=True,
                                 tooltip="Stop this node's llama.cpp process after each run. Errors always unload."),
                io.Combo.Input("prompt_profile", options=list(PROMPT_PROFILES), default=PROMPT_PROFILE_STANDARD),
                io.String.Input("additional_system_instructions", multiline=True, dynamic_prompts=True, default=""),
                *[io.Image.Input(f"reference_image_{index}", optional=True,
                                 tooltip="Optional single image. Empty inputs are skipped; batches are not supported.")
                  for index in range(1, 11)],
            ],
            # Keep the existing three output names, types and order for downstream links.
            outputs=[io.String.Output("h3_prompt"), io.String.Output("selected_skill"),
                     io.String.Output("detected_mode")],
        )

    @classmethod
    def execute(cls, prompt, model_mode="auto", skill="auto",
                t2i_model=DEFAULT_MODELS["T2I"], i2i_model=DEFAULT_MODELS["I2I"],
                think_mode=False, reasoning_effort="medium", seed=0, max_tokens=8192,
                force_unload_model=True, prompt_profile=PROMPT_PROFILE_STANDARD,
                additional_system_instructions="",
                reference_image_1=None, reference_image_2=None, reference_image_3=None,
                reference_image_4=None, reference_image_5=None, reference_image_6=None,
                reference_image_7=None, reference_image_8=None, reference_image_9=None,
                reference_image_10=None) -> io.NodeOutput:
        started = time.perf_counter()
        # Keep cleanup inside the same lock as inference, including error cleanup.
        with INFERENCE_LOCK:
            try:
                if not prompt.strip():
                    raise ValueError("prompt must not be empty")
                images = [image for image in (
                    reference_image_1, reference_image_2, reference_image_3,
                    reference_image_4, reference_image_5, reference_image_6,
                    reference_image_7, reference_image_8, reference_image_9,
                    reference_image_10,
                ) if image is not None]
                mode = resolve_mode(model_mode, len(images))
                if mode == "T2I":
                    images = []
                _validate_reference_images(images)
                model_name = t2i_model if mode == "T2I" else i2i_model
                if Path(model_name).name not in {
                    f"pe_{mode.lower()}_heretic-{quant}.gguf" for quant in ("Q4_K_M", "Q6_K", "Q8_0")
                }:
                    raise ValueError(f"Select a supported {mode} Heretic GGUF model")
                model = resolve_gguf(MODEL_DIR, model_name, LOGGER)
                projector = None
                if mode == "I2I" and images:
                    # Prefer the matching projector beside the selected I2I model.
                    sibling = model.parent / DEFAULT_MMPROJ
                    projector = resolve_gguf(
                        MODEL_DIR,
                        sibling.relative_to(MODEL_DIR.resolve()).as_posix()
                        if sibling.is_file() else DEFAULT_MMPROJ, LOGGER,
                    )
                comfy.model_management.unload_all_models()
                comfy.model_management.soft_empty_cache()
                server, loaded_new = SERVER_MANAGER.acquire(model, projector)
                LOGGER.info("[Qwen Image] mode=%s images=%d model=%s new=%s seed=%d",
                            mode, len(images), model.name, loaded_new, seed)
                selected = skill
                if selected == "auto":
                    selected = DEFAULT_SKILL_ID
                    if len(SKILL_NAMES) > 1:
                        selection, _ = server.chat(
                            router_prompt(prompt, mode), seed=seed, max_tokens=48,
                            temperature=0.0, top_p=1.0, top_k=1, min_p=0.0,
                            presence_penalty=0.0, repetition_penalty=1.0,
                            think_mode=False, reasoning_effort="low")
                        selected = parse_skill_selection(selection)
                messages = [
                    {"role": "system", "content": system_prompt(
                        selected, mode, prompt_profile, additional_system_instructions)},
                    {"role": "user", "content": _user_content(prompt, images)},
                ]
                settings = dict(seed=seed, max_tokens=max_tokens, think_mode=think_mode,
                                reasoning_effort=reasoning_effort, **_sampling_settings(think_mode))
                result, usage = server.chat(messages, **settings)
                try:
                    result = extract_prompt(result, mode)
                except ValueError as error:
                    LOGGER.warning("[Qwen Image] Repairing model output: %s", error)
                    repaired, usage = server.chat([
                        *messages, {"role": "assistant", "content": result},
                        {"role": "user", "content": "Return the complete valid JSON object required "
                         "by the system prompt. Preserve the requested content. Problem: " + str(error)},
                    ], **settings)
                    result = extract_prompt(repaired, mode)
                LOGGER.info("[Qwen Image] completed in %.2fs | usage=%s", time.perf_counter() - started, usage)
                if force_unload_model:
                    _release_node_resources()
                return io.NodeOutput(result, selected, mode)
            except Exception:
                try:
                    _release_node_resources()
                except Exception:
                    LOGGER.exception("[Qwen Image] Cleanup failed")
                LOGGER.exception("[Qwen Image] Execution failed")
                raise


class QwenImagePromptExtension(ComfyExtension):
    async def get_node_list(self):
        return [QwenImagePrompt]


async def comfy_entrypoint() -> QwenImagePromptExtension:
    return QwenImagePromptExtension()
