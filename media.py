from __future__ import annotations

import base64
import io


import numpy as np
from PIL import Image


def tensor_frame_data_url(image, frame_index: int = 0, max_edge: int = 1280) -> str:
    frame = image[frame_index, ..., :3].detach().cpu().numpy()
    pixels = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
    pil_image = Image.fromarray(pixels, mode="RGB")
    pil_image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=90, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def image_content(image, frame_index: int = 0) -> dict[str, object]:
    return {
        "type": "image_url",
        "image_url": {
            "url": tensor_frame_data_url(image, frame_index),
            "detail": "auto",
        },
    }


def text_content(text: str) -> dict[str, str]:
    return {"type": "text", "text": text}
