from __future__ import annotations

import logging
from pathlib import Path

DEFAULT_MODELS = {
    "T2I": "pe_t2i_heretic-Q4_K_M.gguf",
    "I2I": "pe_i2i_heretic-Q4_K_M.gguf",
}
DEFAULT_MMPROJ = "pe_i2i_heretic.mmproj-bf16.gguf"


def _is_usable_gguf(path: Path, root: Path) -> bool:
    return (path.is_file() and path.suffix.lower() == ".gguf"
            and path.is_relative_to(root) and ".cache" not in path.relative_to(root).parts)


def model_options(model_root: Path, mode: str) -> list[str]:
    root = model_root.resolve()
    allowed = {f"pe_{mode.lower()}_heretic-{quant}.gguf" for quant in ("Q4_K_M", "Q6_K", "Q8_0")}
    options = {path.resolve().relative_to(root).as_posix()
               for path in root.rglob("*.gguf")
               if path.name in allowed and _is_usable_gguf(path.resolve(), root)}
    return sorted(options, key=str.casefold) or [DEFAULT_MODELS[mode]]


def resolve_gguf(model_dir: Path, name: str, logger: logging.Logger | None = None) -> Path:
    root = model_dir.resolve()
    path = (root / name).resolve()
    if _is_usable_gguf(path, root):
        return path
    if Path(name).name == name and root.is_dir():
        matches = sorted({p.resolve() for p in root.rglob(name)
                          if _is_usable_gguf(p.resolve(), root)})
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise FileNotFoundError(f"Ambiguous GGUF filename: {name}; select a relative path")
    raise FileNotFoundError(f"Invalid or missing GGUF model in {root}: {name}")
