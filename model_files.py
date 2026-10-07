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


def is_projector(name: str) -> bool:
    return "mmproj" in Path(name).name.lower()


def model_options(model_root: Path, mode: str | None = None) -> list[str]:
    root = model_root.resolve()
    modes = (mode,) if mode else ("T2I", "I2I")
    options = {path.resolve().relative_to(root).as_posix()
               for path in root.rglob("*")
               if not is_projector(path.name) and _is_usable_gguf(path.resolve(), root)}
    return sorted(options, key=str.casefold) or [DEFAULT_MODELS[family] for family in modes]


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


def projector_options(model_root: Path) -> list[str]:
    root = model_root.resolve()
    options = {path.resolve().relative_to(root).as_posix()
               for path in root.rglob("*")
               if is_projector(path.name) and _is_usable_gguf(path.resolve(), root)}
    return ["", *sorted(options, key=str.casefold)]
