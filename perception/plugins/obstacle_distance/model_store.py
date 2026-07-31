"""外部模型文件的校验与原子缓存。"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path


def ensure_model_file(
    url: str,
    destination: str | Path,
    sha256: str | None = None,
) -> Path:
    """确保模型文件存在；下载完成并校验后才替换目标文件。"""

    target = Path(destination)
    expected = sha256.lower() if sha256 else None
    if target.is_file() and (expected is None or _file_sha256(target) == expected):
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.{os.getpid()}.part")
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            with temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)

        actual = _file_sha256(temporary)
        if expected is not None and actual != expected:
            raise ValueError(
                f"模型 SHA256 校验失败: expected={expected}, actual={actual}"
            )
        os.replace(temporary, target)
        return target
    finally:
        temporary.unlink(missing_ok=True)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        while chunk := model_file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
