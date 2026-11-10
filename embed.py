"""Legacy MiniLM embedder. The class download path stays on the Chroma default."""

from __future__ import annotations

from pathlib import Path

MARKER = "onnx_mini_lm_l6_v2"
_NEEDED = (
    "config.json",
    "model.onnx",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.txt",
)


class MiniLM:
    dimension = 384
    allowed_names = frozenset({"default", "onnx_mini_lm_l6_v2"})
    marker = MARKER

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self._ef = None

    def chroma(self):
        if self._ef is None:
            self._ef = _build(self.model_dir)
        return self._ef


def _build(model_dir: Path):
    from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2

    class _Pinned(ONNXMiniLM_L6_V2):
        def __init__(self) -> None:
            super().__init__()
            self.DOWNLOAD_PATH = Path(model_dir)

        @staticmethod
        def name() -> str:
            return "default"

        def get_config(self):
            return NotImplemented

        @staticmethod
        def build_from_config(config):
            return NotImplemented

        def _download(self, *args, **kwargs):
            raise RuntimeError("archive fetch is disabled")

        def _download_model_if_not_exists(self) -> None:
            folder = Path(self.DOWNLOAD_PATH) / "onnx"
            for name in _NEEDED:
                path = folder / name
                if path.is_symlink() or not path.is_file():
                    raise RuntimeError("embedding archive is incomplete")

    return _Pinned()
