import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
SRC = os.environ.get("HERMES_SRC", "")
for entry in (SRC, str(ROOT), str(TESTS)):
    if entry and entry not in sys.path:
        sys.path.insert(0, entry)


@pytest.fixture(autouse=True)
def _block_chroma_archive_download(monkeypatch):
    try:
        from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2
    except ImportError:
        return

    def _refuse(self, *args, **kwargs):
        raise AssertionError("download")

    monkeypatch.setattr(ONNXMiniLM_L6_V2, "_download", _refuse)
