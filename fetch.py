"""Fetch the pinned Chroma embedding archive with an empty proxy."""

from __future__ import annotations

import hashlib
import io
import tarfile
import urllib.request
from urllib.parse import urlparse

ARCHIVE_URL = "https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz"
ARCHIVE_SHA256 = "913d7300ceae3b2dbc2c50d1de4baacab4be7b9380491c27fab7418616a16ec3"
ARCHIVE_BYTES = 83178821
ARCHIVE_HOST = "chroma-onnx-models.s3.amazonaws.com"
READ_CAP = 120_000_000

_FILES = frozenset(
    {
        "onnx/config.json",
        "onnx/model.onnx",
        "onnx/special_tokens_map.json",
        "onnx/tokenizer_config.json",
        "onnx/tokenizer.json",
        "onnx/vocab.txt",
    }
)


class FetchRefused(Exception):
    def __init__(self, kind: str):
        self.kind = kind
        super().__init__(kind)


class _SameHost(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        old = urlparse(req.full_url)
        new = urlparse(newurl)
        if old.scheme != "https" or new.scheme != "https":
            raise FetchRefused("redirect")
        if (old.hostname or "").lower() != ARCHIVE_HOST:
            raise FetchRefused("redirect")
        if (new.hostname or "").lower() != ARCHIVE_HOST:
            raise FetchRefused("redirect")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def build_opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), _SameHost())


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_body(response, cap: int = READ_CAP) -> bytes:
    chunks = []
    total = 0
    while True:
        block = response.read(1024 * 1024)
        if not block:
            break
        if total + len(block) > cap:
            raise FetchRefused("oversize")
        chunks.append(block)
        total += len(block)
    return b"".join(chunks)


def verify_archive(data: bytes) -> None:
    if len(data) != ARCHIVE_BYTES or sha256_bytes(data) != ARCHIVE_SHA256:
        raise FetchRefused("hash")


def fetch_pinned(opener=None, timeout: int = 60) -> bytes:
    opener = opener or build_opener()
    request = urllib.request.Request(ARCHIVE_URL, method="GET")
    try:
        with opener.open(request, timeout=timeout) as response:
            final = urlparse(response.geturl())
            if final.scheme != "https" or (final.hostname or "").lower() != ARCHIVE_HOST:
                raise FetchRefused("host")
            body = read_body(response)
    except FetchRefused:
        raise
    except Exception as exc:
        raise FetchRefused("fetch") from exc
    verify_archive(body)
    return body


def _clean_name(name: str) -> str:
    text = name.replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text


def member_ok(member: tarfile.TarInfo) -> bool:
    name = _clean_name(member.name)
    if not name or name.startswith("/") or "\x00" in name:
        return False
    parts = [part for part in name.split("/") if part != ""]
    if not parts or any(part in ("..", ".") for part in parts):
        return False
    if member.issym() or member.islnk() or member.isdev() or member.isfifo():
        return False
    if member.linkname:
        return False
    if member.isdir():
        return name.rstrip("/") == "onnx"
    if not member.isfile():
        return False
    return name in _FILES


def assert_members(data: bytes) -> None:
    total = 0
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            if not member_ok(member):
                raise FetchRefused("member")
            if member.isfile():
                total += int(member.size)
                if total > READ_CAP:
                    raise FetchRefused("oversize")
        if not members:
            raise FetchRefused("member")


def extract_members(data: bytes, dest) -> str:
    """Write allowed files under dest. Return the sha256 of model.onnx."""
    from safe_write import write_bytes

    verify_archive(data)
    assert_members(data)
    model_hash = ""
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                raise FetchRefused("member")
            payload = extracted.read()
            name = _clean_name(member.name)
            write_bytes(dest, name, payload, create=True)
            if name == "onnx/model.onnx":
                model_hash = sha256_bytes(payload)
    if not model_hash:
        raise FetchRefused("member")
    marker = f"archive {ARCHIVE_SHA256}\nmodel.onnx {model_hash}\n".encode("ascii")
    write_bytes(dest, "extract.sha256", marker, create=True)
    return model_hash
