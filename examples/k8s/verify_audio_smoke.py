"""验证 Kubernetes 音频 smoke 的行保留、成功状态及 Lance Blob v2 写出。"""

from __future__ import annotations

import sys

import daft

from multimodal_toolkit.storage.blob import validate_blob_v2
from multimodal_toolkit.storage.io import daft_io_config


def main(uri: str, expected_rows: int) -> None:
    validate_blob_v2(uri, "audio_blob")
    rows = daft.read_lance(uri, io_config=daft_io_config()).select("status").to_pydict()["status"]
    if len(rows) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} rows, got {len(rows)}")
    if set(rows) != {"ok"}:
        raise RuntimeError(f"audio smoke contains non-success statuses: {rows}")
    print(f"validated {len(rows)} rows with status=ok and Lance Blob v2")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: verify_audio_smoke.py <lance-uri> <expected-rows>")
    main(sys.argv[1], int(sys.argv[2]))
