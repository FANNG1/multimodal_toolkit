from __future__ import annotations

import pyarrow as pa

from . import config

IMAGE_ASSET_SCHEMA = pa.schema(
    [
        pa.field("doc_id", pa.utf8()),
        pa.field("s3_url", pa.utf8()),
        pa.field("image_blob", pa.large_binary()),  # written as lance blob v2 at ingest time
        pa.field("ingest_time", pa.timestamp("us", tz="UTC")),
        pa.field("status", pa.utf8()),  # ok / download_failed / decode_failed / blob_download_failed
        pa.field("width", pa.int32()),
        pa.field("height", pa.int32()),
        pa.field("face_count", pa.int32()),
        pa.field("face_score", pa.float64()),
        pa.field("face_area_ratio", pa.float64()),
        pa.field("blur_score", pa.float64()),
        pa.field("face_blur_score", pa.float64()),
        pa.field("has_face", pa.bool_()),
        pa.field("is_blurry", pa.bool_()),
        pa.field("is_face_blurry", pa.bool_()),
        pa.field("image_embedding", pa.list_(pa.float32(), config.IMAGE_EMBED_DIM)),
    ]
)
