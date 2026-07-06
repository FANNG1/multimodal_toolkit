from __future__ import annotations

import pyarrow as pa

from . import config

AUDIO_ASSET_SCHEMA = pa.schema(
    [
        pa.field("doc_id", pa.utf8()),
        pa.field("s3_url", pa.utf8()),
        pa.field("audio_blob", pa.large_binary()),  # written as lance blob v2 at ingest time
        pa.field("ingest_time", pa.timestamp("us", tz="UTC")),
        pa.field("audio_embedding", pa.list_(pa.float32(), config.EMBED_DIM)),
        pa.field("duration_s", pa.float64()),
        pa.field("transcript", pa.utf8()),
        pa.field("acoustic_emotion", pa.utf8()),
        pa.field("downgrade_related", pa.bool_()),
        pa.field("primary_reason", pa.utf8()),
        pa.field("secondary_reason", pa.utf8()),
        pa.field("summary", pa.utf8()),
        pa.field("confidence", pa.float64()),
        pa.field("text_emotion", pa.utf8()),
        pa.field("bad_tone", pa.bool_()),
        pa.field("emotion_score", pa.float64()),
    ]
)
