from __future__ import annotations

from .io import lance_storage_options


class BlobV2Error(RuntimeError):
    pass


def validate_blob_v2(lance_uri: str, column: str = "audio_blob") -> None:
    import lance

    schema = lance.dataset(lance_uri, storage_options=lance_storage_options(lance_uri)).schema
    field = schema.field(column)
    field_repr = str(field.type)
    if "lance.blob" not in field_repr:
        raise BlobV2Error(
            f"{column} is not Lance blob v2. Actual type: {field_repr}. "
            "Do not continue or silently downgrade to large_binary."
        )
