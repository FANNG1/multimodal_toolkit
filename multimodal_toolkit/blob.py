from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

from .io import lance_storage_options

if TYPE_CHECKING:
    import pyarrow as pa


@dataclass
class EngineAttempt:
    engine: str
    ok: bool
    detail: str


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


def list_doc_ids(lance_uri: str) -> list[str]:
    attempts: list[EngineAttempt] = []
    try:
        import daft

        ids = daft.read_lance(lance_uri).select("doc_id").collect().to_pydict()["doc_id"]
        return [str(x) for x in ids]
    except Exception as exc:
        attempts.append(EngineAttempt("daft", False, repr(exc)))

    try:
        import lance_ray as lr

        ids = lr.read_lance(lance_uri, columns=["doc_id"]).to_pandas()["doc_id"].astype(str).tolist()
        return ids
    except Exception as exc:
        attempts.append(EngineAttempt("lance-ray", False, repr(exc)))

    try:
        import lance

        table = lance.dataset(lance_uri, storage_options=lance_storage_options(lance_uri)).to_table(columns=["doc_id"])
        return [str(x) for x in table["doc_id"].to_pylist()]
    except Exception as exc:
        attempts.append(EngineAttempt("lance", False, repr(exc)))
        raise RuntimeError(f"Unable to list doc_id via any engine: {attempts}") from exc


def read_audio_blobs(lance_uri: str, doc_ids: Iterable[str] | None = None) -> dict[str, bytes | None]:
    """Read audio blob bytes.

    Daft remains the primary engine for manifest/S3/write/query scalar work.
    Daft exposes Lance blob v2 columns as descriptor structs, which is useful
    for metadata scans but not for ASR/embedding. This follows the Guangdong
    image POC: materialize blob bodies with Lance native read_blobs/take_blobs.
    """
    wanted = [str(x) for x in doc_ids] if doc_ids is not None else None
    wanted_set = set(wanted) if wanted is not None else None
    attempts: list[EngineAttempt] = []

    try:
        import lance

        ds = lance.dataset(lance_uri, storage_options=lance_storage_options(lance_uri))
        ids = [str(x) for x in ds.to_table(columns=["doc_id"])["doc_id"].to_pylist()]
        indices = [i for i, doc_id in enumerate(ids) if wanted_set is None or doc_id in wanted_set]
        blob_rows = ds.read_blobs("audio_blob", indices=indices, preserve_order=True)
        out = {}
        for row_idx, blob in blob_rows:
            out[ids[row_idx]] = blob
        return out
    except Exception as exc:
        attempts.append(EngineAttempt("lance", False, repr(exc)))
        raise RuntimeError(f"Unable to read blob v2 via Lance native blob API: {attempts}") from exc


def append_columns_by_doc_id(lance_uri: str, table: "pa.Table") -> None:
    import pyarrow as pa

    if "doc_id" not in table.column_names:
        raise ValueError("append table must include doc_id")

    attempts: list[EngineAttempt] = []
    new_cols = [c for c in table.column_names if c != "doc_id"]
    values = table.to_pydict()
    by_id = {
        str(doc_id): {col: values[col][i] for col in new_cols}
        for i, doc_id in enumerate(values["doc_id"])
    }

    try:
        import lance_ray as lr

        schema = pa.schema([table.schema.field(c) for c in new_cols])

        def transform(batch: pa.RecordBatch) -> pa.RecordBatch:
            ids = [str(x) for x in batch.column("doc_id").to_pylist()]
            arrays = []
            for col in new_cols:
                arrays.append(pa.array([by_id.get(doc_id, {}).get(col) for doc_id in ids], type=schema.field(col).type))
            return pa.record_batch(arrays, schema=schema)

        opts = lance_storage_options(lance_uri)
        prev = os.environ.get("LANCE_STORAGE_OPTIONS")
        if opts:
            os.environ["LANCE_STORAGE_OPTIONS"] = json.dumps(opts)
        try:
            lr.add_columns(lance_uri, transform=transform, read_columns=["doc_id"])
        finally:
            if opts:
                if prev is None:
                    os.environ.pop("LANCE_STORAGE_OPTIONS", None)
                else:
                    os.environ["LANCE_STORAGE_OPTIONS"] = prev
        return
    except Exception as exc:
        attempts.append(EngineAttempt("lance-ray", False, repr(exc)))

    try:
        import lance

        ids = list_doc_ids(lance_uri)
        schema = pa.schema([table.schema.field(c) for c in new_cols])
        arrays = []
        for col in new_cols:
            arrays.append(pa.array([by_id.get(doc_id, {}).get(col) for doc_id in ids], type=schema.field(col).type))
        out = pa.table(dict(zip(new_cols, arrays, strict=False)), schema=schema)
        reader = pa.RecordBatchReader.from_batches(schema, out.to_batches())
        lance.dataset(lance_uri, storage_options=lance_storage_options(lance_uri)).add_columns(reader)
        return
    except Exception as exc:
        attempts.append(EngineAttempt("lance", False, repr(exc)))
        raise RuntimeError(f"Unable to append columns by doc_id: {attempts}") from exc


