from __future__ import annotations

import argparse
from pathlib import Path

from . import config
from .asr import get_asr
from .blob import append_columns_by_doc_id, read_audio_blobs, validate_blob_v2
from .io import write_jsonl
from .prompt import analyze_with_llm


def _duration(audio_bytes: bytes | None) -> float:
    import soundfile as sf

    if not audio_bytes:
        return 0.0
    import io

    info = sf.info(io.BytesIO(audio_bytes))
    return float(info.frames) / info.samplerate if info.samplerate else 0.0


def _suffix_from_doc(doc_id: str) -> str:
    suffix = Path(doc_id).suffix
    return suffix if suffix else ".wav"


def run(lance_uri: str, out_jsonl: str) -> None:
    import pyarrow as pa

    validate_blob_v2(lance_uri, "audio_blob")
    blobs = read_audio_blobs(lance_uri)
    asr = get_asr()
    rows: list[dict] = []
    for doc_id, audio_bytes in blobs.items():
        duration_s = _duration(audio_bytes)
        if duration_s < config.MIN_DURATION_S or duration_s > config.MAX_DURATION_S:
            asr_result = {"transcript": "", "acoustic_emotion": "NEUTRAL"}
            analysis = analyze_with_llm("", "NEUTRAL")
        else:
            asr_result = asr.transcribe_bytes(audio_bytes, _suffix_from_doc(doc_id))
            analysis = analyze_with_llm(asr_result["transcript"], asr_result["acoustic_emotion"])

        row = {
            "doc_id": doc_id,
            "duration_s": duration_s,
            "transcript": asr_result["transcript"],
            "acoustic_emotion": asr_result["acoustic_emotion"],
            **analysis,
        }
        rows.append(row)

    write_jsonl(rows, out_jsonl)

    table = pa.Table.from_pylist(
        rows,
        schema=pa.schema(
            [
                pa.field("doc_id", pa.utf8()),
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
        ),
    )
    append_columns_by_doc_id(lance_uri, table)
    print(f"[ok] wrote analysis JSONL: {out_jsonl}")
    print(f"[ok] appended analysis columns to: {lance_uri}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lance-uri", required=True)
    parser.add_argument("--out-jsonl", required=True)
    args = parser.parse_args()
    run(args.lance_uri, args.out_jsonl)


if __name__ == "__main__":
    main()
