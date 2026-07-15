from __future__ import annotations

import itertools
import random
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

from .config import REPO_ROOT
from .storage import arrow_path, ensure_bucket, s3_filesystem, upload_file


SUPPORTED = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}


@dataclass(frozen=True)
class GeneratedDataset:
    manifest_uri: str
    rows: int
    total_bytes: int
    total_audio_seconds: float


def _duration(path: Path) -> float:
    try:
        info = sf.info(path)
        return float(info.frames) / info.samplerate if info.samplerate else 0.0
    except Exception:
        probe = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(path),
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        return float(probe.stdout.strip())


def _transcode(source: Path, output: Path, duration_s: float) -> None:
    codec = ["-c:a", "pcm_s16le"] if output.suffix == ".wav" else ["-c:a", "libmp3lame", "-b:a", "64k"]
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-stream_loop", "-1", "-i", str(source), "-t", str(duration_s),
            "-ac", "1", "-ar", "16000", *codec, str(output),
        ],
        check=True,
    )


def _mixed_duration(rng: random.Random) -> float:
    bucket = rng.random()
    if bucket < 0.70:
        return rng.uniform(15, 90)
    if bucket < 0.95:
        return rng.uniform(90, 300)
    return rng.uniform(300, 1800)


def generate_dataset(
    *,
    run_id: str,
    bucket: str = "benchmark",
    prefix: str = "audio",
    profile: str = "smoke",
    count: int = 4,
    fixed_duration_s: float = 30.0,
    source_dir: Path | None = None,
    seed: int = 20260715,
) -> GeneratedDataset:
    if count < 1:
        raise ValueError("count must be at least 1")
    if profile not in {"smoke", "fixed", "mixed"}:
        raise ValueError(f"unsupported profile: {profile}")

    source_dir = source_dir or REPO_ROOT / "data" / "audio"
    sources = sorted(p for p in source_dir.iterdir() if p.suffix.lower() in SUPPORTED)
    if not sources:
        raise ValueError(f"No supported seed audio found in {source_dir}")

    fs = s3_filesystem()
    ensure_bucket(fs, bucket)
    rng = random.Random(seed)
    rows: list[dict[str, object]] = []
    total_bytes = 0
    total_seconds = 0.0

    with tempfile.TemporaryDirectory(prefix="audio-benchmark-") as tmp:
        tmp_path = Path(tmp)
        for index, source in zip(range(count), itertools.cycle(sources)):
            if profile == "smoke":
                duration_s = _duration(source)
                generated = source
            else:
                duration_s = fixed_duration_s if profile == "fixed" else _mixed_duration(rng)
                suffix = ".mp3" if rng.random() < 0.8 else ".wav"
                generated = tmp_path / f"audio_{index:06d}{suffix}"
                _transcode(source, generated, duration_s)

            suffix = generated.suffix.lower()
            doc_id = f"{run_id}_{index:06d}{suffix}"
            uri = f"s3://{bucket}/{prefix.strip('/')}/{run_id}/input/{doc_id}"
            size = upload_file(fs, generated, uri)
            rows.append(
                {
                    "doc_id": doc_id,
                    "s3_url": uri,
                    "expected_bytes": size,
                    "expected_duration_s": duration_s,
                }
            )
            total_bytes += size
            total_seconds += duration_s

    manifest_uri = f"s3://{bucket}/{prefix.strip('/')}/{run_id}/manifest.parquet"
    table = pa.Table.from_pylist(rows)
    with fs.open_output_stream(arrow_path(manifest_uri)) as out:
        pq.write_table(table, out)
    return GeneratedDataset(manifest_uri, len(rows), total_bytes, total_seconds)
