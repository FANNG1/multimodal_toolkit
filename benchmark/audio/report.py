from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from multimodal_toolkit.storage.io import lance_storage_options

from .config import write_json


def _percentile(values: Iterable[float], q: float) -> float | None:
    clean = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return float(np.percentile(clean, q)) if clean else None


def _memory_slope_bytes_per_hour(rows: list[dict[str, str]]) -> float | None:
    if len(rows) < 2:
        return None
    x = np.asarray([float(row["timestamp"]) for row in rows], dtype=np.float64)
    y = np.asarray([float(row["memory_used_bytes"]) for row in rows], dtype=np.float64)
    x -= x[0]
    if x[-1] <= 0:
        return None
    return float(np.polyfit(x, y, 1)[0] * 3600)


def build_report(run_dir: Path) -> dict:
    import lance

    summary = json.loads((run_dir / "summary.json").read_text())
    metadata = json.loads((run_dir / "metadata.json").read_text())
    report = {"run_id": summary["run_id"], "state": summary["state"], "summary": summary}
    if summary["state"] == "success":
        ds = lance.dataset(summary["lance_uri"], storage_options=lance_storage_options(summary["lance_uri"]))
        table = ds.to_table(
            columns=["asr_ms", "llm_ms", "input_bytes", "duration_s", "llm_attempts", "status"]
        )
        elapsed = summary["elapsed_s"]
        report["throughput"] = {
            "files_per_s": summary["rows"] / elapsed if elapsed else None,
            "mb_per_s": summary["input_bytes"] / elapsed / 1_000_000 if elapsed else None,
            "audio_seconds_per_wall_second": summary["audio_seconds"] / elapsed if elapsed else None,
        }
        asr_values = [x for x in table.column("asr_ms").to_pylist() if x is not None and x > 0]
        attempts = table.column("llm_attempts").to_pylist()
        llm_values = [
            value
            for value, attempt in zip(table.column("llm_ms").to_pylist(), attempts)
            if value is not None and attempt is not None and attempt > 0
        ]
        report["latency_ms"] = {
            "asr_ms": {"p50": _percentile(asr_values, 50), "p95": _percentile(asr_values, 95), "p99": _percentile(asr_values, 99)},
            "llm_ms": {"p50": _percentile(llm_values, 50), "p95": _percentile(llm_values, 95), "p99": _percentile(llm_values, 99)},
        }
        report["llm_retry_rows"] = sum(x > 1 for x in attempts if x)

    resource_rows = []
    resource_path = run_dir / "resources.csv"
    if resource_path.exists():
        with resource_path.open(newline="") as inp:
            resource_rows = list(csv.DictReader(inp))
    if resource_rows:
        used = [float(row["memory_used_bytes"]) for row in resource_rows]
        rss = [float(row["ray_process_rss_bytes"]) for row in resource_rows]
        report["resources"] = {
            "samples": len(resource_rows),
            "node_memory_used_peak_bytes": max(used),
            "node_memory_used_p95_bytes": _percentile(used, 95),
            "ray_process_rss_peak_bytes": max(rss),
            "ray_process_rss_p95_bytes": _percentile(rss, 95),
            "cpu_percent_p95": _percentile(
                [float(row["cpu_percent"]) for row in resource_rows if row.get("cpu_percent")], 95
            ),
            "object_store_used_peak_bytes": max(
                [float(row["object_store_used_bytes"]) for row in resource_rows if row.get("object_store_used_bytes")]
                or [0.0]
            ),
        }
        by_host: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in resource_rows:
            by_host[row.get("node_id") or row.get("hostname", "unknown")].append(row)
        node_reports = {}
        for hostname, rows in by_host.items():
            node_used = [float(row["memory_used_bytes"]) for row in rows]
            total = max(float(row["memory_total_bytes"]) for row in rows)
            peak = max(node_used)
            p95 = _percentile(node_used, 95)
            slope = _memory_slope_bytes_per_hour(rows)
            node_reports[hostname] = {
                "memory_total_bytes": total,
                "memory_used_peak_bytes": peak,
                "memory_used_p95_bytes": p95,
                "memory_growth_bytes_per_hour": slope,
                "passes_memory_slo": bool(
                    p95 is not None
                    and p95 <= total * 0.80
                    and peak <= total * 0.90
                    and (slope is None or slope < 500_000_000)
                ),
            }
        report["resources"]["nodes"] = node_reports
        report["resources"]["passes_memory_slo"] = all(
            node["passes_memory_slo"] for node in node_reports.values()
        )

    report["versions"] = metadata.get("versions", {})
    write_json(run_dir / "report.json", report)
    _write_markdown(run_dir / "report.md", report)
    return report


def _write_markdown(path: Path, report: dict) -> None:
    summary = report["summary"]
    lines = [
        f"# Audio benchmark {report['run_id']}",
        "",
        f"- State: `{report['state']}`",
        f"- Elapsed: {summary.get('elapsed_s', 0):.2f}s",
        f"- Rows: {summary.get('rows', 0)}",
        f"- Status: `{json.dumps(summary.get('status_counts', {}), ensure_ascii=False)}`",
    ]
    throughput = report.get("throughput")
    if throughput:
        lines.extend(
            [
                "",
                "## Throughput",
                "",
                f"- Files/s: {throughput['files_per_s']:.4f}",
                f"- MB/s: {throughput['mb_per_s']:.4f}",
                f"- Audio seconds / wall second: {throughput['audio_seconds_per_wall_second']:.4f}",
            ]
        )
    latency = report.get("latency_ms")
    if latency:
        lines.extend(["", "## Latency", ""])
        for stage, values in latency.items():
            rendered = ", ".join(
                f"{name.upper()}={value:.2f}ms" for name, value in values.items() if value is not None
            )
            lines.append(f"- {stage}: {rendered or 'no executed rows'}")
    resources = report.get("resources")
    if resources:
        lines.extend(
            [
                "",
                "## Resources",
                "",
                f"- Ray process RSS peak: {resources['ray_process_rss_peak_bytes'] / 1_000_000_000:.2f}GB",
                f"- Node memory SLO: `{'PASS' if resources['passes_memory_slo'] else 'FAIL'}`",
                f"- CPU P95: {resources.get('cpu_percent_p95') or 0:.2f}%",
                f"- Object store peak: {resources.get('object_store_used_peak_bytes') or 0:.0f} bytes",
            ]
        )
    path.write_text("\n".join(lines) + "\n")
