from __future__ import annotations

import csv
import json

from benchmark.audio.report import build_report


def test_failed_run_report_does_not_require_lance(tmp_path):
    (tmp_path / "summary.json").write_text(
        json.dumps({"run_id": "run-1", "state": "failed", "elapsed_s": 2.5, "error": "boom"})
    )
    (tmp_path / "metadata.json").write_text(json.dumps({"versions": {"daft": "test"}}))
    with (tmp_path / "resources.csv").open("w", newline="") as out:
        writer = csv.DictWriter(
            out,
            fieldnames=[
                "timestamp", "hostname", "memory_total_bytes", "memory_used_bytes",
                "ray_process_rss_bytes", "cpu_percent", "object_store_used_bytes",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "timestamp": 1, "hostname": "node", "memory_total_bytes": 1000,
                "memory_used_bytes": 100, "ray_process_rss_bytes": 50,
                "cpu_percent": 10, "object_store_used_bytes": 20,
            }
        )
        writer.writerow(
            {
                "timestamp": 2, "hostname": "node", "memory_total_bytes": 1000,
                "memory_used_bytes": 200, "ray_process_rss_bytes": 75,
                "cpu_percent": 20, "object_store_used_bytes": 30,
            }
        )

    report = build_report(tmp_path)
    assert report["state"] == "failed"
    assert report["resources"]["node_memory_used_peak_bytes"] == 200
    assert report["resources"]["nodes"]["node"]["memory_used_p95_bytes"] == 195
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "report.md").exists()
