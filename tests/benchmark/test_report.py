from __future__ import annotations

import csv
import json

from benchmark.audio.report import build_baseline_report, build_report


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


def test_baseline_report_compares_repeat_throughput(tmp_path):
    reports = [
        {
            "run_id": "base-repeat-01",
            "state": "success",
            "summary": {"elapsed_s": 100.0},
            "throughput": {"audio_seconds_per_wall_second": 2.0, "files_per_s": 0.5},
        },
        {
            "run_id": "base-repeat-02",
            "state": "success",
            "summary": {"elapsed_s": 102.0},
            "throughput": {"audio_seconds_per_wall_second": 1.96, "files_per_s": 0.49},
        },
    ]
    result = build_baseline_report(
        tmp_path,
        reports,
        {"run_id": "base", "rows": 50, "duration_s": 60.0},
    )
    assert result["repeats"] == 2
    assert result["passes_repeatability_slo"] is True
    assert result["throughput_cv"] < 0.02
    assert (tmp_path / "baseline-summary.json").exists()
    assert (tmp_path / "baseline-summary.md").exists()
