from __future__ import annotations

import csv
import os
import socket
import threading
import time
from pathlib import Path
from typing import Any


def _node_probe_class():
    import ray

    @ray.remote(num_cpus=0)
    class NodeProbe:
        def sample(self) -> dict[str, Any]:
            import psutil
            import ray

            vm = psutil.virtual_memory()
            ray_rss = 0
            for proc in psutil.process_iter(["memory_info", "cmdline"]):
                try:
                    cmdline = " ".join(proc.info.get("cmdline") or ())
                    if "ray" in cmdline.lower() or "daft" in cmdline.lower():
                        ray_rss += proc.info["memory_info"].rss
                except (psutil.AccessDenied, psutil.NoSuchProcess, TypeError):
                    continue
            object_store_used = None
            try:
                from ray._private.internal_api import get_store_used_memory

                object_store_used = int(get_store_used_memory())
            except Exception:
                pass
            return {
                "timestamp": time.time(),
                "hostname": socket.gethostname(),
                "node_id": ray.get_runtime_context().get_node_id(),
                "pid": os.getpid(),
                "cpu_percent": psutil.cpu_percent(interval=None),
                "memory_total_bytes": vm.total,
                "memory_used_bytes": vm.used,
                "memory_percent": vm.percent,
                "ray_process_rss_bytes": ray_rss,
                "object_store_used_bytes": object_store_used,
            }

    return NodeProbe


class ResourceSampler:
    def __init__(self, output: Path, interval_s: float = 5.0):
        self.output = output
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._actors: list[Any] = []

    def start(self) -> None:
        import ray
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        probe_cls = _node_probe_class()
        for node in ray.nodes():
            if not node.get("Alive"):
                continue
            self._actors.append(
                probe_cls.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(node["NodeID"], soft=False)
                ).remote()
            )
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, name="benchmark-resource-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        import ray

        self._stop.set()
        if self._thread:
            self._thread.join(timeout=max(10.0, self.interval_s * 2))
        for actor in self._actors:
            try:
                ray.kill(actor, no_restart=True)
            except Exception:
                pass
        self._actors.clear()

    def _run(self) -> None:
        import ray

        fieldnames = [
            "timestamp", "hostname", "pid", "cpu_percent", "memory_total_bytes",
            "node_id",
            "memory_used_bytes", "memory_percent", "ray_process_rss_bytes",
            "object_store_used_bytes",
        ]
        with self.output.open("w", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=fieldnames)
            writer.writeheader()
            while not self._stop.is_set():
                try:
                    rows = ray.get([actor.sample.remote() for actor in self._actors], timeout=self.interval_s)
                    writer.writerows(rows)
                    out.flush()
                except Exception:
                    # Metrics are diagnostic and must never fail the benchmark query.
                    pass
                self._stop.wait(self.interval_s)
