"""Stage 5：按 ingest_time 范围删除 Lance 资产表中的行。

表管理仍遵循项目的数据 API 优先级：
  delete        → pylance ds.delete()               （Daft/lance-ray 没有等价 API）
  compact       → 暂不执行；升级 pylance 后仍需单独验证 lance-ray 和对象存储路径
  cleanup       → pylance ds.cleanup_old_versions() （没有替代 API）

  --before DATE   删除 ingest_time < DATE 的行
  --after  DATE   删除 ingest_time > DATE 的行

DATE 使用 ISO 8601 格式，例如 2025-01-01 或 2025-01-01T00:00:00。
至少传一个边界；两个边界可以组合成日期范围。
"""
from __future__ import annotations

import argparse

import lance

from ..storage.io import lance_storage_options


def delete_by_date(
    lance_uri: str,
    before: str | None = None,
    after: str | None = None,
) -> None:
    if not before and not after:
        raise ValueError("Provide at least one of: --before or --after")

    clauses = []
    if before:
        clauses.append(f"ingest_time < timestamp '{before}'")
    if after:
        clauses.append(f"ingest_time > timestamp '{after}'")
    filter_str = " AND ".join(clauses)

    # 删除目前只有 pylance 提供 API；这里保留直接调用，不能为了统一入口而绕过
    # Lance 自身的事务提交语义。
    ds = lance.dataset(lance_uri, storage_options=lance_storage_options(lance_uri))
    ds.delete(filter_str)

    print(f"[ok] deleted rows where: {filter_str}")

    # pylance 9.0.0 已修复 Blob v2 compaction 的解码问题，但版本升级本身不等于
    # 分布式 compaction 链路已经验证。这里继续跳过，待后续改动同时覆盖 lance-ray、
    # 本地 Lance URI 和 MinIO/S3 后再开启，避免依赖升级悄悄扩大表管理行为。
    print("[warn] compaction skipped; enable after validating lance-ray and object storage")
    ds.cleanup_old_versions()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lance-uri", required=True, help="lance asset table URI (S3)")
    parser.add_argument("--before", help="delete rows with ingest_time before this date (ISO 8601)")
    parser.add_argument("--after", help="delete rows with ingest_time after this date (ISO 8601)")
    args = parser.parse_args()
    delete_by_date(args.lance_uri, args.before, args.after)


if __name__ == "__main__":
    main()
