"""Read rows from a path: directory of JSONL/Parquet files or a single file."""
import json
import os
from typing import Dict, Iterator, List, Optional

import pyarrow.parquet as pq

PARQUET_BATCH_SIZE = 1024


def _is_data_file(name: str) -> bool:
    lower = name.lower()
    return lower.endswith(".jsonl") or lower.endswith(".json") or lower.endswith(".parquet")


def list_data_files(path: str) -> List[str]:
    """Return sorted list of JSONL/Parquet file paths under path.

    - If path is a file and has .jsonl/.json/.parquet extension, return [path].
    - If path is a directory, return sorted paths of all .jsonl/.json/.parquet files in it (one level).
    """
    path = os.path.abspath(path)
    if os.path.isfile(path):
        if _is_data_file(path):
            return [path]
        raise ValueError(f"Not a data file (expect .jsonl/.parquet): {path}")
    if not os.path.isdir(path):
        raise ValueError(f"Not a file or directory: {path}")
    names = sorted(os.listdir(path))
    out = []
    for name in names:
        if _is_data_file(name):
            out.append(os.path.join(path, name))
    return out


def _format_from_path(file_path: str) -> str:
    path_lower = file_path.lower()
    if path_lower.endswith(".parquet"):
        return "parquet"
    if path_lower.endswith(".jsonl") or path_lower.endswith(".json"):
        return "jsonl"
    return "jsonl"


def _requested_columns(text_key: str, id_key: Optional[str]) -> List[str]:
    columns = [text_key]
    if id_key and id_key not in columns:
        columns.append(id_key)
    return columns


def _columns_for_file(
    file_path: str,
    required_columns: List[str],
    optional_columns: Optional[List[str]] = None,
) -> List[str]:
    fmt = _format_from_path(file_path)
    columns = list(required_columns)
    for c in optional_columns or []:
        if c not in columns:
            columns.append(c)
    if fmt != "parquet" or not optional_columns:
        return columns

    schema_names = set(pq.ParquetFile(file_path).schema_arrow.names)
    missing_required = [c for c in required_columns if c not in schema_names]
    if missing_required:
        raise ValueError(f"Missing required parquet columns in {file_path}: {missing_required}")
    return [c for c in columns if c in schema_names]


def _iter_rows_from_file(file_path: str, parquet_columns: Optional[List[str]] = None) -> Iterator[Dict]:
    """Yield all rows from a single JSONL or Parquet file (no partition)."""
    fmt = _format_from_path(file_path)
    if fmt == "jsonl":
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
        return
    if fmt == "parquet":
        parquet_file = pq.ParquetFile(file_path)
        for batch in parquet_file.iter_batches(
            batch_size=PARQUET_BATCH_SIZE,
            columns=parquet_columns,
            use_threads=True,
        ):
            column_names = batch.schema.names
            columns = {
                name: batch.column(i).to_pylist()
                for i, name in enumerate(column_names)
            }
            for row_idx in range(batch.num_rows):
                yield {name: values[row_idx] for name, values in columns.items()}


def _assigned_files_for_part(files: List[str], part_id: int, total_parts: int) -> List[str]:
    """Assign whole files to a partition when there are enough shards to avoid duplicate IO."""
    if total_parts <= 1:
        return files
    if len(files) < total_parts:
        return []
    return [file_path for idx, file_path in enumerate(files) if idx % total_parts == part_id]


def iter_rows(
    path: str,
    text_key: str,
    id_key: Optional[str] = None,
    rank: int = 0,
    world_size: int = 1,
    worker_id: int = 0,
    num_workers: int = 1,
    extra_columns: Optional[List[str]] = None,
) -> Iterator[Dict]:
    """Yield rows from path (directory or single file). Reads all .jsonl and .parquet under path.

    Rows are partitioned by (rank, worker) so each process gets a disjoint subset.
    """
    files = list_data_files(path)
    if not files:
        raise ValueError(f"No .jsonl or .parquet files found under path: {path}")

    total_parts = world_size * num_workers
    part_id = rank * num_workers + worker_id
    parquet_columns = _requested_columns(text_key=text_key, id_key=id_key)
    if extra_columns:
        for c in extra_columns:
            if c not in parquet_columns:
                parquet_columns.append(c)
    assigned_files = _assigned_files_for_part(files, part_id=part_id, total_parts=total_parts)
    if assigned_files:
        for file_path in assigned_files:
            yield from _iter_rows_from_file(
                file_path,
                parquet_columns=_columns_for_file(
                    file_path,
                    required_columns=parquet_columns,
                    optional_columns=extra_columns,
                ),
            )
        return

    global_index = 0
    for file_path in files:
        columns_for_file = _columns_for_file(
            file_path,
            required_columns=parquet_columns,
            optional_columns=extra_columns,
        )
        for row in _iter_rows_from_file(file_path, parquet_columns=columns_for_file):
            if global_index % total_parts == part_id:
                yield row
            global_index += 1
