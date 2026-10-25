import argparse
import cProfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
import hashlib
import io
import logging
import os
from pathlib import Path
import pstats
import threading
import time
from typing import Any, Callable, Dict, List, Optional
from openai import OpenAI
import pymilvus

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Lock for thread-safe writes to local Milvus Lite database file
milvus_lock = threading.Lock()


class FunctionProfiler:
    """Thread-safe tracker for execution time of each profiled function and indexing throughput."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.total_bytes_processed: int = 0
        self.total_chunks_embedded: int = 0
        self.embed_durations: List[float] = []
        self.insert_durations: List[float] = []

    def record(self, func_name: str, elapsed_sec: float) -> None:
        with self._lock:
            self.timings[func_name].append(elapsed_sec)

    def record_embedding(self, num_chunks: int, num_bytes: int, elapsed_sec: float) -> None:
        with self._lock:
            self.total_chunks_embedded += num_chunks
            self.total_bytes_processed += num_bytes
            self.embed_durations.append(elapsed_sec)

    def record_insert(self, elapsed_sec: float) -> None:
        with self._lock:
            self.insert_durations.append(elapsed_sec)

    def profile(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                self.record(func.__name__, elapsed)
                logger.info("[PROFILE] %s completed in %.3fs", func.__name__, elapsed)

        return wrapper

    def log_summary(self, total_wall_sec: float) -> None:
        logger.info("=" * 72)
        logger.info("FUNCTION SPEED PROFILE SUMMARY")
        logger.info(
            "%-38s %6s %10s %10s %10s %10s",
            "Function",
            "Calls",
            "Total(s)",
            "Avg(s)",
            "Min(s)",
            "Max(s)",
        )
        logger.info("-" * 72)
        for func_name, durations in sorted(
            self.timings.items(), key=lambda item: sum(item[1]), reverse=True
        ):
            count = len(durations)
            total_t = sum(durations)
            avg_t = total_t / count if count else 0.0
            min_t = min(durations) if count else 0.0
            max_t = max(durations) if count else 0.0
            logger.info(
                "%-38s %6d %10.3f %10.3f %10.3f %10.3f",
                func_name,
                count,
                total_t,
                avg_t,
                min_t,
                max_t,
            )

        logger.info("-" * 72)
        total_kb = self.total_bytes_processed / 1024.0
        total_mb = total_kb / 1024.0
        embed_time = sum(self.embed_durations)
        insert_time = sum(self.insert_durations)
        wall_chunks_per_s = (
            self.total_chunks_embedded / total_wall_sec if total_wall_sec > 0 else 0.0
        )
        wall_kbps = total_kb / total_wall_sec if total_wall_sec > 0 else 0.0
        logger.info(
            "INDEXING SPEED SUMMARY: files=%d | chunks=%d | size=%.2f KB (%.2f MB) | "
            "cum_embed_time=%.3fs | cum_insert_time=%.3fs | total_wall_time=%.3fs (%.1f chunks/s, %.2f KB/s)",
            len(self.embed_durations),
            self.total_chunks_embedded,
            total_kb,
            total_mb,
            embed_time,
            insert_time,
            total_wall_sec,
            wall_chunks_per_s,
            wall_kbps,
        )
        logger.info("=" * 72)


profiler = FunctionProfiler()


@profiler.profile
def read_markdown_from_disk(file_path: Path) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


@profiler.profile
def chunk_text(text: str, max_chars: int = 1500, overlap: int = 200) -> list[str]:
    """Splits text into overlapping character chunks (~350-400 tokens per 1500 chars)."""
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + max_chars
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += max_chars - overlap
    return chunks


def parse_filename_metadata(file_path: Path) -> dict:
    """Extracts year, quarter, and ticker from filenames like '2019_3_AAPL.md'."""
    parts = file_path.stem.split("_")
    if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
        return {
            "year": int(parts[0]),
            "quarter": int(parts[1]),
            "ticker": parts[2],
        }
    return {"year": 0, "quarter": 0, "ticker": "UNKNOWN"}


def make_chunk_id(file_name: str, chunk_idx: int) -> int:
    """Generates a deterministic positive 63-bit int64 ID for (file_name, chunk_idx)."""
    digest = hashlib.blake2b(
        f"{file_name}:{chunk_idx}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") & 0x7FFFFFFFFFFFFFFF


@profiler.profile
def fetch_embedding_batch(
    oai_client: OpenAI, batch: list[str], model: str
) -> list[list[float]]:
    resp = oai_client.embeddings.create(
        model=model,
        input=batch,
    )
    return [item.embedding for item in resp.data]


@profiler.profile
def get_embeddings(
    oai_client: OpenAI,
    texts: list[str],
    model: str = "bge-base-en-v1.5",
    batch_size: int = 16,
) -> list[list[float]]:
    """Fetches embeddings in batches from the OpenAI-compatible embedding server."""
    t0 = time.perf_counter()
    all_embeddings: list[list[float]] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for batch_idx, i in enumerate(range(0, len(texts), batch_size), start=1):
        batch = texts[i : i + batch_size]
        bt0 = time.perf_counter()
        batch_vecs = fetch_embedding_batch(oai_client, batch, model)
        b_elapsed = time.perf_counter() - bt0
        all_embeddings.extend(batch_vecs)
        logger.debug(
            "Embedded batch %d/%d (%d chunks) in %.3fs",
            batch_idx,
            total_batches,
            len(batch),
            b_elapsed,
        )

    elapsed = time.perf_counter() - t0
    byte_size = sum(len(t.encode("utf-8")) for t in texts)
    profiler.record_embedding(len(texts), byte_size, elapsed)
    chunks_per_sec = len(texts) / elapsed if elapsed > 0 else 0.0
    speed_kbps = (byte_size / 1024.0) / elapsed if elapsed > 0 else 0.0
    logger.info(
        "[SPEED] Embedded %d chunks (%d bytes / %.2f KB) in %.3fs (%.1f chunks/s, %.2f KB/s)",
        len(texts),
        byte_size,
        byte_size / 1024.0,
        elapsed,
        chunks_per_sec,
        speed_kbps,
    )
    return all_embeddings


@profiler.profile
def write_rows_to_milvus(
    mclient: pymilvus.MilvusClient, collection_name: str, rows: list[dict]
) -> None:
    t0 = time.perf_counter()
    with milvus_lock:
        mclient.upsert(collection_name=collection_name, data=rows)
    elapsed = time.perf_counter() - t0
    profiler.record_insert(elapsed)


@profiler.profile
def insert_to_milvus(
    file_path: Path,
    mclient: pymilvus.MilvusClient,
    oai_client: OpenAI,
    collection_name: str,
    model_name: str,
    batch_size: int,
) -> int:
    t_start = time.perf_counter()
    logger.info("Starting indexing for %s", file_path)

    try:
        txt = read_markdown_from_disk(file_path)
    except OSError as exc:
        logger.error("Failed to read file %s: %s", file_path, exc)
        return 0

    if not txt.strip():
        logger.warning("File %s is empty, skipping", file_path)
        return 0

    meta = parse_filename_metadata(file_path)
    chunks = chunk_text(txt)
    if not chunks:
        logger.warning("No valid chunks extracted from %s, skipping", file_path)
        return 0

    logger.info(
        "Split %s into %d chunks (ticker=%s, year=%d, quarter=%d); requesting embeddings...",
        file_path.name,
        len(chunks),
        meta["ticker"],
        meta["year"],
        meta["quarter"],
    )

    t_embed = time.perf_counter()
    vectors = get_embeddings(
        oai_client=oai_client,
        texts=chunks,
        model=model_name,
        batch_size=batch_size,
    )
    embed_sec = time.perf_counter() - t_embed

    rows = [
        {
            "id": make_chunk_id(file_path.name, idx),
            "vector": vec,
            "text": chunk,
            "year": meta["year"],
            "quarter": meta["quarter"],
            "ticker": meta["ticker"],
            "source_file": file_path.name,
            "chunk_id": idx,
        }
        for idx, (chunk, vec) in enumerate(zip(chunks, vectors))
    ]

    t_insert = time.perf_counter()
    write_rows_to_milvus(mclient, collection_name, rows)
    insert_sec = time.perf_counter() - t_insert

    total_sec = time.perf_counter() - t_start
    logger.info(
        "Finished indexing %s: %d chunks | embed=%.3fs (%.1f chunks/s) | milvus_upsert=%.3fs | total=%.3fs",
        file_path.name,
        len(rows),
        embed_sec,
        len(rows) / embed_sec if embed_sec > 0 else 0.0,
        insert_sec,
        total_sec,
    )
    return len(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index SEC markdown filings into Milvus")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/fillings",
        help="Directory containing markdown filings to index",
    )
    parser.add_argument(
        "--db_path",
        type=str,
        default="vector/10qvector.db",
        help="Path to local Milvus Lite database file",
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        default=None,
        help="Milvus collection name (defaults to '10q_<dimension>')",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=None,
        help="Embedding vector dimension (auto-detected from server if omitted)",
    )
    parser.add_argument(
        "--embedding_url",
        type=str,
        default="http://10.0.0.242:2100/v1",
        help="Base URL for OpenAI-compatible embedding server",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="bge-base-en-v1.5",
        help="Embedding model name",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Number of chunks per embedding API request",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of worker threads",
    )
    parser.add_argument(
        "--cprofile",
        action="store_true",
        help="Enable detailed Python cProfile function-level profiling",
    )
    parser.add_argument(
        "--cprofile_top",
        type=int,
        default=25,
        help="Number of top functions to display when --cprofile is enabled",
    )

    args = parser.parse_args()

    dir_path = Path(args.data_dir)
    if not dir_path.exists():
        logger.error("Data directory does not exist: %s", dir_path)
        raise SystemExit(1)

    all_mds = sorted(dir_path.rglob("*.md"))
    if not all_mds:
        logger.warning("No .md files found in %s", dir_path)
        raise SystemExit(0)

    logger.info(
        "Found %d markdown file(s) in %s | embedding_url=%s | workers=%d | batch_size=%d",
        len(all_mds),
        dir_path,
        args.embedding_url,
        args.workers,
        args.batch_size,
    )

    cprof: Optional[cProfile.Profile] = None
    if args.cprofile:
        cprof = cProfile.Profile()
        cprof.enable()

    oai_client = OpenAI(
        base_url=args.embedding_url,
        api_key=os.environ.get("OPENAI_API_KEY", "not-needed"),
    )

    dimension = args.dimension
    if dimension is None:
        probe_resp = oai_client.embeddings.create(
            model=args.model,
            input=["dimension probe"],
        )
        dimension = len(probe_resp.data[0].embedding)
        logger.info("Auto-detected embedding dimension=%d from %s", dimension, args.embedding_url)

    collection_name = args.collection_name or f"10q_{dimension}"

    db_path = Path(args.db_path)
    os.makedirs(db_path.parent, exist_ok=True)
    if db_path.is_file() and db_path.stat().st_size == 0:
        logger.warning(
            "Removing empty placeholder file %s so Milvus Lite can initialize its storage",
            db_path,
        )
        db_path.unlink()

    mclient = pymilvus.MilvusClient(str(db_path))

    if not mclient.has_collection(collection_name):
        logger.info(
            "Creating Milvus collection '%s' (dimension=%d, auto_id=False) at %s",
            collection_name,
            dimension,
            args.db_path,
        )
        mclient.create_collection(
            collection_name=collection_name,
            dimension=dimension,
            auto_id=False,
        )
    else:
        logger.info(
            "Using existing Milvus collection '%s' at %s",
            collection_name,
            args.db_path,
        )

    wall_start = time.perf_counter()
    total_chunks = 0
    succeeded_files = 0
    failed_files = 0

    with ThreadPoolExecutor(max_workers=args.workers) as tp:
        future_to_path = {
            tp.submit(
                insert_to_milvus,
                md_path,
                mclient,
                oai_client,
                collection_name,
                args.model,
                args.batch_size,
            ): md_path
            for md_path in all_mds
        }

        for future in as_completed(future_to_path):
            md_path = future_to_path[future]
            try:
                chunks_indexed = future.result()
                total_chunks += chunks_indexed
                succeeded_files += 1
            except Exception as exc:
                failed_files += 1
                logger.exception("Failed indexing %s: %s", md_path, exc)

    wall_elapsed = time.perf_counter() - wall_start

    if cprof is not None:
        cprof.disable()
        s = io.StringIO()
        stats = pstats.Stats(cprof, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
        stats.print_stats(args.cprofile_top)
        logger.info(
            "cProfile Top %d Functions (by cumulative time):\n%s",
            args.cprofile_top,
            s.getvalue(),
        )

    logger.info(
        "Indexing complete: files=%d/%d succeeded (%d failed) | total_chunks=%d | wall_time=%.3fs (%.1f chunks/s)",
        succeeded_files,
        len(all_mds),
        failed_files,
        total_chunks,
        wall_elapsed,
        total_chunks / wall_elapsed if wall_elapsed > 0 else 0.0,
    )
    profiler.log_summary(wall_elapsed)


