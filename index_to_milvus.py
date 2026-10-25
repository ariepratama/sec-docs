import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import os
from pathlib import Path
import threading
import time
from openai import OpenAI
import pymilvus

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Lock for thread-safe writes to local Milvus Lite database file
milvus_lock = threading.Lock()


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


def get_embeddings(
    oai_client: OpenAI,
    texts: list[str],
    model: str = "bge-base-en-v1.5",
    batch_size: int = 16,
) -> list[list[float]]:
    """Fetches embeddings in batches from the OpenAI-compatible embedding server."""
    all_embeddings: list[list[float]] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for batch_idx, i in enumerate(range(0, len(texts), batch_size), start=1):
        batch = texts[i : i + batch_size]
        t0 = time.perf_counter()
        resp = oai_client.embeddings.create(
            model=model,
            input=batch,
        )
        elapsed = time.perf_counter() - t0
        batch_vecs = [item.embedding for item in resp.data]
        all_embeddings.extend(batch_vecs)
        logger.debug(
            "Embedded batch %d/%d (%d chunks) in %.3fs",
            batch_idx,
            total_batches,
            len(batch),
            elapsed,
        )

    return all_embeddings


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
        with open(file_path, "r", encoding="utf-8") as f:
            txt = f.read()
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
    with milvus_lock:
        mclient.insert(collection_name=collection_name, data=rows)
    insert_sec = time.perf_counter() - t_insert

    total_sec = time.perf_counter() - t_start
    logger.info(
        "Finished indexing %s: %d chunks | embed=%.3fs (%.1f chunks/s) | milvus_insert=%.3fs | total=%.3fs",
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
        default="10q_768",
        help="Milvus collection name",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=768,
        help="Embedding vector dimension (768 for BAAI/bge-base-en-v1.5)",
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

    os.makedirs(Path(args.db_path).parent, exist_ok=True)
    mclient = pymilvus.MilvusClient(args.db_path)

    if not mclient.has_collection(args.collection_name):
        logger.info(
            "Creating Milvus collection '%s' (dimension=%d, auto_id=True) at %s",
            args.collection_name,
            args.dimension,
            args.db_path,
        )
        mclient.create_collection(
            collection_name=args.collection_name,
            dimension=args.dimension,
            auto_id=True,
        )
    else:
        logger.info(
            "Using existing Milvus collection '%s' at %s",
            args.collection_name,
            args.db_path,
        )

    oai_client = OpenAI(
        base_url=args.embedding_url,
        api_key=os.environ.get("OPENAI_API_KEY", "not-needed"),
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
                args.collection_name,
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
    logger.info(
        "Indexing complete: files=%d/%d succeeded (%d failed) | total_chunks=%d | wall_time=%.3fs (%.1f chunks/s)",
        succeeded_files,
        len(all_mds),
        failed_files,
        total_chunks,
        wall_elapsed,
        total_chunks / wall_elapsed if wall_elapsed > 0 else 0.0,
    )

