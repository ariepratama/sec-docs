import argparse
import cProfile
from collections import defaultdict
from functools import wraps
import io
import logging
import os
import pstats
import time
from typing import Any, Callable, Dict, List, Optional
from edgar import Company, set_identity
from edgar.exceptions import CompanyNotFoundError, EdgarError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


class FunctionProfiler:
    """Tracks execution time of each profiled function and overall download throughput."""

    def __init__(self) -> None:
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.total_bytes_downloaded: int = 0
        self.download_durations: List[float] = []

    def record(self, func_name: str, elapsed_sec: float) -> None:
        self.timings[func_name].append(elapsed_sec)

    def record_download(self, num_bytes: int, elapsed_sec: float) -> None:
        self.total_bytes_downloaded += num_bytes
        self.download_durations.append(elapsed_sec)

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
        total_kb = self.total_bytes_downloaded / 1024.0
        total_mb = total_kb / 1024.0
        dl_time = sum(self.download_durations)
        effective_kbps = total_kb / dl_time if dl_time > 0 else 0.0
        overall_kbps = total_kb / total_wall_sec if total_wall_sec > 0 else 0.0
        logger.info(
            "DOWNLOAD SPEED SUMMARY: files=%d | size=%.2f KB (%.2f MB) | "
            "download+parse_time=%.3fs (%.2f KB/s) | total_wall_time=%.3fs (%.2f KB/s)",
            len(self.download_durations),
            total_kb,
            total_mb,
            dl_time,
            effective_kbps,
            total_wall_sec,
            overall_kbps,
        )
        logger.info("=" * 72)


profiler = FunctionProfiler()


@profiler.profile
def init_company(company: str) -> Company:
    return Company(company)


@profiler.profile
def fetch_company_filings(company_obj: Company, form: str, year: int, quarter: int):
    return company_obj.get_filings(form=form, year=year, quarter=quarter)


@profiler.profile
def download_and_convert_to_markdown(filing) -> str:
    t0 = time.perf_counter()
    md = filing.markdown()
    elapsed = time.perf_counter() - t0
    if md:
        byte_size = len(md.encode("utf-8"))
        profiler.record_download(byte_size, elapsed)
        speed_kbps = (byte_size / 1024.0) / elapsed if elapsed > 0 else 0.0
        logger.info(
            "[SPEED] Downloaded & converted filing (%d bytes / %.2f KB) in %.3fs (%.2f KB/s)",
            byte_size,
            byte_size / 1024.0,
            elapsed,
            speed_kbps,
        )
    return md


@profiler.profile
def write_markdown_to_disk(file_path: str, filing_md: str) -> int:
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(filing_md)
    return len(filing_md.encode("utf-8"))


@profiler.profile
def download_quarter_filing_data_markdown(
    company: str, form: str, year: int, quarter: int, output_dir: str
) -> bool:
    logger.info(
        "Fetching filings for company=%s form=%s year=%d quarter=%d",
        company,
        form,
        year,
        quarter,
    )
    try:
        company_obj = init_company(company)
        filings = fetch_company_filings(company_obj, form=form, year=year, quarter=quarter)
    except CompanyNotFoundError:
        logger.error("Company not found: company=%s", company)
        return False
    except (EdgarError, Exception) as exc:
        logger.exception(
            "Failed to retrieve filings for company=%s year=%d quarter=%d form=%s: %s",
            company,
            year,
            quarter,
            form,
            exc,
        )
        return False

    if not filings or len(filings) < 1:
        logger.warning(
            "No filings found for company=%s year=%d quarter=%d form=%s, skipping",
            company,
            year,
            quarter,
            form,
        )
        return False

    filing = filings[0]
    logger.info(
        "Found %d filing(s) for company=%s; converting latest filing (accession=%s, filing_date=%s) to markdown",
        len(filings),
        company,
        getattr(filing, "accession_no", "unknown"),
        getattr(filing, "filing_date", "unknown"),
    )

    try:
        filing_md = download_and_convert_to_markdown(filing)
    except Exception as exc:
        logger.exception(
            "Failed to convert filing to markdown for company=%s year=%d quarter=%d form=%s: %s",
            company,
            year,
            quarter,
            form,
            exc,
        )
        return False

    if not filing_md:
        logger.warning(
            "Generated markdown is empty for company=%s year=%d quarter=%d form=%s, skipping write",
            company,
            year,
            quarter,
            form,
        )
        return False

    os.makedirs(output_dir, exist_ok=True)
    file_name = f"{year}_{quarter}_{company}.md"
    file_path = os.path.join(output_dir, file_name)

    try:
        bytes_written = write_markdown_to_disk(file_path, filing_md)
        logger.info(
            "Successfully wrote filing (%d chars, %.2f KB) to %s (company=%s year=%d quarter=%d form=%s)",
            len(filing_md),
            bytes_written / 1024.0,
            file_path,
            company,
            year,
            quarter,
            form,
        )
        return True
    except OSError as exc:
        logger.exception("Failed to write filing to %s: %s", file_path, exc)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Download filing data")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/fillings",
        help="Directory to save the downloaded filing data",
    )
    parser.add_argument(
        "--company_list",
        type=str,
        required=True,
        help="Comma-separated list of company tickers to download filing data for",
    )
    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="Year for which to download filing data",
    )
    parser.add_argument(
        "--quarter",
        type=int,
        required=True,
        choices=[1, 2, 3, 4],
        help="Quarter (1-4) for which to download filing data",
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

    set_identity("example@example.com")

    companies = [c.strip().upper() for c in args.company_list.split(",") if c.strip()]
    if not companies:
        logger.error("No valid company tickers provided in --company_list")
        raise SystemExit(1)

    logger.info(
        "Starting download for %d company(s): %s (year=%d, quarter=%d, output_dir=%s)",
        len(companies),
        ", ".join(companies),
        args.year,
        args.quarter,
        args.output_dir,
    )

    cprof: Optional[cProfile.Profile] = None
    if args.cprofile:
        cprof = cProfile.Profile()
        cprof.enable()

    wall_start = time.perf_counter()
    success_count = 0
    for company in companies:
        if download_quarter_filing_data_markdown(
            company=company,
            form="10-Q",
            year=args.year,
            quarter=args.quarter,
            output_dir=args.output_dir,
        ):
            success_count += 1
    total_wall_sec = time.perf_counter() - wall_start

    if cprof is not None:
        cprof.disable()
        s = io.StringIO()
        stats = pstats.Stats(cprof, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
        stats.print_stats(args.cprofile_top)
        logger.info("cProfile Top %d Functions (by cumulative time):\n%s", args.cprofile_top, s.getvalue())

    logger.info(
        "Completed filing download: %d/%d succeeded in %.3fs",
        success_count,
        len(companies),
        total_wall_sec,
    )
    profiler.log_summary(total_wall_sec)


if __name__ == "__main__":
    main()



