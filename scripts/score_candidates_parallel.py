import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from haven.adapters.sql_repo import SqlPropertyRepository
from haven.adapters.config import config
# from haven.services.scoring import score_property  # adapt to your real func


def dummy_score_property(p) -> float:
    # Replace with real scoring logic
    # e.g., use trained model + DSCR/CoC/etc.
    return float(p.list_price or 0) / 1_000_000.0


def main():
    repo = SqlPropertyRepository(uri=config.DB_URI)
    props = repo.search(limit=5000)  # or filter by zip

    print(f"Scoring {len(props)} properties in parallel...")

    t0 = time.perf_counter()
    results = []

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(dummy_score_property, p): p.id for p in props}
        for fut in as_completed(futures):
            score = fut.result()
            pid = futures[fut]
            results.append((pid, score))

    dt = time.perf_counter() - t0

    print(f"Scored {len(results)} properties in {dt:.2f}s")
    if dt > 0:
        print(f"Throughput: {len(results)/dt:.1f} props/s")

    # You can persist scores back via repo if needed:
    # repo.update_scores(results)


if __name__ == "__main__":
    main()
