from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional, Any


def batch_in_thread_pool(func: Callable[[Any], Optional[list]], items: list) -> list:
    if not items:
        return []
    max_workers = min(100, len(items))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(func, item) for item in items]
        results: list = []
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)
        return results
