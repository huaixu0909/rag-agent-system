import hmac
import os
import time
from collections import defaultdict, deque
from collections.abc import Callable

from fastapi import Header, HTTPException, Request


ADMIN_HEADER_NAME = "X-Admin-API-Key"
_RATE_LIMIT_BUCKETS: dict[str, deque[float]] = defaultdict(deque)


def require_admin(x_admin_api_key: str | None = Header(default=None, alias=ADMIN_HEADER_NAME)) -> None:
    expected_key = os.getenv("ADMIN_API_KEY", "").strip()
    if not expected_key:
        raise HTTPException(
            status_code=503,
            detail="ADMIN_API_KEY is not configured. Dangerous write operations are disabled.",
        )
    if not x_admin_api_key or not hmac.compare_digest(x_admin_api_key, expected_key):
        raise HTTPException(status_code=401, detail="Admin API key is required.")


def rate_limit(bucket: str, limit: int, window_seconds: int) -> Callable[[Request], None]:
    def dependency(request: Request) -> None:
        client_host = request.client.host if request.client else "unknown"
        key = f"{bucket}:{client_host}"
        now = time.monotonic()
        timestamps = _RATE_LIMIT_BUCKETS[key]

        while timestamps and now - timestamps[0] > window_seconds:
            timestamps.popleft()

        if len(timestamps) >= limit:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Try again in {window_seconds} seconds.",
            )

        timestamps.append(now)

    return dependency
