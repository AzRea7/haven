# src/haven/adapters/rentcast_client.py
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests


class RentCastError(RuntimeError):
    pass


def _env(key: str, default: str | None = None) -> str | None:
    v = os.getenv(key)
    return v if v is not None else default


@dataclass(frozen=True)
class RentCastClient:
    base_url: str
    api_key: str
    timeout_s: float = 20.0
    max_retries: int = 4
    backoff_base_s: float = 0.8

    def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = self.base_url.rstrip("/") + "/" + path.lstrip("/")

        headers = {
            "Accept": "application/json",
            "X-Api-Key": self.api_key,  # per RentCast docs
        }

        last_err: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.get(
                    url,
                    headers=headers,
                    params=params or {},
                    timeout=self.timeout_s,
                )

                # Handle rate limiting / transient gateway errors with retry
                if resp.status_code in (429, 502, 503, 504):
                    wait = self.backoff_base_s * (2**attempt)
                    # Respect Retry-After if present
                    ra = resp.headers.get("Retry-After")
                    if ra:
                        try:
                            wait = max(wait, float(ra))
                        except Exception:
                            pass
                    time.sleep(wait)
                    continue

                if resp.status_code >= 400:
                    raise RentCastError(
                        f"RentCast HTTP {resp.status_code}: {resp.text}"
                    )

                return resp.json()

            except Exception as e:
                last_err = e
                # network errors -> retry with backoff
                if attempt < self.max_retries:
                    time.sleep(self.backoff_base_s * (2**attempt))
                    continue
                break

        raise RentCastError(f"RentCast request failed after retries: {last_err!r}")


def make_rentcast_client() -> RentCastClient:
    api_key = _env("HAVEN_RENTCAST_API_KEY")
    if not api_key:
        raise RentCastError(
            "Missing HAVEN_RENTCAST_API_KEY. Set it in your environment before calling RentCast."
        )

    base_url = _env("HAVEN_RENTCAST_BASE_URL", "https://api.rentcast.io/v1")

    timeout_s = float(_env("HAVEN_RENTCAST_TIMEOUT_S", "20") or 20)
    max_retries = int(float(_env("HAVEN_RENTCAST_MAX_RETRIES", "4") or 4))
    backoff_base = float(_env("HAVEN_RENTCAST_BACKOFF_BASE_S", "0.8") or 0.8)

    return RentCastClient(
        base_url=base_url,
        api_key=api_key,
        timeout_s=timeout_s,
        max_retries=max_retries,
        backoff_base_s=backoff_base,
    )
