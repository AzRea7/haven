from typing import Any

from haven.domain.ports import DealRepository


class InMemoryDealRepository(DealRepository):
    def __init__(self) -> None:
        self._items: list[dict[str, Any]] = []

    def save_analysis(
        self,
        analysis: dict[str, Any],
        request_payload: dict[str, Any] | None = None,
    ) -> int | None:
        rec = analysis.copy()
        if request_payload is not None:
            rec["request_payload"] = request_payload
        self._items.append(rec)
        return None

    def all(self) -> list[dict[str, Any]]:
        return list(self._items)
