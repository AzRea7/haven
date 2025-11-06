from typing import Any

from haven.domain.ports import DealRepository


class InMemoryDealRepository(DealRepository):
    def __init__(self) -> None:
        self._items: list[dict[str, Any]] = []

    def save_analysis(self, analysis: dict[str, Any]) -> None:
        self._items.append(analysis)

    def all(self) -> list[dict[str, Any]]:
        return list(self._items)
