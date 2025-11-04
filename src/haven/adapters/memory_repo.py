from typing import List, Dict, Any
from haven.domain.ports import DealRepository

class InMemoryDealRepository(DealRepository):
    def __init__(self):
        self._items: List[Dict[str, Any]] = []

    def save_analysis(self, analysis: Dict[str, Any]) -> None:
        self._items.append(analysis)

    def all(self) -> List[Dict[str, Any]]:
        return list(self._items)
