from abc import ABC, abstractmethod
from typing import Dict, Any

class DataSource(ABC):
    name: str

    @abstractmethod
    async def health(self) -> Dict[str, Any]:
        ...

