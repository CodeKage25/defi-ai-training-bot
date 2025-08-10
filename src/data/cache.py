import time
from typing import Any, Dict, Tuple, Optional

class MemoryCache:
    def __init__(self): self.store: Dict[str, Tuple[float, Any]] = {}
    def set(self, key: str, value: Any, ttl: int = 30):
        self.store[key] = (time.time() + ttl, value)
    def get(self, key: str) -> Optional[Any]:
        exp, val = self.store.get(key, (0, None))
        if exp and exp > time.time(): return val
        if key in self.store: del self.store[key]
        return None

cache = MemoryCache()
