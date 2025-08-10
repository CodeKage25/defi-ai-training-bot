from typing import Dict, Any, Optional
from .base import DataSource
from ..http_client import get_json
from ..models import PriceQuote

class CoinGecko(DataSource):
    name = "coingecko"
    BASE = "https://api.coingecko.com/api/v3"

    async def health(self) -> Dict[str, Any]:
        try:
            await get_json(f"{self.BASE}/ping")
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def simple_price(self, symbol: str, vs: str = "usd") -> Optional[PriceQuote]:
        data = await get_json(f"{self.BASE}/simple/price", params={"ids": symbol.lower(), "vs_currencies": vs})
        price = data.get(symbol.lower(), {}).get(vs)
        if price is None:
            return None
        return PriceQuote(source=self.name, base=symbol.upper(), quote=vs.upper(), price=float(price))
