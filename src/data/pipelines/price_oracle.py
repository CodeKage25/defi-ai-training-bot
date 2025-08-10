from typing import Optional, List
from ..models import PriceQuote
from ..registry import registry

async def best_price(symbol: str, vs: str="usd") -> Optional[PriceQuote]:
    cg = await registry.coingecko.simple_price(symbol, vs)
    return cg
