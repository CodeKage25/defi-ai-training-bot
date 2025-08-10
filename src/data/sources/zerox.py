from typing import Dict, Any, Optional
from .base import DataSource
from ..http_client import get_json
from ..models import PriceQuote

class ZeroX(DataSource):
    name = "0x"
    BASES = {
        "ethereum": "https://api.0x.org/",
        "polygon": "https://polygon.api.0x.org/",
        "bsc": "https://bsc.api.0x.org/",
    }

    async def health(self) -> Dict[str, Any]:
        return {"ok": True}

    async def quote(self, chain: str, sell_token: str, buy_token: str, sell_amount_wei: int) -> Optional[PriceQuote]:
        base = self.BASES.get(chain)
        if not base:
            return None
        data = await get_json(
            f"{base}swap/v1/quote",
            params={"sellToken": sell_token, "buyToken": buy_token, "sellAmount": sell_amount_wei},
        )
        price = float(data["price"])
        liq = float(data.get("estimatedPriceImpact", 0))
        # Note: here price is buy/sell; normalize to quote per base if needed
        return PriceQuote(source=self.name, base=sell_token, quote=buy_token, price=price, liquidity_usd=None, fee_bps=None)
