from ..models import GasInfo
from datetime import datetime
import random


async def gas_prices(chain: str) -> GasInfo:
    base = {"ethereum": 25, "polygon": 2, "bsc": 5}.get(chain, 25)
    return GasInfo(chain=chain, slow=base*0.8, standard=base, fast=base*1.3, urgent=base*1.8, ts=datetime.utcnow())
