from typing import List
from ..models import PoolLiquidity
from ..sources.uniswap_subgraph import pool_liquidity

async def uni_snapshot(chain: str, pool_ids: List[str]) -> List[PoolLiquidity]:
    out: List[PoolLiquidity] = []
    for pid in pool_ids:
        liq = await pool_liquidity(chain, pid)
        if liq: out.append(liq)
    return out
