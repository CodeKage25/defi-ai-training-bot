from typing import Optional
from .thegraph import graph_query
from ..models import PoolLiquidity

UNI_V3 = "https://api.thegraph.com/subgraphs/defi-ai-trading-bot/uniswap/uniswap-v3"

POOL_QUERY = """
{
  pools(first: 1, where:{id:"%s"}) {
    id totalValueLockedUSD volumeUSD
  }
}
"""

async def pool_liquidity(chain: str, pool_id: str) -> Optional[PoolLiquidity]:
    data = await graph_query(UNI_V3, POOL_QUERY % pool_id)
    pools = data.get("data", {}).get("pools", [])
    if not pools: return None
    p = pools[0]
    return PoolLiquidity(
        chain=chain, dex="uniswap_v3", pair=pool_id,
        total_liquidity_usd=float(p["totalValueLockedUSD"]),
        volume_24h_usd=float(p["volumeUSD"])
    )
