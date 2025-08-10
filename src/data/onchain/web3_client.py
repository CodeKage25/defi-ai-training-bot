from web3 import Web3
from functools import lru_cache
import os

@lru_cache(maxsize=16)
def get_w3(chain: str) -> Web3:
    url = {
        "ethereum": os.getenv("WEB3_PROVIDER_ETHEREUM", ""),
        "polygon": os.getenv("WEB3_PROVIDER_POLYGON", ""),
        "bsc": os.getenv("WEB3_PROVIDER_BSC", "")
    }.get(chain)
    if not url:
        raise RuntimeError(f"No RPC for chain={chain}")
    w3 = Web3(Web3.HTTPProvider(url, request_kwargs={"timeout": 15}))
    return w3
