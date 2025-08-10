from typing import Dict, Any
from ..http_client import get_json

async def graph_query(url: str, query: str, variables: Dict[str, Any] | None = None):
    return await get_json(url, params={"query": query, "variables": variables})
