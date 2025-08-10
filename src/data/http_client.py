import aiohttp, asyncio
from loguru import logger
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

DEFAULT_TIMEOUT = aiohttp.ClientTimeout(total=15)

@asynccontextmanager
async def http_session():
    async with aiohttp.ClientSession(timeout=DEFAULT_TIMEOUT) as s:
        yield s

async def get_json(url: str, params: Optional[Dict[str, Any]]=None, headers: Optional[Dict[str,str]]=None):
    async with http_session() as s:
        async with s.get(url, params=params, headers=headers) as r:
            r.raise_for_status()
            return await r.json()
