from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime

class PriceQuote(BaseModel):
    source: str
    base: str
    quote: str = "USDC"
    price: float
    liquidity_usd: Optional[float] = None
    fee_bps: Optional[int] = None
    ts: datetime = Field(default_factory=datetime.utcnow)

class OHLCV(BaseModel):
    symbol: str
    open: float; high: float; low: float; close: float; volume: float
    ts: datetime

class GasInfo(BaseModel):
    chain: str
    slow: float; standard: float; fast: float; urgent: float  # gwei
    ts: datetime = Field(default_factory=datetime.utcnow)

class PoolLiquidity(BaseModel):
    chain: str
    dex: str
    pair: str
    total_liquidity_usd: float
    depth_1pct_usd: Optional[float] = None
    volume_24h_usd: Optional[float] = None
    ts: datetime = Field(default_factory=datetime.utcnow)

class TokenInfo(BaseModel):
    chain: str; symbol: str; address: str; decimals: int
