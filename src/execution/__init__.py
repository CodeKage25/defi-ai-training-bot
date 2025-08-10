"""
Execution Engine - Advanced trade execution and order management
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from decimal import Decimal
import asyncio
from enum import Enum
import numpy as np
import json
import time
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from loguru import logger
from web3 import Web3
from web3.contract import Contract
from web3.exceptions import TransactionNotFound, BlockNotFound
import aiohttp


class OrderType(Enum):
    """Order types supported by the execution engine"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TWAP = "twap"  # Time-weighted average price
    ICEBERG = "iceberg"
    POST_ONLY = "post_only"


class OrderStatus(Enum):
    """Order execution status"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL_FILLED = "partial_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    FAILED = "failed"


class ExecutionStrategy(Enum):
    """Execution strategies"""
    AGGRESSIVE = "aggressive"  # Fast execution, higher slippage
    PASSIVE = "passive"       # Slow execution, lower slippage  
    BALANCED = "balanced"     # Balance between speed and slippage
    STEALTH = "stealth"       # Hidden execution, minimal market impact
    MEV_PROTECTED = "mev_protected"  # MEV-resistant execution


@dataclass
class OrderRequest:
    """Order execution request"""
    order_id: str
    token_in: str
    token_out: str
    amount_in: float
    order_type: OrderType
    chain: str
    
    # Optional parameters
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    slippage_tolerance: float = 0.01
    gas_limit: Optional[int] = None
    gas_price: Optional[float] = None
    deadline: Optional[int] = None
    
    # Execution strategy
    execution_strategy: ExecutionStrategy = ExecutionStrategy.BALANCED
    
    # Advanced options
    mev_protection: bool = True
    time_in_force: str = "GTC"  # Good Till Cancelled
    iceberg_size: Optional[float] = None
    twap_duration: Optional[int] = None
    
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None


@dataclass
class Fill:
    """Individual fill/execution"""
    fill_id: str
    order_id: str
    dex: str
    amount_in: float
    amount_out: float
    price: float
    gas_used: int
    gas_price: float
    transaction_hash: str
    block_number: int
    timestamp: datetime
    fees: Dict[str, float] = field(default_factory=dict)


@dataclass
class OrderResult:
    """Order execution result"""
    order_id: str
    status: OrderStatus
    fills: List[Fill] = field(default_factory=list)
    
    # Execution summary
    total_amount_in: float = 0.0
    total_amount_out: float = 0.0
    average_price: float = 0.0
    total_gas_cost: float = 0.0
    actual_slippage: float = 0.0
    
    # Timing
    submitted_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_time: float = 0.0
    
    # Results
    success: bool = False
    error_message: Optional[str] = None
    
    def add_fill(self, fill: Fill):
        """Add a fill to the order"""
        self.fills.append(fill)
        self.total_amount_in += fill.amount_in
        self.total_amount_out += fill.amount_out
        self.total_gas_cost += fill.gas_used * fill.gas_price / 1e18  # Convert to ETH
        
        # Recalculate average price
        if self.total_amount_in > 0:
            self.average_price = self.total_amount_out / self.total_amount_in


class DEXInterface(ABC):
    """Abstract interface for DEX interactions"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    async def get_quote(self, token_in: str, token_out: str, amount_in: float, chain: str) -> Dict[str, Any]:
        """Get price quote"""
        pass
    
    @abstractmethod
    async def execute_trade(self, order: OrderRequest) -> Fill:
        """Execute trade on this DEX"""
        pass
    
    @abstractmethod
    async def get_liquidity(self, token_in: str, token_out: str, chain: str) -> Dict[str, Any]:
        """Get liquidity information"""
        pass


class UniswapV3Interface(DEXInterface):
    """Uniswap V3 DEX interface"""
    
    def __init__(self, web3_provider: Web3):
        self.w3 = web3_provider
        self.name = "uniswap_v3"
        
        # Contract addresses (mainnet)
        self.router_address = "0xE592427A0AEce92De3Edee1F18E0157C05861564"
        self.quoter_address = "0xb27308f9F90D607463bb33eA1BeBb41C27CE5AB6"
        
    @property
    def name(self) -> str:
        return "uniswap_v3"
    
    async def get_quote(self, token_in: str, token_out: str, amount_in: float, chain: str) -> Dict[str, Any]:
        """Get Uniswap V3 quote"""
        try:
            # Mock quote for now - in production, call actual Uniswap quoter
            base_rate = 2500 if token_in == "ETH" and token_out == "USDC" else 1/2500
            amount_out = amount_in * base_rate * np.random.uniform(0.995, 1.005)  # Small variance
            
            return {
                "dex": self.name,
                "amount_in": amount_in,
                "amount_out": amount_out,
                "price": amount_out / amount_in if amount_in > 0 else 0,
                "gas_estimate": 150000,
                "fee_tier": 3000,  # 0.3%
                "price_impact": abs(np.random.uniform(-0.002, 0.002)),
                "liquidity": np.random.uniform(1000000, 5000000)
            }
        except Exception as e:
            logger.error(f"Error getting Uniswap quote: {e}")
            return {"error": str(e)}
    
    async def execute_trade(self, order: OrderRequest) -> Fill:
        """Execute trade on Uniswap V3"""
        try:
            # Get fresh quote
            quote = await self.get_quote(order.token_in, order.token_out, order.amount_in, order.chain)
            
            if "error" in quote:
                raise Exception(quote["error"])
            
            # Simulate transaction execution
            await asyncio.sleep(np.random.uniform(1, 3))  # Simulate network delay
            
            # Mock successful execution
            fill = Fill(
                fill_id=f"fill_{int(time.time() * 1000)}",
                order_id=order.order_id,
                dex=self.name,
                amount_in=order.amount_in,
                amount_out=quote["amount_out"] * (1 - order.slippage_tolerance/2),  # Account for slippage
                price=quote["price"],
                gas_used=quote["gas_estimate"],
                gas_price=np.random.uniform(20, 50) * 1e9,  # Mock gas price in wei
                transaction_hash=f"0x{''.join([f'{np.random.randint(0, 15):x}' for _ in range(64)])}",
                block_number=np.random.randint(18000000, 19000000),
                timestamp=datetime.now(),
                fees={"protocol_fee": order.amount_in * 0.003}  # 0.3% fee
            )
            
            return fill
            
        except Exception as e:
            logger.error(f"Error executing Uniswap trade: {e}")
            raise e
    
    async def get_liquidity(self, token_in: str, token_out: str, chain: str) -> Dict[str, Any]:
        """Get liquidity information for token pair"""
        return {
            "dex": self.name,
            "token_pair": f"{token_in}/{token_out}",
            "total_liquidity": np.random.uniform(5000000, 20000000),
            "available_liquidity": np.random.uniform(1000000, 5000000),
            "fee_tiers": [500, 3000, 10000],  # 0.05%, 0.3%, 1%
            "active_liquidity": np.random.uniform(500000, 2000000)
        }


class SushiswapInterface(DEXInterface):
    """Sushiswap DEX interface"""
    
    def __init__(self, web3_provider: Web3):
        self.w3 = web3_provider
        self._name = "sushiswap"
        self.router_address = "0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F"
        
    @property
    def name(self) -> str:
        return self._name
    
    async def get_quote(self, token_in: str, token_out: str, amount_in: float, chain: str) -> Dict[str, Any]:
        """Get Sushiswap quote"""
        try:
            # Mock quote with different rates than Uniswap
            base_rate = 2485 if token_in == "ETH" and token_out == "USDC" else 1/2485
            amount_out = amount_in * base_rate * np.random.uniform(0.992, 1.008)  # Slightly different variance
            
            return {
                "dex": self.name,
                "amount_in": amount_in,
                "amount_out": amount_out,
                "price": amount_out / amount_in if amount_in > 0 else 0,
                "gas_estimate": 180000,
                "fee": 0.003,  # 0.3%
                "price_impact": abs(np.random.uniform(-0.003, 0.003)),
                "liquidity": np.random.uniform(500000, 2000000)
            }
        except Exception as e:
            logger.error(f"Error getting Sushiswap quote: {e}")
            return {"error": str(e)}
    
    async def execute_trade(self, order: OrderRequest) -> Fill:
        """Execute trade on Sushiswap"""
        try:
            quote = await self.get_quote(order.token_in, order.token_out, order.amount_in, order.chain)
            
            if "error" in quote:
                raise Exception(quote["error"])
            
            await asyncio.sleep(np.random.uniform(0.8, 2.5))
            
            fill = Fill(
                fill_id=f"fill_{int(time.time() * 1000)}",
                order_id=order.order_id,
                dex=self.name,
                amount_in=order.amount_in,
                amount_out=quote["amount_out"] * (1 - order.slippage_tolerance/2),
                price=quote["price"],
                gas_used=quote["gas_estimate"],
                gas_price=np.random.uniform(18, 45) * 1e9,
                transaction_hash=f"0x{''.join([f'{np.random.randint(0, 15):x}' for _ in range(64)])}",
                block_number=np.random.randint(18000000, 19000000),
                timestamp=datetime.now(),
                fees={"protocol_fee": order.amount_in * 0.003}
            )
            
            return fill
            
        except Exception as e:
            logger.error(f"Error executing Sushiswap trade: {e}")
            raise e
    
    async def get_liquidity(self, token_in: str, token_out: str, chain: str) -> Dict[str, Any]:
        """Get Sushiswap liquidity information"""
        return {
            "dex": self.name,
            "token_pair": f"{token_in}/{token_out}",
            "total_liquidity": np.random.uniform(2000000, 8000000),
            "available_liquidity": np.random.uniform(500000, 2000000),
            "fee": 0.003,
            "volume_24h": np.random.uniform(1000000, 5000000)
        }


class CurveInterface(DEXInterface):
    """Curve Finance DEX interface (for stablecoin trading)"""
    
    def __init__(self, web3_provider: Web3):
        self.w3 = web3_provider
        self._name = "curve"
        
    @property
    def name(self) -> str:
        return self._name
    
    async def get_quote(self, token_in: str, token_out: str, amount_in: float, chain: str) -> Dict[str, Any]:
        """Get Curve quote (optimized for stablecoins)"""
        try:
            # Curve is optimized for stablecoin swaps
            is_stablecoin_pair = all(token in ["USDC", "USDT", "DAI", "FRAX"] for token in [token_in, token_out])
            
            if is_stablecoin_pair:
                # Very low slippage for stablecoin pairs
                amount_out = amount_in * np.random.uniform(0.9998, 1.0002)
                price_impact = abs(np.random.uniform(-0.0005, 0.0005))
            else:
                # Higher slippage for non-stablecoin pairs
                base_rate = 2490 if token_in == "ETH" and token_out == "USDC" else 1/2490
                amount_out = amount_in * base_rate * np.random.uniform(0.996, 1.004)
                price_impact = abs(np.random.uniform(-0.002, 0.002))
            
            return {
                "dex": self.name,
                "amount_in": amount_in,
                "amount_out": amount_out,
                "price": amount_out / amount_in if amount_in > 0 else 0,
                "gas_estimate": 120000,
                "fee": 0.0004,  # 0.04% for stablecoin pools
                "price_impact": price_impact,
                "liquidity": np.random.uniform(10000000, 50000000) if is_stablecoin_pair else np.random.uniform(1000000, 5000000)
            }
        except Exception as e:
            logger.error(f"Error getting Curve quote: {e}")
            return {"error": str(e)}
    
    async def execute_trade(self, order: OrderRequest) -> Fill:
        """Execute trade on Curve"""
        try:
            quote = await self.get_quote(order.token_in, order.token_out, order.amount_in, order.chain)
            
            if "error" in quote:
                raise Exception(quote["error"])
            
            await asyncio.sleep(np.random.uniform(0.5, 2.0))
            
            fill = Fill(
                fill_id=f"fill_{int(time.time() * 1000)}",
                order_id=order.order_id,
                dex=self.name,
                amount_in=order.amount_in,
                amount_out=quote["amount_out"] * (1 - order.slippage_tolerance/3),  # Lower slippage impact
                price=quote["price"],
                gas_used=quote["gas_estimate"],
                gas_price=np.random.uniform(15, 40) * 1e9,
                transaction_hash=f"0x{''.join([f'{np.random.randint(0, 15):x}' for _ in range(64)])}",
                block_number=np.random.randint(18000000, 19000000),
                timestamp=datetime.now(),
                fees={"protocol_fee": order.amount_in * quote["fee"]}
            )
            
            return fill
            
        except Exception as e:
            logger.error(f"Error executing Curve trade: {e}")
            raise e
    
    async def get_liquidity(self, token_in: str, token_out: str, chain: str) -> Dict[str, Any]:
        """Get Curve liquidity information"""
        is_stablecoin_pair = all(token in ["USDC", "USDT", "DAI", "FRAX"] for token in [token_in, token_out])
        
        return {
            "dex": self.name,
            "token_pair": f"{token_in}/{token_out}",
            "total_liquidity": np.random.uniform(50000000, 200000000) if is_stablecoin_pair else np.random.uniform(5000000, 20000000),
            "available_liquidity": np.random.uniform(20000000, 80000000) if is_stablecoin_pair else np.random.uniform(2000000, 8000000),
            "fee": 0.0004 if is_stablecoin_pair else 0.003,
            "a_parameter": np.random.randint(100, 2000),  # Curve's A parameter
            "virtual_price": np.random.uniform(1.0, 1.1)
        }


class DEXAggregator:
    """DEX aggregator for optimal trade routing"""
    
    def __init__(self, web3_providers: Dict[str, Web3]):
        self.web3_providers = web3_providers
        self.dex_interfaces = {}
        self._initialize_dex_interfaces()
    
    def _initialize_dex_interfaces(self):
        """Initialize DEX interfaces"""
        for chain, w3 in self.web3_providers.items():
            if chain == "ethereum":
                self.dex_interfaces[f"uniswap_v3_{chain}"] = UniswapV3Interface(w3)
                self.dex_interfaces[f"sushiswap_{chain}"] = SushiswapInterface(w3)
                self.dex_interfaces[f"curve_{chain}"] = CurveInterface(w3)
        
        logger.info(f"Initialized {len(self.dex_interfaces)} DEX interfaces")
    
    async def get_best_route(self, order: OrderRequest) -> Tuple[str, Dict[str, Any]]:
        """Find the best route for an order"""
        best_quote = None
        best_dex = None
        
        # Get quotes from all available DEXs for the chain
        chain_dexs = [name for name in self.dex_interfaces.keys() if order.chain in name]
        
        quotes = {}
        for dex_name in chain_dexs:
            try:
                dex = self.dex_interfaces[dex_name]
                quote = await dex.get_quote(order.token_in, order.token_out, order.amount_in, order.chain)
                
                if "error" not in quote:
                    quotes[dex_name] = quote
                    
                    # Score the quote based on strategy
                    score = self._score_quote(quote, order.execution_strategy)
                    
                    if best_quote is None or score > best_quote.get("score", 0):
                        best_quote = quote.copy()
                        best_quote["score"] = score
                        best_dex = dex_name
                        
            except Exception as e:
                logger.error(f"Error getting quote from {dex_name}: {e}")
                continue
        
        if best_dex is None:
            raise Exception("No viable DEX found for trade")
        
        return best_dex, best_quote
    
    def _score_quote(self, quote: Dict[str, Any], strategy: ExecutionStrategy) -> float:
        """Score a quote based on execution strategy"""
        amount_out = quote.get("amount_out", 0)
        gas_estimate = quote.get("gas_estimate", 0)
        price_impact = quote.get("price_impact", 0)
        liquidity = quote.get("liquidity", 0)
        
        if strategy == ExecutionStrategy.AGGRESSIVE:
            # Prioritize speed and amount out
            score = amount_out * 0.7 - gas_estimate * 0.0001 - price_impact * 1000
        elif strategy == ExecutionStrategy.PASSIVE:
            # Prioritize low price impact and gas costs
            score = amount_out * 0.4 - gas_estimate * 0.0002 - price_impact * 2000 + liquidity * 0.00001
        elif strategy == ExecutionStrategy.STEALTH:
            # Prioritize liquidity and low market impact
            score = liquidity * 0.0001 - price_impact * 3000 + amount_out * 0.3
        elif strategy == ExecutionStrategy.MEV_PROTECTED:
            # Balance all factors with slight preference for established DEXs
            score = amount_out * 0.5 - gas_estimate * 0.0001 - price_impact * 1500 + liquidity * 0.00005
        else:  # BALANCED
            score = amount_out * 0.6 - gas_estimate * 0.0001 - price_impact * 1200 + liquidity * 0.00002
        
        return score
    
    async def execute_split_order(self, order: OrderRequest, splits: Dict[str, float]) -> List[Fill]:
        """Execute order across multiple DEXs"""
        fills = []
        
        for dex_name, split_ratio in splits.items():
            if split_ratio <= 0:
                continue
                
            try:
                # Create sub-order for this split
                split_order = OrderRequest(
                    order_id=f"{order.order_id}_{dex_name}",
                    token_in=order.token_in,
                    token_out=order.token_out,
                    amount_in=order.amount_in * split_ratio,
                    order_type=order.order_type,
                    chain=order.chain,
                    slippage_tolerance=order.slippage_tolerance,
                    execution_strategy=order.execution_strategy
                )
                
                dex = self.dex_interfaces[dex_name]
                fill = await dex.execute_trade(split_order)
                fills.append(fill)
                
            except Exception as e:
                logger.error(f"Error executing split order on {dex_name}: {e}")
                continue
        
        return fills


class GasOptimizer:
    """Gas price optimization and transaction timing"""
    
    def __init__(self, web3_providers: Dict[str, Web3]):
        self.web3_providers = web3_providers
        self.gas_history = {}
        
    async def get_optimal_gas_price(self, chain: str, priority: str = "standard") -> Dict[str, Any]:
        """Get optimal gas price for transaction"""
        w3 = self.web3_providers.get(chain)
        
        if not w3:
            # Return mock gas prices
            mock_prices = {
                "ethereum": {"slow": 15, "standard": 25, "fast": 40, "urgent": 60},
                "polygon": {"slow": 1, "standard": 2, "fast": 5, "urgent": 10},
                "bsc": {"slow": 3, "standard": 5, "fast": 8, "urgent": 15},
                "arbitrum": {"slow": 0.1, "standard": 0.2, "fast": 0.5, "urgent": 1.0}
            }
            
            chain_prices = mock_prices.get(chain, mock_prices["ethereum"])
            return {
                "recommended_gas_price": chain_prices[priority],
                "estimated_cost_usd": chain_prices[priority] * 150000 / 1e9 * 2000,  # Mock calculation
                "confidence": 0.8,
                "wait_time_minutes": {"slow": 10, "standard": 3, "fast": 1, "urgent": 0.5}[priority]
            }
        
        try:
            # Get current gas price from network
            current_gas_price = w3.eth.gas_price / 1e9  # Convert to gwei
            
            # Adjust based on priority
            multipliers = {"slow": 0.8, "standard": 1.0, "fast": 1.3, "urgent": 1.8}
            recommended_gas = current_gas_price * multipliers.get(priority, 1.0)
            
            return {
                "recommended_gas_price": recommended_gas,
                "current_gas_price": current_gas_price,
                "estimated_cost_usd": recommended_gas * 150000 / 1e9 * 2000,
                "confidence": 0.9,
                "wait_time_minutes": {"slow": 10, "standard": 3, "fast": 1, "urgent": 0.5}[priority]
            }
            
        except Exception as e:
            logger.error(f"Error getting gas price for {chain}: {e}")
            return self.get_optimal_gas_price(chain, priority)  # Fallback to mock
    
    async def estimate_gas_limit(self, order: OrderRequest, dex_name: str) -> int:
        """Estimate gas limit for order"""
        # Base gas estimates by DEX and operation type
        base_estimates = {
            "uniswap_v3": 150000,
            "sushiswap": 180000,
            "curve": 120000,
            "1inch": 200000
        }
        
        base_gas = base_estimates.get(dex_name.split("_")[0], 150000)
        
        # Adjust based on order type
        if order.order_type == OrderType.LIMIT:
            base_gas += 50000  # Additional gas for limit order logic
        elif order.order_type == OrderType.TWAP:
            base_gas += 100000  # More complex execution logic
        
        # Add buffer
        return int(base_gas * 1.2)
    
    async def should_delay_transaction(self, chain: str, gas_price_limit: Optional[float] = None) -> Dict[str, Any]:
        """Determine if transaction should be delayed for better gas prices"""
        gas_info = await self.get_optimal_gas_price(chain, "standard")
        current_gas = gas_info["recommended_gas_price"]
        
        if gas_price_limit and current_gas > gas_price_limit:
            return {
                "should_delay": True,
                "current_gas_price": current_gas,
                "target_gas_price": gas_price_limit,
                "estimated_wait_time": 30 + np.random.uniform(0, 60),  # 30-90 minutes
                "potential_savings": (current_gas - gas_price_limit) * 150000 / 1e9 * 2000
            }
        
        return {
            "should_delay": False,
            "current_gas_price": current_gas,
            "recommendation": "Execute now"
        }


class MEVProtection:
    """MEV (Maximum Extractable Value) protection mechanisms"""
    
    def __init__(self):
        self.private_pools = ["flashbots", "eden", "mistx"]
        self.protection_strategies = ["private_mempool", "commit_reveal", "time_delay", "sandwich_protection"]
    
    async def get_mev_protection_strategy(self, order: OrderRequest) -> Dict[str, Any]:
        """Determine optimal MEV protection strategy"""
        order_value = order.amount_in * 2500  # Mock USD value
        
        if order_value > 50000:  # Large orders need strong protection
            return {
                "strategy": "private_mempool",
                "private_pool": "flashbots",
                "additional_gas_cost": 0.002,  # 0.2% of order value
                "protection_level": "high",
                "expected_savings": order_value * 0.005  # 0.5% protection from MEV
            }
        elif order_value > 10000:
            return {
                "strategy": "sandwich_protection",
                "slippage_buffer": 0.002,  # Additional slippage buffer
                "protection_level": "medium",
                "expected_savings": order_value * 0.002
            }
        else:
            return {
                "strategy": "standard_execution",
                "protection_level": "low",
                "recommendation": "Standard execution sufficient for small orders"
            }
    
    async def submit_to_private_pool(self, transaction_data: Dict[str, Any], pool: str = "flashbots") -> Dict[str, Any]:
        """Submit transaction to private mempool"""
        # Mock private pool submission
        await asyncio.sleep(0.5)
        
        return {
            "pool": pool,
            "submission_id": f"pvt_{int(time.time() * 1000)}",
            "status": "submitted",
            "estimated_inclusion_time": 15,  # seconds
            "mev_protection_active": True
        }


class OrderManager:
    """Advanced order management and execution orchestration"""
    
    def __init__(self, web3_providers: Dict[str, Web3]):
        self.web3_providers = web3_providers
        self.dex_aggregator = DEXAggregator(web3_providers)
        self.gas_optimizer = GasOptimizer(web3_providers)
        self.mev_protection = MEVProtection()
        
        # Order tracking
        self.active_orders: Dict[str, OrderResult] = {}
        self.order_history: List[OrderResult] = []
        
        # Performance metrics
        self.execution_stats = {
            "total_orders": 0,
            "successful_orders": 0,
            "failed_orders": 0,
            "average_execution_time": 0.0,
            "total_gas_savings": 0.0,
            "total_slippage_savings": 0.0
        }
    
    async def submit_order(self, order: OrderRequest) -> str:
        """Submit order for execution"""
        logger.info(f"Submitting order {order.order_id}: {order.amount_in} {order.token_in} → {order.token_out}")
        
        # Initialize order result
        order_result = OrderResult(
            order_id=order.order_id,
            status=OrderStatus.SUBMITTED,
            submitted_at=datetime.now()
        )
        
        self.active_orders[order.order_id] = order_result
        
        # Execute order based on type
        try:
            if order.order_type == OrderType.MARKET:
                await self._execute_market_order(order)
            elif order.order_type == OrderType.LIMIT:
                await self._execute_limit_order(order)
            elif order.order_type == OrderType.TWAP:
                await self._execute_twap_order(order)
            elif order.order_type == OrderType.ICEBERG:
                await self._execute_iceberg_order(order)
            else:
                raise Exception(f"Unsupported order type: {order.order_type}")
                
        except Exception as e:
            logger.error(f"Error executing order {order.order_id}: {e}")
            order_result.status = OrderStatus.FAILED
            order_result.error_message = str(e)
            order_result.completed_at = datetime.now()
        
        return order.order_id
    
    async def _execute_market_order(self, order: OrderRequest):
        """Execute market order with optimal routing"""
        order_result = self.active_orders[order.order_id]
        
        try:
            # Get optimal route
            best_dex, quote = await self.dex_aggregator.get_best_route(order)
            logger.info(f"Best route for {order.order_id}: {best_dex} with output {quote['amount_out']}")
            
            # Check if we should split the order
            if order.amount_in * 2500 > 25000:  # Split large orders ($25k+)
                fills = await self._execute_split_order(order, quote)
            else:
                # Single DEX execution
                dex = self.dex_aggregator.dex_interfaces[best_dex]
                fill = await dex.execute_trade(order)
                fills = [fill]
            
            # Update order result
            for fill in fills:
                order_result.add_fill(fill)
            
            if fills:
                order_result.status = OrderStatus.FILLED
                order_result.success = True
                order_result.completed_at = datetime.now()
                order_result.execution_time = (order_result.completed_at - order_result.submitted_at).total_seconds()
                
                # Calculate actual slippage
                expected_price = quote.get("price", 0)
                if expected_price > 0:
                    order_result.actual_slippage = abs(order_result.average_price - expected_price) / expected_price
                
                logger.info(f"Order {order.order_id} completed successfully")
            else:
                raise Exception("No fills received")
                
        except Exception as e:
            logger.error(f"Market order execution failed: {e}")
            order_result.status = OrderStatus.FAILED
            order_result.error_message = str(e)
            raise e
    
    async def _execute_split_order(self, order: OrderRequest, quote: Dict[str, Any]) -> List[Fill]:
        """Execute order split across multiple DEXs"""
        # Simple split strategy: 60% best DEX, 40% second best
        best_dex, _ = await self.dex_aggregator.get_best_route(order)
        
        splits = {best_dex: 0.6}
        
        # Find second best DEX
        chain_dexs = [name for name in self.dex_aggregator.dex_interfaces.keys() if order.chain in name and name != best_dex]
        if chain_dexs:
            # Get quotes for remaining DEXs
            second_best = None
            second_best_score = -1
            
            for dex_name in chain_dexs[:2]:  # Check top 2 alternatives
                try:
                    dex = self.dex_aggregator.dex_interfaces[dex_name]
                    alt_quote = await dex.get_quote(order.token_in, order.token_out, order.amount_in * 0.4, order.chain)
                    
                    if "error" not in alt_quote:
                        score = self.dex_aggregator._score_quote(alt_quote, order.execution_strategy)
                        if score > second_best_score:
                            second_best = dex_name
                            second_best_score = score
                except:
                    continue
            
            if second_best:
                splits[second_best] = 0.4
            else:
                splits[best_dex] = 1.0  # Use single DEX if no good alternative
        
        return await self.dex_aggregator.execute_split_order(order, splits)
    
    async def _execute_limit_order(self, order: OrderRequest):
        """Execute limit order (simplified implementation)"""
        order_result = self.active_orders[order.order_id]
        
        if not order.limit_price:
            raise Exception("Limit price required for limit orders")
        
        # Monitor market price and execute when conditions are met
        max_wait_time = 300  # 5 minutes for demo
        start_time = time.time()
        
        order_result.status = OrderStatus.PENDING
        
        while time.time() - start_time < max_wait_time:
            try:
                # Get current market price
                best_dex, quote = await self.dex_aggregator.get_best_route(order)
                current_price = quote.get("price", 0)
                
                # Check if limit conditions are met
                if ((order.token_in == "ETH" and current_price >= order.limit_price) or
                    (order.token_out == "ETH" and current_price <= order.limit_price)):
                    
                    # Execute at market
                    await self._execute_market_order(order)
                    return
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error monitoring limit order: {e}")
                await asyncio.sleep(5)
        
        # Order expired
        order_result.status = OrderStatus.EXPIRED
        order_result.completed_at = datetime.now()
        logger.info(f"Limit order {order.order_id} expired")
    
    async def _execute_twap_order(self, order: OrderRequest):
        """Execute Time-Weighted Average Price order"""
        order_result = self.active_orders[order.order_id]
        
        if not order.twap_duration:
            order.twap_duration = 600  # Default 10 minutes
        
        # Split order into smaller chunks over time
        num_chunks = min(10, order.twap_duration // 60)  # 1 chunk per minute, max 10
        chunk_size = order.amount_in / num_chunks
        chunk_interval = order.twap_duration / num_chunks
        
        logger.info(f"Executing TWAP order: {num_chunks} chunks of {chunk_size} over {order.twap_duration}s")
        
        order_result.status = OrderStatus.EXECUTING
        
        for i in range(num_chunks):
            try:
                # Create chunk order
                chunk_order = OrderRequest(
                    order_id=f"{order.order_id}_chunk_{i}",
                    token_in=order.token_in,
                    token_out=order.token_out,
                    amount_in=chunk_size,
                    order_type=OrderType.MARKET,
                    chain=order.chain,
                    slippage_tolerance=order.slippage_tolerance,
                    execution_strategy=ExecutionStrategy.PASSIVE  # Use passive for TWAP
                )
                
                # Execute chunk
                best_dex, _ = await self.dex_aggregator.get_best_route(chunk_order)
                dex = self.dex_aggregator.dex_interfaces[best_dex]
                fill = await dex.execute_trade(chunk_order)
                
                order_result.add_fill(fill)
                
                logger.info(f"TWAP chunk {i+1}/{num_chunks} executed: {fill.amount_out} {order.token_out}")
                
                # Wait before next chunk (except for last chunk)
                if i < num_chunks - 1:
                    await asyncio.sleep(chunk_interval)
                    
            except Exception as e:
                logger.error(f"Error executing TWAP chunk {i}: {e}")
                continue
        
        if order_result.fills:
            order_result.status = OrderStatus.FILLED
            order_result.success = True
        else:
            order_result.status = OrderStatus.FAILED
            order_result.error_message = "No successful fills"
        
        order_result.completed_at = datetime.now()
        order_result.execution_time = (order_result.completed_at - order_result.submitted_at).total_seconds()
    
    async def _execute_iceberg_order(self, order: OrderRequest):
        """Execute iceberg order (large order split into smaller visible chunks)"""
        order_result = self.active_orders[order.order_id]
        
        if not order.iceberg_size:
            order.iceberg_size = order.amount_in * 0.1  
        
        remaining_amount = order.amount_in
        
        order_result.status = OrderStatus.EXECUTING