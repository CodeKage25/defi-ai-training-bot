"""
Market Intelligence Agent - Multi-chain opportunity scanning and analysis
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import asyncio
from decimal import Decimal

from spoon_ai.agents import ToolCallAgent
from spoon_ai.tools.base import BaseTool
from spoon_ai.tools import ToolManager
from pydantic import Field
from loguru import logger


class ChainbaseAnalyticsTool(BaseTool):
    """On-chain analytics and transaction analysis using Chainbase"""
    
    name: str = "chainbase_analytics"
    description: str = "Analyze on-chain data, token flows, and whale movements"
    parameters: dict = {
        "type": "object",
        "properties": {
            "chain": {
                "type": "string",
                "description": "Blockchain to analyze (ethereum, polygon, bsc, arbitrum)",
                "enum": ["ethereum", "polygon", "bsc", "arbitrum"]
            },
            "token_address": {
                "type": "string",
                "description": "Token contract address to analyze"
            },
            "analysis_type": {
                "type": "string",
                "description": "Type of analysis to perform",
                "enum": ["whale_movements", "token_flows", "liquidity_analysis", "holder_distribution"]
            },
            "timeframe": {
                "type": "string",
                "description": "Analysis timeframe",
                "enum": ["1h", "24h", "7d", "30d"],
                "default": "24h"
            }
        },
        "required": ["chain", "analysis_type"]
    }

    async def execute(self, chain: str, analysis_type: str, token_address: str = None, timeframe: str = "24h") -> Dict[str, Any]:
        """Execute on-chain analytics"""
        logger.info(f"Analyzing {chain} - {analysis_type} for timeframe {timeframe}")
        
        # Simulate Chainbase API call
        await asyncio.sleep(1)  # Simulate API latency
        
        if analysis_type == "whale_movements":
            return {
                "chain": chain,
                "analysis": "whale_movements",
                "timeframe": timeframe,
                "large_transactions": [
                    {
                        "hash": "0x1234...",
                        "value_usd": 1_500_000,
                        "from": "0xwhale1...",
                        "to": "0xexchange...",
                        "token": token_address or "ETH",
                        "timestamp": datetime.now().isoformat()
                    },
                    {
                        "hash": "0x5678...", 
                        "value_usd": 800_000,
                        "from": "0xwhale2...",
                        "to": "0xdefi_protocol...",
                        "token": token_address or "USDC",
                        "timestamp": (datetime.now() - timedelta(hours=2)).isoformat()
                    }
                ],
                "summary": {
                    "total_whale_volume": 12_300_000,
                    "net_flow": "outbound",
                    "sentiment": "bearish"
                }
            }
        
        elif analysis_type == "liquidity_analysis":
            return {
                "chain": chain,
                "analysis": "liquidity_analysis", 
                "token": token_address,
                "pools": [
                    {
                        "dex": "uniswap_v3",
                        "liquidity_usd": 15_600_000,
                        "volume_24h": 2_800_000,
                        "fees_24h": 8_400,
                        "price_impact_1_percent": 0.02
                    },
                    {
                        "dex": "sushiswap",
                        "liquidity_usd": 8_900_000,
                        "volume_24h": 1_200_000,
                        "fees_24h": 3_600,
                        "price_impact_1_percent": 0.04
                    }
                ],
                "total_liquidity": 24_500_000,
                "liquidity_trend": "increasing",
                "risk_score": 0.3
            }
        
        return {"analysis": analysis_type, "status": "completed", "data": {}}


class PriceAggregatorTool(BaseTool):
    """Multi-source price aggregation and arbitrage detection"""
    
    name: str = "price_aggregator"
    description: str = "Aggregate prices from multiple sources and detect arbitrage opportunities"
    parameters: dict = {
        "type": "object",
        "properties": {
            "token_symbol": {
                "type": "string",
                "description": "Token symbol to analyze (e.g., ETH, BTC, USDC)"
            },
            "vs_currency": {
                "type": "string", 
                "description": "Quote currency",
                "default": "USD"
            },
            "sources": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Price sources to compare",
                "default": ["coingecko", "coinmarketcap", "dex_screener"]
            }
        },
        "required": ["token_symbol"]
    }

    async def execute(self, token_symbol: str, vs_currency: str = "USD", sources: List[str] = None) -> Dict[str, Any]:
        """Get aggregated prices and detect arbitrage opportunities"""
        if sources is None:
            sources = ["coingecko", "coinmarketcap", "dex_screener"]
            
        logger.info(f"Aggregating prices for {token_symbol}/{vs_currency} from {sources}")
        
        # Simulate price fetching
        await asyncio.sleep(0.5)
        
        # Mock price data with slight variations for arbitrage detection
        base_price = 2500.0 if token_symbol == "ETH" else 50000.0 if token_symbol == "BTC" else 1.0
        
        price_data = {
            "token": token_symbol,
            "vs_currency": vs_currency,
            "timestamp": datetime.now().isoformat(),
            "prices": {
                "coingecko": base_price * 0.999,
                "coinmarketcap": base_price * 1.001,
                "dex_screener": base_price * 0.997
            },
            "average_price": base_price,
            "price_spread": {
                "min": base_price * 0.997,
                "max": base_price * 1.001,
                "spread_percentage": 0.4
            },
            "arbitrage_opportunities": [
                {
                    "buy_exchange": "dex_screener",
                    "sell_exchange": "coinmarketcap", 
                    "buy_price": base_price * 0.997,
                    "sell_price": base_price * 1.001,
                    "profit_percentage": 0.4,
                    "estimated_profit_usd": base_price * 0.004
                }
            ] if base_price * 0.004 > 10 else []  # Only show if profit > $10
        }
        
        return price_data


class DEXMonitorTool(BaseTool):
    """DEX liquidity and arbitrage opportunity monitoring"""
    
    name: str = "dex_monitor"
    description: str = "Monitor DEX liquidity, volumes, and cross-DEX arbitrage opportunities"
    parameters: dict = {
        "type": "object",
        "properties": {
            "chain": {
                "type": "string",
                "description": "Blockchain to monitor",
                "enum": ["ethereum", "polygon", "bsc", "arbitrum"]
            },
            "token_pair": {
                "type": "string",
                "description": "Token pair to monitor (e.g., ETH/USDC, BTC/USDT)"
            },
            "dexs": {
                "type": "array",
                "items": {"type": "string"},
                "description": "DEXs to monitor",
                "default": ["uniswap", "sushiswap", "curve"]
            },
            "min_liquidity": {
                "type": "number",
                "description": "Minimum liquidity threshold in USD",
                "default": 100000
            }
        },
        "required": ["chain", "token_pair"]
    }

    async def execute(self, chain: str, token_pair: str, dexs: List[str] = None, min_liquidity: float = 100000) -> Dict[str, Any]:
        """Monitor DEX data for arbitrage opportunities"""
        if dexs is None:
            dexs = ["uniswap", "sushiswap", "curve"]
            
        logger.info(f"Monitoring {token_pair} on {chain} across DEXs: {dexs}")
        
        await asyncio.sleep(1)
        
        # Mock DEX data
        dex_data = []
        base_price = 2500.0 if "ETH" in token_pair else 1.0
        
        for i, dex in enumerate(dexs):
            price_variation = 1 + (i * 0.002 - 0.002)  # Small price differences
            dex_data.append({
                "dex": dex,
                "chain": chain,
                "pair": token_pair,
                "price": base_price * price_variation,
                "liquidity_usd": min_liquidity * (2 + i * 0.5),
                "volume_24h": min_liquidity * 0.8,
                "fees_apy": 0.05 + (i * 0.01),
                "last_update": datetime.now().isoformat()
            })
        
        # Find best arbitrage opportunity
        prices = [data["price"] for data in dex_data]
        min_price = min(prices)
        max_price = max(prices)
        arbitrage_profit = ((max_price - min_price) / min_price) * 100
        
        return {
            "chain": chain,
            "pair": token_pair,
            "timestamp": datetime.now().isoformat(),
            "dex_data": dex_data,
            "best_arbitrage": {
                "buy_dex": dex_data[prices.index(min_price)]["dex"],
                "sell_dex": dex_data[prices.index(max_price)]["dex"],
                "buy_price": min_price,
                "sell_price": max_price,
                "profit_percentage": arbitrage_profit,
                "recommended": arbitrage_profit > 0.1  # Recommend if >0.1% profit
            },
            "total_liquidity": sum(data["liquidity_usd"] for data in dex_data),
            "average_price": sum(prices) / len(prices)
        }


class MarketIntelligenceAgent(ToolCallAgent):
    """AI agent for multi-chain market intelligence and opportunity scanning"""
    
    name: str = "market_intelligence_agent"
    description: str = """
    Advanced AI agent that monitors multi-chain DeFi markets for trading opportunities.
    Analyzes on-chain data, price movements, liquidity conditions, and whale activity
    to identify profitable trading opportunities with calculated risk assessments.
    """

    system_prompt: str = """You are an advanced DeFi market intelligence AI agent.

    Your responsibilities:
    1. Monitor multiple blockchain networks for trading opportunities
    2. Analyze on-chain data including whale movements and transaction flows  
    3. Detect arbitrage opportunities across different DEXs and chains
    4. Assess liquidity conditions and market risks
    5. Provide actionable trading recommendations with risk analysis

    Key capabilities:
    - Cross-chain opportunity scanning (Ethereum, Polygon, BSC, Arbitrum)
    - Real-time price aggregation and comparison
    - Whale movement tracking and sentiment analysis
    - DEX liquidity monitoring and arbitrage detection
    - Risk assessment and profit estimation

    Always provide:
    - Clear reasoning for each recommendation
    - Risk assessment and probability estimates
    - Specific entry/exit criteria
    - Estimated profit potential and time horizons
    - Gas cost considerations for transaction execution

    Be precise with numbers, conservative with risk estimates, and always prioritize
    capital preservation over aggressive profit-seeking.
    """

    next_step_prompt: str = """
    Based on the current market analysis, what is the next best action to take?
    Consider:
    1. Current opportunities and their risk/reward ratios
    2. Market conditions and volatility
    3. Available liquidity and execution feasibility
    4. Gas costs and optimal timing
    5. Portfolio diversification needs
    """

    max_steps: int = 15

    # Define available tools for market intelligence
    available_tools: ToolManager = Field(default_factory=lambda: ToolManager([
        ChainbaseAnalyticsTool(),
        PriceAggregatorTool(), 
        DEXMonitorTool(),
    ]))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.info(f"Initialized {self.name} with {len(self.available_tools.tools)} tools")

    async def scan_market_opportunities(self, chains: List[str] = None, tokens: List[str] = None) -> Dict[str, Any]:
        """
        Comprehensive market opportunity scan across multiple chains
        
        Args:
            chains: List of chains to scan (default: all supported)
            tokens: List of tokens to focus on (default: major tokens)
        """
        if chains is None:
            chains = ["ethereum", "polygon", "bsc", "arbitrum"]
        if tokens is None:
            tokens = ["ETH", "BTC", "USDC", "USDT"]
            
        logger.info(f"Starting comprehensive market scan for chains: {chains}, tokens: {tokens}")
        
        prompt = f"""
        Perform a comprehensive market intelligence scan with the following parameters:
        - Chains to analyze: {', '.join(chains)}
        - Focus tokens: {', '.join(tokens)}
        
        Please:
        1. Check whale movements on each chain for the past 24h
        2. Analyze current liquidity conditions for major token pairs
        3. Identify any arbitrage opportunities across DEXs
        4. Assess overall market sentiment and risk factors
        5. Provide top 3 trading recommendations with full analysis
        
        Be thorough but efficient in your analysis.
        """
        
        return await self.run(prompt)

    async def analyze_specific_opportunity(self, chain: str, token_pair: str, opportunity_type: str = "arbitrage") -> Dict[str, Any]:
        """
        Deep dive analysis of a specific trading opportunity
        
        Args:
            chain: Blockchain to analyze
            token_pair: Token pair (e.g., "ETH/USDC")
            opportunity_type: Type of opportunity (arbitrage, yield, trend)
        """
        logger.info(f"Analyzing {opportunity_type} opportunity for {token_pair} on {chain}")
        
        prompt = f"""
        Perform a detailed analysis of a specific trading opportunity:
        
        Chain: {chain}
        Token Pair: {token_pair}
        Opportunity Type: {opportunity_type}
        
        Please provide:
        1. Current market conditions and liquidity analysis
        2. Historical performance and patterns
        3. Risk assessment including smart contract risks
        4. Optimal entry and exit strategies
        5. Expected returns and time horizon
        6. Gas cost analysis and net profit calculation
        
        Give me a complete trading plan with specific parameters.
        """
        
        return await self.run(prompt)

    async def get_risk_assessment(self, strategy: str, amount_usd: float) -> Dict[str, Any]:
        """
        Get AI-powered risk assessment for a trading strategy
        
        Args:
            strategy: Description of the trading strategy
            amount_usd: Amount to risk in USD
        """
        logger.info(f"Assessing risk for strategy: {strategy}, amount: ${amount_usd}")
        
        prompt = f"""
        Provide a comprehensive risk assessment for the following trading strategy:
        
        Strategy: {strategy}
        Capital Amount: ${amount_usd:,.2f}
        
        Please analyze:
        1. Market risks (volatility, liquidity, slippage)
        2. Technical risks (smart contract, bridge, oracle risks)
        3. Operational risks (gas costs, MEV, timing)
        4. Maximum potential loss scenarios
        5. Risk mitigation strategies
        6. Position sizing recommendations
        7. Stop-loss and take-profit levels
        
        Provide a risk score from 1-10 and clear recommendations.
        """
        
        return await self.run(prompt)

    def clear_analysis_cache(self):
        """Clear the agent's memory/cache for fresh analysis"""
        self.clear()
        logger.info("Market intelligence cache cleared")