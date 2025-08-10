"""
Execution Manager Agent - Handles trade execution and position management
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import asyncio
from decimal import Decimal
import json
from enum import Enum

from spoon_ai.agents import ToolCallAgent
from spoon_ai.tools.base import BaseTool
from spoon_ai.tools import ToolManager
from pydantic import Field
from loguru import logger
import numpy as np



class TradeStatus(Enum):
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DEXAggregatorTool(BaseTool):
    """Optimal trade execution across multiple DEXs"""
    
    name: str = "dex_aggregator"
    description: str = "Execute trades across multiple DEXs for best prices and minimal slippage"
    parameters: dict = {
        "type": "object",
        "properties": {
            "token_in": {
                "type": "string",
                "description": "Token to sell (address or symbol)"
            },
            "token_out": {
                "type": "string",
                "description": "Token to buy (address or symbol)"
            },
            "amount_in": {
                "type": "number",
                "description": "Amount to trade (in token_in units)"
            },
            "chain": {
                "type": "string",
                "description": "Blockchain network",
                "enum": ["ethereum", "polygon", "bsc", "arbitrum"]
            },
            "slippage_tolerance": {
                "type": "number",
                "description": "Maximum acceptable slippage percentage",
                "default": 0.01
            },
            "execution_strategy": {
                "type": "string",
                "description": "Execution strategy",
                "enum": ["best_price", "lowest_gas", "fastest", "mev_protected"],
                "default": "best_price"
            }
        },
        "required": ["token_in", "token_out", "amount_in", "chain"]
    }

    async def execute(self, token_in: str, token_out: str, amount_in: float, chain: str,
                     slippage_tolerance: float = 0.01, execution_strategy: str = "best_price") -> Dict[str, Any]:
        """Execute optimal trade across DEXs"""
        logger.info(f"Executing trade: {amount_in} {token_in} -> {token_out} on {chain}")
        
        # Simulate DEX aggregation and route finding
        await asyncio.sleep(2)
        
        # Mock DEX routes and pricing
        available_dexs = {
            "ethereum": ["uniswap_v3", "sushiswap", "curve", "1inch"],
            "polygon": ["quickswap", "sushiswap", "curve", "1inch"],
            "bsc": ["pancakeswap", "biswap", "1inch"],
            "arbitrum": ["uniswap_v3", "sushiswap", "curve", "1inch"]
        }
        
        dexs = available_dexs.get(chain, ["uniswap_v3"])
        
        # Generate mock routes
        routes = []
        base_output = amount_in * 2500 if token_out == "USDC" else amount_in / 2500  # Mock conversion
        
        for i, dex in enumerate(dexs):
            price_impact = 0.001 + (i * 0.0005)  # Varying price impact
            gas_cost = 50 + (i * 10)  # Varying gas costs
            
            expected_output = base_output * (1 - price_impact)
            routes.append({
                "dex": dex,
                "path": [token_in, token_out],
                "expected_output": expected_output,
                "price_impact": price_impact,
                "gas_estimate": gas_cost,
                "execution_time": 15 + (i * 5),  # Estimated execution time in seconds
                "liquidity": 1000000 + (i * 500000)
            })
        
        # Select best route based on strategy
        if execution_strategy == "best_price":
            best_route = max(routes, key=lambda x: x["expected_output"])
        elif execution_strategy == "lowest_gas":
            best_route = min(routes, key=lambda x: x["gas_estimate"])
        elif execution_strategy == "fastest":
            best_route = min(routes, key=lambda x: x["execution_time"])
        else:  # mev_protected
            best_route = max(routes, key=lambda x: x["expected_output"] - x["gas_estimate"] * 0.01)
        
        # Simulate transaction execution
        transaction_hash = f"0x{''.join([format(i, '02x') for i in range(32)])}"
        
        return {
            "trade_id": f"trade_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "status": "completed",
            "transaction_hash": transaction_hash,
            "chain": chain,
            "input": {
                "token": token_in,
                "amount": amount_in
            },
            "output": {
                "token": token_out,
                "amount": best_route["expected_output"],
                "actual_price": best_route["expected_output"] / amount_in
            },
            "execution_details": {
                "selected_route": best_route,
                "all_routes": routes,
                "slippage_tolerance": slippage_tolerance,
                "actual_slippage": best_route["price_impact"],
                "gas_used": best_route["gas_estimate"],
                "execution_time": best_route["execution_time"],
                "mev_protection": execution_strategy == "mev_protected"
            },
            "timestamp": datetime.now().isoformat(),
            "profit_loss": 0,  # Will be calculated after execution
            "fees": {
                "gas_fee_usd": best_route["gas_estimate"] * 0.002,  # Mock gas fee
                "dex_fee_usd": amount_in * 0.003,  # 0.3% DEX fee
                "total_fee_usd": best_route["gas_estimate"] * 0.002 + amount_in * 0.003
            }
        }


class PositionManagerTool(BaseTool):
    """Position tracking and automated management"""
    
    name: str = "position_manager"
    description: str = "Track and manage trading positions with automated stop-loss and take-profit"
    parameters: dict = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "Position management action",
                "enum": ["create", "update", "close", "list", "monitor"]
            },
            "position_id": {
                "type": "string",
                "description": "Position ID (required for update/close/monitor)"
            },
            "token": {
                "type": "string", 
                "description": "Token symbol (required for create)"
            },
            "amount": {
                "type": "number",
                "description": "Position size (required for create)"
            },
            "entry_price": {
                "type": "number",
                "description": "Entry price (required for create)"
            },
            "stop_loss": {
                "type": "number",
                "description": "Stop loss price/percentage"
            },
            "take_profit": {
                "type": "number",
                "description": "Take profit price/percentage"
            }
        },
        "required": ["action"]
    }

    def __init__(self):
        super().__init__()
        self.positions = {}  # In-memory position storage

    async def execute(self, action: str, position_id: str = None, token: str = None, 
                     amount: float = None, entry_price: float = None, 
                     stop_loss: float = None, take_profit: float = None) -> Dict[str, Any]:
        """Manage trading positions"""
        logger.info(f"Position management action: {action}")
        
        await asyncio.sleep(0.5)
        
        if action == "create":
            if not all([token, amount, entry_price]):
                return {"error": "Missing required fields for position creation"}
            
            position_id = f"pos_{token}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            current_price = entry_price * (1 + (np.random.uniform(-0.02, 0.02)))  # Mock price movement
            
            position = {
                "position_id": position_id,
                "token": token,
                "amount": amount,
                "entry_price": entry_price,
                "current_price": current_price,
                "stop_loss": stop_loss or entry_price * 0.95,  # Default 5% stop loss
                "take_profit": take_profit or entry_price * 1.15,  # Default 15% take profit
                "created_at": datetime.now().isoformat(),
                "status": "active",
                "pnl_usd": (current_price - entry_price) * amount,
                "pnl_percentage": ((current_price - entry_price) / entry_price) * 100
            }
            
            self.positions[position_id] = position
            
            return {
                "action": "create",
                "success": True,
                "position": position,
                "message": f"Position created for {amount} {token} at ${entry_price:.4f}"
            }
        
        elif action == "list":
            return {
                "action": "list",
                "total_positions": len(self.positions),
                "positions": list(self.positions.values()),
                "summary": {
                    "active_positions": len([p for p in self.positions.values() if p["status"] == "active"]),
                    "total_pnl": sum(p["pnl_usd"] for p in self.positions.values()),
                    "winning_positions": len([p for p in self.positions.values() if p["pnl_usd"] > 0])
                }
            }
        
        elif action == "monitor":
            if position_id and position_id in self.positions:
                position = self.positions[position_id]
                # Mock price update
                price_change = np.random.uniform(-0.01, 0.01)
                position["current_price"] *= (1 + price_change)
                position["pnl_usd"] = (position["current_price"] - position["entry_price"]) * position["amount"]
                position["pnl_percentage"] = ((position["current_price"] - position["entry_price"]) / position["entry_price"]) * 100
                
                # Check stop loss and take profit triggers
                alerts = []
                if position["current_price"] <= position["stop_loss"]:
                    alerts.append({
                        "type": "stop_loss_triggered",
                        "message": f"Stop loss triggered for {position['token']} at ${position['current_price']:.4f}",
                        "action_required": "close_position"
                    })
                elif position["current_price"] >= position["take_profit"]:
                    alerts.append({
                        "type": "take_profit_triggered", 
                        "message": f"Take profit triggered for {position['token']} at ${position['current_price']:.4f}",
                        "action_required": "close_position"
                    })
                
                return {
                    "action": "monitor",
                    "position": position,
                    "alerts": alerts,
                    "recommendations": self._get_position_recommendations(position)
                }
            else:
                return {"error": f"Position {position_id} not found"}
        
        elif action == "close":
            if position_id and position_id in self.positions:
                position = self.positions[position_id]
                position["status"] = "closed"
                position["closed_at"] = datetime.now().isoformat()
                
                return {
                    "action": "close",
                    "success": True,
                    "position": position,
                    "final_pnl": position["pnl_usd"],
                    "message": f"Position closed with PnL: ${position['pnl_usd']:.2f}"
                }
            else:
                return {"error": f"Position {position_id} not found"}
        
        return {"error": f"Unknown action: {action}"}
    
    def _get_position_recommendations(self, position: Dict) -> List[str]:
        """Generate position management recommendations"""
        recommendations = []
        
        if position["pnl_percentage"] > 10:
            recommendations.append("Consider taking partial profits")
        elif position["pnl_percentage"] < -3:
            recommendations.append("Monitor closely, consider cutting losses")
        
        if abs(position["pnl_percentage"]) > 5:
            recommendations.append("Consider adjusting stop loss to lock in gains/limit losses")
        
        return recommendations


class GasOptimizerTool(BaseTool):
    """Gas price optimization and transaction timing"""
    
    name: str = "gas_optimizer"
    description: str = "Optimize gas prices and transaction timing for cost-effective execution"
    parameters: dict = {
        "type": "object",
        "properties": {
            "chain": {
                "type": "string",
                "description": "Blockchain network",
                "enum": ["ethereum", "polygon", "bsc", "arbitrum"]
            },
            "priority": {
                "type": "string",
                "description": "Transaction priority",
                "enum": ["low", "standard", "high", "urgent"],
                "default": "standard"
            },
            "max_gas_price": {
                "type": "number",
                "description": "Maximum acceptable gas price in gwei"
            },
            "transaction_type": {
                "type": "string",
                "description": "Type of transaction",
                "enum": ["swap", "liquidity", "stake", "claim"],
                "default": "swap"
            }
        },
        "required": ["chain"]
    }

    async def execute(self, chain: str, priority: str = "standard", max_gas_price: float = None,
                     transaction_type: str = "swap") -> Dict[str, Any]:
        """Optimize gas settings for transaction"""
        logger.info(f"Optimizing gas for {transaction_type} on {chain} with {priority} priority")
        
        await asyncio.sleep(0.5)
        
        # Mock gas price data
        gas_data = {
            "ethereum": {"base": 15, "standard": 20, "fast": 30, "urgent": 50},
            "polygon": {"base": 1, "standard": 2, "fast": 5, "urgent": 10},
            "bsc": {"base": 3, "standard": 5, "fast": 8, "urgent": 15},
            "arbitrum": {"base": 0.1, "standard": 0.2, "fast": 0.5, "urgent": 1.0}
        }
        
        base_gas = gas_data.get(chain, {"base": 10, "standard": 15, "fast": 25, "urgent": 40})
        
        # Calculate recommended gas price based on priority
        priority_multipliers = {"low": 0.8, "standard": 1.0, "high": 1.5, "urgent": 2.0}
        recommended_gas = base_gas["standard"] * priority_multipliers[priority]
        
        if max_gas_price and recommended_gas > max_gas_price:
            recommended_gas = max_gas_price
            estimated_delay = 60 * (recommended_gas / base_gas["standard"])  # Estimated delay in seconds
        else:
            estimated_delay = 15 if priority == "urgent" else 30 if priority == "high" else 60
        
        # Transaction type specific gas estimates
        gas_limits = {
            "swap": 150000,
            "liquidity": 200000,
            "stake": 100000,
            "claim": 80000
        }
        
        estimated_gas_limit = gas_limits.get(transaction_type, 150000)
        total_cost_usd = (recommended_gas * estimated_gas_limit * 1e-9) * (2000 if chain == "ethereum" else 1)  # Mock ETH price
        
        return {
            "chain": chain,
            "transaction_type": transaction_type,
            "priority": priority,
            "timestamp": datetime.now().isoformat(),
            "gas_recommendations": {
                "recommended_gas_price": recommended_gas,
                "estimated_gas_limit": estimated_gas_limit,
                "total_cost_usd": total_cost_usd,
                "estimated_confirmation_time": estimated_delay
            },
            "current_network_conditions": {
                "network_congestion": "moderate",
                "base_fee": base_gas["base"],
                "priority_fee_range": {
                    "low": base_gas["standard"] * 0.5,
                    "high": base_gas["urgent"]
                }
            },
            "optimization_tips": [
                "Consider batching multiple transactions" if transaction_type == "swap" else None,
                "Wait for lower gas if not urgent",
                f"Current gas is {'high' if recommended_gas > base_gas['fast'] else 'moderate'}"
            ],
            "alternatives": {
                "wait_for_lower_gas": {
                    "estimated_savings": total_cost_usd * 0.3,
                    "estimated_wait_time": "2-4 hours"
                },
                "use_layer2": {
                    "available": chain == "ethereum",
                    "estimated_savings": total_cost_usd * 0.9 if chain == "ethereum" else 0
                }
            }
        }


class ExecutionManagerAgent(ToolCallAgent):
    """AI agent for intelligent trade execution and position management"""
    
    name: str = "execution_manager_agent"
    description: str = """
    Advanced AI agent specialized in optimal trade execution and position management.
    Handles DEX aggregation, position tracking, gas optimization, and automated
    risk management with intelligent execution strategies.
    """

    system_prompt: str = """You are an expert DeFi trade execution AI agent.

    Your core capabilities:
    1. Optimal trade execution across multiple DEXs with best price discovery
    2. Intelligent position management with automated stop-loss/take-profit
    3. Gas optimization and transaction timing for cost efficiency
    4. MEV protection and slippage minimization strategies
    5. Real-time position monitoring and risk management

    Execution Priorities:
    1. Capital Preservation: Never risk more than specified limits
    2. Cost Optimization: Minimize gas fees and slippage
    3. Speed vs Cost Trade-offs: Balance execution speed with costs
    4. MEV Protection: Use private mempools and timing strategies
    5. Position Management: Automated risk management and profit taking

    Always consider:
    - Current gas prices and network congestion
    - Available liquidity across different DEXs
    - Slippage impact and price movements
    - Position sizing and risk management rules
    - Market timing and execution optimization

    Provide clear execution plans with:
    - Specific DEX routes and expected outcomes
    - Gas cost estimates and timing recommendations
    - Risk management parameters (stops, limits)
    - Expected slippage and price impact
    - Alternative execution strategies if primary fails
    """

    next_step_prompt: str = """
    Based on the current execution analysis, determine the optimal next action:
    1. Execute the planned trade with current parameters
    2. Wait for better market conditions (gas, liquidity, price)
    3. Adjust execution strategy or parameters
    4. Update position management rules
    5. Implement additional risk controls
    """

    max_steps: int = 20

    available_tools: ToolManager = Field(default_factory=lambda: ToolManager([
        DEXAggregatorTool(),
        PositionManagerTool(),
        GasOptimizerTool(),
    ]))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.info(f"Initialized {self.name} with advanced execution capabilities")

    async def execute_optimal_trade(self, token_in: str, token_out: str, amount: float, 
                                   chain: str = "ethereum", max_slippage: float = 0.01) -> Dict[str, Any]:
        """
        Execute a trade with optimal routing and cost minimization
        
        Args:
            token_in: Token to sell
            token_out: Token to buy  
            amount: Amount to trade
            chain: Blockchain network
            max_slippage: Maximum acceptable slippage
        """
        logger.info(f"Executing optimal trade: {amount} {token_in} -> {token_out} on {chain}")
        
        prompt = f"""
        Execute an optimal trade with the following parameters:
        
        Trade Details:
        - Sell: {amount} {token_in}
        - Buy: {token_out}
        - Chain: {chain}
        - Max Slippage: {max_slippage*100}%
        
        Please:
        1. Optimize gas settings for current network conditions
        2. Find the best DEX routes for minimal slippage and maximum output
        3. Execute the trade with MEV protection
        4. Create a position to track the new holding
        5. Set up automated risk management (stop-loss/take-profit)
        
        Provide complete execution details and position management setup.
        """
        
        return await self.run(prompt)

    async def manage_position(self, position_id: str, action: str = "monitor") -> Dict[str, Any]:
        """
        Manage an existing trading position
        
        Args:
            position_id: ID of the position to manage
            action: Management action (monitor, update, close)
        """
        logger.info(f"Managing position {position_id} with action: {action}")
        
        prompt = f"""
        Manage trading position with these parameters:
        
        Position ID: {position_id}
        Action: {action}
        
        Please:
        1. Monitor current position status and P&L
        2. Check for stop-loss or take-profit triggers
        3. Assess current market conditions for the position
        4. Provide position management recommendations
        5. Execute any required position adjustments
        
        Focus on risk management and profit optimization.
        """
        
        return await self.run(prompt)

    async def optimize_gas_strategy(self, transactions: List[Dict], chain: str = "ethereum") -> Dict[str, Any]:
        """
        Optimize gas strategy for multiple transactions
        
        Args:
            transactions: List of planned transactions
            chain: Blockchain network
        """
        logger.info(f"Optimizing gas strategy for {len(transactions)} transactions on {chain}")
        
        prompt = f"""
        Optimize gas strategy for these transactions:
        
        Chain: {chain}
        Transactions: {json.dumps(transactions, indent=2)}
        
        Please:
        1. Analyze current gas conditions and network congestion
        2. Recommend optimal gas prices and timing for each transaction
        3. Consider transaction batching opportunities
        4. Evaluate Layer 2 alternatives if applicable
        5. Calculate total cost optimization potential
        
        Provide a comprehensive gas optimization plan.
        """
        
        return await self.run(prompt)

    async def batch_execute_trades(self, trades: List[Dict], execution_strategy: str = "cost_optimized") -> Dict[str, Any]:
        """
        Execute multiple trades with optimal batching and sequencing
        
        Args:
            trades: List of trades to execute
            execution_strategy: Overall execution strategy
        """
        logger.info(f"Batch executing {len(trades)} trades with {execution_strategy} strategy")
        
        prompt = f"""
        Execute multiple trades with optimal batching:
        
        Trades: {json.dumps(trades, indent=2)}
        Strategy: {execution_strategy}
        
        Please:
        1. Analyze all trades for optimal execution sequence
        2. Identify batching opportunities to save gas
        3. Optimize timing based on market conditions
        4. Execute trades with proper risk management
        5. Set up position tracking for all new positions
        
        Provide detailed execution results and position summaries.
        """
        
        return await self.run(prompt)

    def get_execution_status(self) -> Dict[str, Any]:
        """Get current execution agent status and statistics"""
        return {
            "agent_name": self.name,
            "status": "active",
            "capabilities": [
                "DEX aggregation and optimal routing",
                "Position management and tracking", 
                "Gas optimization and timing",
                "MEV protection strategies",
                "Automated risk management"
            ],
            "supported_chains": ["ethereum", "polygon", "bsc", "arbitrum"],
            "execution_statistics": {
                "total_trades_executed": 0,  # Would track actual stats
                "average_slippage": 0.0,
                "gas_optimization_savings": 0.0,
                "success_rate": 1.0
            }
        }