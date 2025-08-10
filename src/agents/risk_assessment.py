"""
Risk Assessment Agent - AI-powered risk analysis and portfolio management
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import asyncio
from decimal import Decimal
import json
import numpy as np

from spoon_ai.agents import ToolCallAgent
from spoon_ai.tools.base import BaseTool
from spoon_ai.tools import ToolManager
from pydantic import Field
from loguru import logger


class SmartContractAnalyzerTool(BaseTool):
    """Smart contract security and risk assessment tool"""
    
    name: str = "smart_contract_analyzer"
    description: str = "Analyze smart contract security risks and vulnerabilities"
    parameters: dict = {
        "type": "object",
        "properties": {
            "contract_address": {
                "type": "string",
                "description": "Smart contract address to analyze"
            },
            "chain": {
                "type": "string",
                "description": "Blockchain network",
                "enum": ["ethereum", "polygon", "bsc", "arbitrum"]
            },
            "analysis_depth": {
                "type": "string",
                "description": "Depth of security analysis",
                "enum": ["basic", "comprehensive", "audit_level"],
                "default": "comprehensive"
            }
        },
        "required": ["contract_address", "chain"]
    }

    async def execute(self, contract_address: str, chain: str, analysis_depth: str = "comprehensive") -> Dict[str, Any]:
        """Analyze smart contract security and risks"""
        logger.info(f"Analyzing contract {contract_address} on {chain} with {analysis_depth} depth")
        
        # Simulate contract analysis
        await asyncio.sleep(2)
        
        # Mock security analysis results
        security_checks = {
            "reentrancy": {"status": "pass", "risk": "low", "details": "No reentrancy vulnerabilities found"},
            "overflow": {"status": "pass", "risk": "none", "details": "SafeMath implemented"},
            "access_control": {"status": "warning", "risk": "medium", "details": "Admin functions not time-locked"},
            "oracle_manipulation": {"status": "pass", "risk": "low", "details": "Chainlink oracles with proper validation"},
            "flash_loan_attacks": {"status": "pass", "risk": "low", "details": "Protected against flash loan manipulation"}
        }
        
        # Calculate overall risk score
        risk_scores = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
        total_risk = sum(risk_scores[check["risk"]] for check in security_checks.values())
        max_risk = len(security_checks) * 4
        risk_score = (total_risk / max_risk) * 10  # Scale to 0-10
        
        return {
            "contract_address": contract_address,
            "chain": chain,
            "timestamp": datetime.now().isoformat(),
            "analysis_depth": analysis_depth,
            "security_checks": security_checks,
            "overall_risk_score": round(risk_score, 2),
            "risk_level": "low" if risk_score < 3 else "medium" if risk_score < 6 else "high",
            "audit_status": {
                "has_audit": True,
                "auditor": "CertiK",
                "audit_date": "2024-01-15",
                "issues_found": 2,
                "critical_issues": 0
            },
            "recommendations": [
                "Monitor admin key activities",
                "Implement time-lock for critical functions",
                "Regular security updates"
            ],
            "trust_score": 8.5  # Out of 10
        }


class VolatilityPredictorTool(BaseTool):
    """AI-powered volatility and price prediction tool"""
    
    name: str = "volatility_predictor"
    description: str = "Predict price volatility and potential price movements using AI models"
    parameters: dict = {
        "type": "object",
        "properties": {
            "token_symbol": {
                "type": "string",
                "description": "Token to analyze (e.g., ETH, BTC, USDC)"
            },
            "prediction_horizon": {
                "type": "string",
                "description": "Time horizon for prediction",
                "enum": ["1h", "4h", "24h", "7d"],
                "default": "24h"
            },
            "model_type": {
                "type": "string",
                "description": "ML model to use for prediction",
                "enum": ["lstm", "transformer", "ensemble"],
                "default": "ensemble"
            },
            "confidence_threshold": {
                "type": "number",
                "description": "Minimum confidence threshold for predictions",
                "default": 0.7
            }
        },
        "required": ["token_symbol"]
    }

    async def execute(self, token_symbol: str, prediction_horizon: str = "24h", 
                     model_type: str = "ensemble", confidence_threshold: float = 0.7) -> Dict[str, Any]:
        """Predict volatility and price movements"""
        logger.info(f"Predicting volatility for {token_symbol} over {prediction_horizon} using {model_type}")
        
        # Simulate ML model inference
        await asyncio.sleep(1.5)
        
        # Mock prediction results
        current_price = 2500.0 if token_symbol == "ETH" else 50000.0 if token_symbol == "BTC" else 1.0
        
        # Generate realistic volatility predictions
        volatility_24h = np.random.uniform(0.02, 0.08)  # 2-8% daily volatility
        price_change_probability = np.random.uniform(0.6, 0.9)  # 60-90% confidence
        
        predicted_range = {
            "low": current_price * (1 - volatility_24h),
            "high": current_price * (1 + volatility_24h),
            "expected": current_price * (1 + np.random.uniform(-0.01, 0.01))
        }
        
        return {
            "token": token_symbol,
            "current_price": current_price,
            "prediction_horizon": prediction_horizon,
            "model_type": model_type,
            "timestamp": datetime.now().isoformat(),
            "volatility_metrics": {
                "predicted_volatility_24h": round(volatility_24h * 100, 2),
                "historical_volatility_7d": round(volatility_24h * 1.2 * 100, 2),
                "volatility_percentile": 65,  # Compared to historical data
                "volatility_trend": "increasing"
            },
            "price_predictions": {
                "predicted_range": predicted_range,
                "confidence_score": price_change_probability,
                "directional_bias": "bullish" if predicted_range["expected"] > current_price else "bearish",
                "key_levels": {
                    "support": current_price * 0.95,
                    "resistance": current_price * 1.05
                }
            },
            "risk_indicators": {
                "vix_crypto": 45.2,  # Crypto fear & greed index
                "correlation_with_btc": 0.78,
                "market_regime": "trending",
                "liquidity_risk": "low"
            },
            "model_performance": {
                "accuracy_7d": 0.73,
                "precision": 0.81,
                "last_calibration": "2024-01-15T10:30:00Z"
            }
        }


class PortfolioOptimizerTool(BaseTool):
    """Portfolio allocation and risk optimization tool"""
    
    name: str = "portfolio_optimizer"
    description: str = "Optimize portfolio allocation based on risk-return objectives"
    parameters: dict = {
        "type": "object",
        "properties": {
            "current_positions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "token": {"type": "string"},
                        "amount_usd": {"type": "number"},
                        "current_price": {"type": "number"}
                    }
                },
                "description": "Current portfolio positions"
            },
            "risk_tolerance": {
                "type": "string",
                "description": "Risk tolerance level",
                "enum": ["conservative", "moderate", "aggressive"],
                "default": "moderate"
            },
            "optimization_objective": {
                "type": "string",
                "description": "Portfolio optimization objective",
                "enum": ["max_sharpe", "min_volatility", "max_return"],
                "default": "max_sharpe"
            },
            "total_portfolio_value": {
                "type": "number",
                "description": "Total portfolio value in USD"
            }
        },
        "required": ["current_positions", "total_portfolio_value"]
    }

    async def execute(self, current_positions: List[Dict], total_portfolio_value: float,
                     risk_tolerance: str = "moderate", optimization_objective: str = "max_sharpe") -> Dict[str, Any]:
        """Optimize portfolio allocation"""
        logger.info(f"Optimizing portfolio of ${total_portfolio_value:,.2f} with {risk_tolerance} risk tolerance")
        
        await asyncio.sleep(1)
        
        # Calculate current allocation
        current_allocation = {}
        for position in current_positions:
            token = position["token"]
            allocation = (position["amount_usd"] / total_portfolio_value) * 100
            current_allocation[token] = allocation
        
        # Generate optimized allocation based on risk tolerance
        risk_multipliers = {"conservative": 0.6, "moderate": 1.0, "aggressive": 1.4}
        multiplier = risk_multipliers[risk_tolerance]
        
        # Mock optimal allocation (in practice, this would use Modern Portfolio Theory)
        if risk_tolerance == "conservative":
            optimal_allocation = {"BTC": 40, "ETH": 30, "USDC": 20, "stable_yield": 10}
        elif risk_tolerance == "moderate": 
            optimal_allocation = {"BTC": 35, "ETH": 35, "ALT_COINS": 20, "USDC": 10}
        else:  # aggressive
            optimal_allocation = {"ETH": 40, "ALT_COINS": 35, "BTC": 20, "DEFI_TOKENS": 5}
        
        # Calculate rebalancing needed
        rebalancing_actions = []
        for token, target_pct in optimal_allocation.items():
            current_pct = current_allocation.get(token, 0)
            difference = target_pct - current_pct
            if abs(difference) > 2:  # Only rebalance if >2% difference
                action = "buy" if difference > 0 else "sell"
                amount = abs(difference) * total_portfolio_value / 100
                rebalancing_actions.append({
                    "token": token,
                    "action": action,
                    "amount_usd": amount,
                    "current_allocation": current_pct,
                    "target_allocation": target_pct
                })
        
        return {
            "timestamp": datetime.now().isoformat(),
            "portfolio_value": total_portfolio_value,
            "risk_tolerance": risk_tolerance,
            "optimization_objective": optimization_objective,
            "current_allocation": current_allocation,
            "optimal_allocation": optimal_allocation,
            "rebalancing_actions": rebalancing_actions,
            "portfolio_metrics": {
                "expected_annual_return": 0.15 * multiplier,
                "expected_volatility": 0.25 * multiplier,
                "sharpe_ratio": 0.6,
                "max_drawdown": 0.15 * multiplier,
                "diversification_ratio": 0.75
            },
            "risk_analysis": {
                "portfolio_beta": 1.2 * multiplier,
                "concentration_risk": "low" if len(optimal_allocation) >= 4 else "medium",
                "correlation_risk": "medium",
                "liquidity_risk": "low"
            },
            "recommendations": [
                "Maintain diversification across asset classes",
                f"Rebalance portfolio to align with {risk_tolerance} risk profile",
                "Monitor correlation between positions",
                "Set stop-losses at portfolio level"
            ]
        }


class RiskAssessmentAgent(ToolCallAgent):
    """AI agent specialized in risk analysis and portfolio management"""
    
    name: str = "risk_assessment_agent"
    description: str = """
    Advanced AI agent focused on comprehensive risk analysis and portfolio optimization.
    Evaluates smart contract risks, predicts market volatility, and optimizes portfolio
    allocations based on sophisticated risk models and AI predictions.
    """

    system_prompt: str = """You are an expert DeFi risk assessment AI agent.

    Your core expertise includes:
    1. Smart contract security analysis and audit evaluation
    2. Market volatility prediction and risk modeling
    3. Portfolio optimization using Modern Portfolio Theory
    4. Multi-dimensional risk assessment (market, technical, operational)
    5. Dynamic position sizing and risk management

    Risk Assessment Framework:
    - Market Risk: Price volatility, correlation, liquidity
    - Credit Risk: Counterparty, protocol, smart contract risks
    - Operational Risk: Gas costs, MEV, timing, execution risks
    - Regulatory Risk: Compliance and legal considerations

    Always provide:
    - Quantitative risk scores (1-10 scale)
    - Specific risk mitigation strategies
    - Position sizing recommendations
    - Stop-loss and take-profit levels
    - Worst-case scenario analysis
    - Expected vs. maximum loss calculations

    Be conservative in risk estimates and always prioritize capital preservation.
    Use statistical models and historical data to support your assessments.
    """

    next_step_prompt: str = """
    Based on the risk analysis completed, determine the next appropriate action:
    1. Additional risk factors to evaluate
    2. Portfolio adjustments needed
    3. Risk mitigation strategies to implement
    4. Position sizing modifications
    5. Monitoring and alert setup requirements
    """

    max_steps: int = 12

    available_tools: ToolManager = Field(default_factory=lambda: ToolManager([
        SmartContractAnalyzerTool(),
        VolatilityPredictorTool(),
        PortfolioOptimizerTool(),
    ]))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.info(f"Initialized {self.name} with advanced risk analysis capabilities")

    async def assess_trading_risk(self, strategy: str, amount_usd: float, tokens: List[str] = None) -> Dict[str, Any]:
        """
        Comprehensive risk assessment for a trading strategy
        
        Args:
            strategy: Description of the trading strategy
            amount_usd: Capital amount to risk
            tokens: List of tokens involved in the strategy
        """
        if tokens is None:
            tokens = ["ETH", "BTC"]
            
        logger.info(f"Assessing risk for strategy: {strategy}, capital: ${amount_usd:,.2f}")
        
        prompt = f"""
        Perform a comprehensive risk assessment for this trading strategy:
        
        Strategy: {strategy}
        Capital Amount: ${amount_usd:,.2f}
        Tokens Involved: {', '.join(tokens)}
        
        Please analyze:
        1. Smart contract risks for all involved protocols
        2. Market volatility and price prediction for each token
        3. Portfolio impact and optimal position sizing
        4. Maximum potential loss scenarios (95% and 99% confidence)
        5. Risk-adjusted return expectations
        6. Specific risk mitigation strategies
        
        Provide a detailed risk report with actionable recommendations.
        """
        
        return await self.run(prompt)

    async def optimize_portfolio(self, current_positions: List[Dict], risk_profile: str = "moderate") -> Dict[str, Any]:
        """
        Optimize portfolio allocation based on risk-return objectives
        
        Args:
            current_positions: List of current portfolio positions
            risk_profile: Risk tolerance (conservative, moderate, aggressive)
        """
        total_value = sum(pos["amount_usd"] for pos in current_positions)
        logger.info(f"Optimizing portfolio worth ${total_value:,.2f} with {risk_profile} risk profile")
        
        prompt = f"""
        Optimize this portfolio allocation:
        
        Current Positions: {json.dumps(current_positions, indent=2)}
        Risk Profile: {risk_profile}
        Total Portfolio Value: ${total_value:,.2f}
        
        Please:
        1. Analyze current portfolio allocation and risks
        2. Predict volatility for each asset
        3. Generate optimal allocation recommendations
        4. Calculate rebalancing actions needed
        5. Assess portfolio-level risk metrics
        6. Provide specific implementation steps
        
        Focus on risk-adjusted returns and proper diversification.
        """
        
        return await self.run(prompt)

    async def evaluate_protocol_safety(self, protocol_name: str, contract_addresses: List[str], 
                                     chain: str = "ethereum") -> Dict[str, Any]:
        """
        Evaluate DeFi protocol safety and security risks
        
        Args:
            protocol_name: Name of the DeFi protocol
            contract_addresses: List of smart contract addresses
            chain: Blockchain network
        """
        logger.info(f"Evaluating safety of {protocol_name} protocol on {chain}")
        
        prompt = f"""
        Evaluate the safety and security of this DeFi protocol:
        
        Protocol: {protocol_name}
        Chain: {chain}
        Contract Addresses: {', '.join(contract_addresses)}
        
        Please assess:
        1. Smart contract security for each contract
        2. Protocol governance and admin risks
        3. Historical performance and incidents
        4. Liquidity and counterparty risks
        5. Oracle dependencies and manipulation risks
        6. Overall protocol trust score
        
        Provide a comprehensive safety evaluation with specific risk ratings.
        """
        
        return await self.run(prompt)

    async def calculate_position_size(self, strategy_risk_score: float, portfolio_value: float, 
                                    max_risk_per_trade: float = 0.02) -> Dict[str, Any]:
        """
        Calculate optimal position size based on risk management principles
        
        Args:
            strategy_risk_score: Risk score for the strategy (1-10)
            portfolio_value: Total portfolio value
            max_risk_per_trade: Maximum risk per trade as percentage of portfolio
        """
        logger.info(f"Calculating position size for risk score {strategy_risk_score}")
        
        prompt = f"""
        Calculate optimal position sizing:
        
        Strategy Risk Score: {strategy_risk_score}/10
        Portfolio Value: ${portfolio_value:,.2f}
        Max Risk Per Trade: {max_risk_per_trade*100}%
        
        Please calculate:
        1. Kelly Criterion optimal position size
        2. Risk-adjusted position size recommendations
        3. Stop-loss and take-profit levels
        4. Position size for different confidence scenarios
        5. Risk budget allocation
        6. Position monitoring requirements
        
        Provide specific position sizing recommendations with rationale.
        """
        
        return await self.run(prompt)

    def get_risk_limits(self) -> Dict[str, float]:
        """Get current risk management limits"""
        return {
            "max_portfolio_risk": 0.02,  # 2% max risk per trade
            "max_position_size": 0.1,    # 10% max position size
            "max_correlation": 0.7,      # Maximum correlation between positions
            "min_liquidity": 100000,     # Minimum liquidity in USD
            "max_drawdown": 0.15,        # 15% maximum portfolio drawdown
            "volatility_limit": 0.8      # Maximum volatility threshold
        }