"""
Risk Management - Comprehensive risk assessment and portfolio protection
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from decimal import Decimal
import asyncio
from enum import Enum
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json
import math

from loguru import logger
from pydantic import BaseModel, Field


class RiskLevel(Enum):
    """Risk level classifications"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"
    CRITICAL = "critical"


class RiskType(Enum):
    """Types of risks in DeFi trading"""
    MARKET_RISK = "market_risk"
    LIQUIDITY_RISK = "liquidity_risk"
    SMART_CONTRACT_RISK = "smart_contract_risk"
    BRIDGE_RISK = "bridge_risk"
    ORACLE_RISK = "oracle_risk"
    REGULATORY_RISK = "regulatory_risk"
    OPERATIONAL_RISK = "operational_risk"
    COUNTERPARTY_RISK = "counterparty_risk"


@dataclass
class RiskMetric:
    """Individual risk metric"""
    name: str
    value: float
    risk_level: RiskLevel
    description: str
    impact_score: float = 0.0
    probability: float = 0.0
    mitigation_strategies: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PositionRisk:
    """Risk assessment for a specific position"""
    position_id: str
    token_pair: str
    position_size: float
    current_value: float
    entry_price: float
    current_price: float
    
    # Risk metrics
    var_95: float  # Value at Risk (95% confidence)
    var_99: float  # Value at Risk (99% confidence)
    expected_shortfall: float  # Conditional VaR
    max_drawdown: float
    beta: float  # Market beta
    correlation_btc: float
    correlation_eth: float
    
    # Risk scores
    market_risk_score: float
    liquidity_risk_score: float
    volatility_score: float
    concentration_risk_score: float
    
    overall_risk_score: float
    risk_level: RiskLevel
    
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class PortfolioRisk:
    """Portfolio-level risk assessment"""
    total_value: float
    positions_count: int
    
    # Portfolio metrics
    portfolio_var_95: float
    portfolio_var_99: float
    portfolio_beta: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    
    # Diversification metrics
    concentration_ratio: float  # HHI
    correlation_matrix: Dict[str, Dict[str, float]]
    
    # Risk scores by category
    market_risk: float
    liquidity_risk: float
    smart_contract_risk: float
    operational_risk: float
    
    overall_portfolio_risk: float
    risk_level: RiskLevel
    
    position_risks: List[PositionRisk] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    alerts: List[str] = field(default_factory=list)


class RiskModel(ABC):
    """Base class for risk models"""
    
    @abstractmethod
    async def calculate_var(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        pass
    
    @abstractmethod
    async def calculate_expected_shortfall(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        pass


class HistoricalSimulationModel(RiskModel):
    """Historical simulation risk model"""
    
    async def calculate_var(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate VaR using historical simulation"""
        if len(returns) == 0:
            return 0.0
        
        percentile = (1 - confidence) * 100
        var = np.percentile(returns, percentile)
        return abs(var)
    
    async def calculate_expected_shortfall(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate Expected Shortfall"""
        if len(returns) == 0:
            return 0.0
        
        var = await self.calculate_var(returns, confidence)
        percentile = (1 - confidence) * 100
        tail_returns = returns[returns <= np.percentile(returns, percentile)]
        
        if len(tail_returns) == 0:
            return var
        
        return abs(np.mean(tail_returns))


class MonteCarloModel(RiskModel):
    """Monte Carlo simulation risk model"""
    
    def __init__(self, num_simulations: int = 10000):
        self.num_simulations = num_simulations
    
    async def calculate_var(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate VaR using Monte Carlo simulation"""
        if len(returns) < 2:
            return 0.0
        
        # Generate random scenarios
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        simulated_returns = np.random.normal(
            mean_return, std_return, self.num_simulations
        )
        
        percentile = (1 - confidence) * 100
        var = np.percentile(simulated_returns, percentile)
        return abs(var)
    
    async def calculate_expected_shortfall(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate Expected Shortfall using Monte Carlo"""
        if len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        simulated_returns = np.random.normal(
            mean_return, std_return, self.num_simulations
        )
        
        percentile = (1 - confidence) * 100
        threshold = np.percentile(simulated_returns, percentile)
        tail_returns = simulated_returns[simulated_returns <= threshold]
        
        if len(tail_returns) == 0:
            return abs(threshold)
        
        return abs(np.mean(tail_returns))


class SmartContractRiskAnalyzer:
    """Analyzes smart contract risks"""
    
    def __init__(self):
        # Mock smart contract risk database
        self.contract_risks = {
            "uniswap_v3": {"risk_score": 0.1, "audited": True, "tvl": 5_000_000_000},
            "sushiswap": {"risk_score": 0.15, "audited": True, "tvl": 2_000_000_000},
            "curve": {"risk_score": 0.12, "audited": True, "tvl": 3_500_000_000},
            "compound": {"risk_score": 0.08, "audited": True, "tvl": 8_000_000_000},
            "aave": {"risk_score": 0.05, "audited": True, "tvl": 12_000_000_000},
            "yearn": {"risk_score": 0.2, "audited": True, "tvl": 1_000_000_000},
            "new_protocol": {"risk_score": 0.8, "audited": False, "tvl": 10_000_000}
        }
    
    async def analyze_protocol_risk(self, protocol_name: str) -> Dict[str, Any]:
        """Analyze smart contract risk for a protocol"""
        protocol_key = protocol_name.lower().replace(" ", "_")
        
        if protocol_key not in self.contract_risks:
            # Unknown protocol - assign high risk
            return {
                "protocol": protocol_name,
                "risk_score": 0.9,
                "risk_level": RiskLevel.VERY_HIGH,
                "factors": {
                    "audit_status": "unknown",
                    "tvl": 0,
                    "age_days": 0,
                    "exploit_history": "unknown"
                },
                "recommendations": [
                    "Avoid unknown protocols",
                    "Wait for security audit",
                    "Use only small test amounts"
                ]
            }
        
        risk_data = self.contract_risks[protocol_key]
        risk_score = risk_data["risk_score"]
        
        # Adjust risk based on TVL (higher TVL = lower risk)
        tvl_factor = min(risk_data["tvl"] / 1_000_000_000, 1.0)  # Max factor of 1.0 at $1B+ TVL
        adjusted_risk = risk_score * (1.2 - tvl_factor * 0.2)  # Reduce risk by up to 20% for high TVL
        
        risk_level = self._get_risk_level(adjusted_risk)
        
        return {
            "protocol": protocol_name,
            "risk_score": adjusted_risk,
            "risk_level": risk_level,
            "factors": {
                "audit_status": "audited" if risk_data["audited"] else "unaudited",
                "tvl": risk_data["tvl"],
                "base_risk": risk_score,
                "tvl_adjustment": tvl_factor
            },
            "recommendations": self._get_protocol_recommendations(adjusted_risk, risk_data)
        }
    
    def _get_risk_level(self, risk_score: float) -> RiskLevel:
        """Convert risk score to risk level"""
        if risk_score < 0.1:
            return RiskLevel.VERY_LOW
        elif risk_score < 0.25:
            return RiskLevel.LOW
        elif risk_score < 0.5:
            return RiskLevel.MEDIUM
        elif risk_score < 0.75:
            return RiskLevel.HIGH
        else:
            return RiskLevel.VERY_HIGH
    
    def _get_protocol_recommendations(self, risk_score: float, risk_data: Dict) -> List[str]:
        """Get recommendations based on protocol risk"""
        recommendations = []
        
        if risk_score > 0.7:
            recommendations.extend([
                "Consider avoiding this protocol",
                "If using, limit exposure to <5% of portfolio",
                "Monitor for security updates"
            ])
        elif risk_score > 0.4:
            recommendations.extend([
                "Use with caution",
                "Limit exposure to <20% of portfolio",
                "Monitor protocol governance"
            ])
        elif risk_score > 0.2:
            recommendations.extend([
                "Generally safe for moderate use",
                "Consider as part of diversified strategy",
                "Stay updated on protocol changes"
            ])
        else:
            recommendations.extend([
                "Low risk protocol",
                "Suitable for large allocations",
                "Monitor for new risks"
            ])
        
        if not risk_data.get("audited", False):
            recommendations.append("Wait for security audit completion")
        
        if risk_data.get("tvl", 0) < 50_000_000:  # Less than $50M TVL
            recommendations.append("Low TVL increases risk - use smaller amounts")
        
        return recommendations


class LiquidityRiskAnalyzer:
    """Analyzes liquidity risks for trading positions"""
    
    async def analyze_liquidity_risk(self, token_pair: str, position_size: float, 
                                   chain: str = "ethereum") -> Dict[str, Any]:
        """Analyze liquidity risk for a token pair"""
        
        # Mock liquidity data
        liquidity_data = {
            "ETH/USDC": {"total_liquidity": 500_000_000, "daily_volume": 100_000_000, "depth_1pct": 5_000_000},
            "BTC/ETH": {"total_liquidity": 200_000_000, "daily_volume": 50_000_000, "depth_1pct": 2_000_000},
            "MATIC/USDC": {"total_liquidity": 50_000_000, "daily_volume": 10_000_000, "depth_1pct": 500_000},
            "LINK/ETH": {"total_liquidity": 30_000_000, "daily_volume": 8_000_000, "depth_1pct": 300_000},
        }
        
        if token_pair not in liquidity_data:
            return {
                "token_pair": token_pair,
                "liquidity_risk_score": 0.8,
                "risk_level": RiskLevel.HIGH,
                "estimated_slippage": 0.05,  # 5%
                "recommendations": ["Insufficient liquidity data", "Use with extreme caution"]
            }
        
        data = liquidity_data[token_pair]
        
        # Calculate liquidity metrics
        position_to_liquidity_ratio = position_size / data["total_liquidity"]
        position_to_volume_ratio = position_size / data["daily_volume"]
        position_to_depth_ratio = position_size / data["depth_1pct"]
        
        # Estimate slippage based on position size
        if position_to_depth_ratio < 0.1:  # Less than 10% of 1% depth
            estimated_slippage = 0.001  # 0.1%
        elif position_to_depth_ratio < 0.5:
            estimated_slippage = 0.005  # 0.5%
        elif position_to_depth_ratio < 1.0:
            estimated_slippage = 0.01   # 1%
        else:
            estimated_slippage = 0.05   # 5%+
        
        # Calculate liquidity risk score
        liquidity_risk_score = min(
            position_to_liquidity_ratio * 10 +
            position_to_volume_ratio * 5 +
            position_to_depth_ratio * 2,
            1.0
        )
        
        risk_level = self._get_liquidity_risk_level(liquidity_risk_score)
        
        recommendations = self._get_liquidity_recommendations(
            liquidity_risk_score, estimated_slippage, position_to_volume_ratio
        )
        
        return {
            "token_pair": token_pair,
            "chain": chain,
            "liquidity_risk_score": liquidity_risk_score,
            "risk_level": risk_level,
            "estimated_slippage": estimated_slippage,
            "metrics": {
                "total_liquidity": data["total_liquidity"],
                "daily_volume": data["daily_volume"],
                "position_size": position_size,
                "position_to_liquidity_ratio": position_to_liquidity_ratio,
                "position_to_volume_ratio": position_to_volume_ratio,
                "depth_impact": position_to_depth_ratio
            },
            "recommendations": recommendations
        }
    
    def _get_liquidity_risk_level(self, risk_score: float) -> RiskLevel:
        """Convert liquidity risk score to risk level"""
        if risk_score < 0.1:
            return RiskLevel.VERY_LOW
        elif risk_score < 0.25:
            return RiskLevel.LOW
        elif risk_score < 0.5:
            return RiskLevel.MEDIUM
        elif risk_score < 0.75:
            return RiskLevel.HIGH
        else:
            return RiskLevel.VERY_HIGH
    
    def _get_liquidity_recommendations(self, risk_score: float, slippage: float, 
                                     volume_ratio: float) -> List[str]:
        """Get liquidity-based recommendations"""
        recommendations = []
        
        if risk_score > 0.7:
            recommendations.extend([
                "High liquidity risk - consider reducing position size",
                "Split large orders across multiple transactions",
                "Monitor order book depth before trading"
            ])
        elif risk_score > 0.4:
            recommendations.extend([
                "Moderate liquidity risk",
                "Consider using limit orders",
                "Monitor market impact"
            ])
        else:
            recommendations.append("Good liquidity conditions")
        
        if slippage > 0.02:  # >2% slippage
            recommendations.append(f"High expected slippage ({slippage*100:.1f}%)")
        
        if volume_ratio > 0.1:  # Position >10% of daily volume
            recommendations.append("Position size large relative to daily volume")
        
        return recommendations


class RiskManager:
    """Comprehensive risk management system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.risk_model = HistoricalSimulationModel()
        self.mc_model = MonteCarloModel()
        self.contract_analyzer = SmartContractRiskAnalyzer()
        self.liquidity_analyzer = LiquidityRiskAnalyzer()
        
        # Risk limits
        self.position_limits = self.config.get("position_limits", {})
        self.portfolio_limits = self.config.get("portfolio_limits", {})
        
        # Historical data storage (in production, use proper database)
        self.price_history = {}
        self.return_history = {}
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default risk management configuration"""
        return {
            "position_limits": {
                "max_position_size": 0.2,  # 20% of portfolio
                "max_single_token_exposure": 0.3,  # 30% in any single token
                "max_protocol_exposure": 0.4,  # 40% in any protocol
                "max_chain_exposure": 0.6,  # 60% on any chain
            },
            "portfolio_limits": {
                "max_leverage": 2.0,
                "max_correlation": 0.8,
                "min_diversification_ratio": 0.3,
                "max_drawdown": 0.15,  # 15%
                "var_limit_95": 0.05,  # 5% daily VaR
                "var_limit_99": 0.08   # 8% daily VaR
            },
            "risk_thresholds": {
                "position_risk_limit": 0.7,
                "portfolio_risk_limit": 0.6,
                "liquidity_risk_limit": 0.5,
                "smart_contract_risk_limit": 0.6
            }
        }
    
    async def assess_position_risk(self, position: Dict[str, Any]) -> PositionRisk:
        """Comprehensive position risk assessment"""
        token_pair = position.get("token_pair", "")
        position_size = position.get("size", 0)
        entry_price = position.get("entry_price", 0)
        current_price = position.get("current_price", entry_price)
        current_value = position_size * current_price
        
        # Generate mock historical returns for the token
        returns = self._generate_mock_returns(token_pair, 100)
        
        # Calculate risk metrics
        var_95 = await self.risk_model.calculate_var(returns, 0.95) * current_value
        var_99 = await self.risk_model.calculate_var(returns, 0.99) * current_value
        expected_shortfall = await self.risk_model.calculate_expected_shortfall(returns, 0.95) * current_value
        
        # Calculate other metrics
        volatility = np.std(returns) if len(returns) > 0 else 0
        max_drawdown = self._calculate_max_drawdown(returns)
        beta = self._calculate_beta(returns, self._generate_mock_returns("market", 100))
        
        # Market correlations
        btc_returns = self._generate_mock_returns("BTC/USDC", 100)
        eth_returns = self._generate_mock_returns("ETH/USDC", 100)
        
        correlation_btc = np.corrcoef(returns, btc_returns)[0, 1] if len(returns) > 1 else 0
        correlation_eth = np.corrcoef(returns, eth_returns)[0, 1] if len(returns) > 1 else 0
        
        # Risk scores
        market_risk_score = min(volatility * 5, 1.0)  # Scale volatility to 0-1
        volatility_score = min(volatility * 10, 1.0)
        
        # Liquidity risk
        liquidity_analysis = await self.liquidity_analyzer.analyze_liquidity_risk(
            token_pair, current_value
        )
        liquidity_risk_score = liquidity_analysis["liquidity_risk_score"]
        
        # Concentration risk (simplified)
        concentration_risk_score = min(current_value / 10000, 1.0)  # Relative to $10k portfolio
        
        # Overall risk score (weighted average)
        overall_risk_score = (
            market_risk_score * 0.3 +
            liquidity_risk_score * 0.25 +
            volatility_score * 0.25 +
            concentration_risk_score * 0.2
        )
        
        risk_level = self._get_position_risk_level(overall_risk_score)
        
        # Generate recommendations
        recommendations = self._get_position_recommendations(
            overall_risk_score, var_95, current_value, volatility
        )
        
        # Generate warnings
        warnings = self._get_position_warnings(
            overall_risk_score, var_95, current_value, liquidity_analysis
        )
        
        return PositionRisk(
            position_id=position.get("position_id", "unknown"),
            token_pair=token_pair,
            position_size=position_size,
            current_value=current_value,
            entry_price=entry_price,
            current_price=current_price,
            var_95=var_95,
            var_99=var_99,
            expected_shortfall=expected_shortfall,
            max_drawdown=max_drawdown,
            beta=beta,
            correlation_btc=correlation_btc,
            correlation_eth=correlation_eth,
            market_risk_score=market_risk_score,
            liquidity_risk_score=liquidity_risk_score,
            volatility_score=volatility_score,
            concentration_risk_score=concentration_risk_score,
            overall_risk_score=overall_risk_score,
            risk_level=risk_level,
            recommendations=recommendations,
            warnings=warnings
        )
    
    async def assess_portfolio_risk(self, positions: List[Dict[str, Any]]) -> PortfolioRisk:
        """Comprehensive portfolio risk assessment"""
        if not positions:
            return self._create_empty_portfolio_risk()
        
        # Calculate portfolio metrics
        total_value = sum(pos.get("size", 0) * pos.get("current_price", 0) for pos in positions)
        position_values = [pos.get("size", 0) * pos.get("current_price", 0) for pos in positions]
        
        # Portfolio-level risk calculations
        portfolio_returns = self._calculate_portfolio_returns(positions)
        
        portfolio_var_95 = await self.risk_model.calculate_var(portfolio_returns, 0.95) * total_value
        portfolio_var_99 = await self.risk_model.calculate_var(portfolio_returns, 0.99) * total_value
        
        # Calculate portfolio metrics
        portfolio_beta = self._calculate_portfolio_beta(positions)
        sharpe_ratio = self._calculate_sharpe_ratio(portfolio_returns)
        sortino_ratio = self._calculate_sortino_ratio(portfolio_returns)
        max_drawdown = self._calculate_max_drawdown(portfolio_returns)
        
        # Diversification metrics
        concentration_ratio = self._calculate_concentration_ratio(position_values)
        correlation_matrix = await self._calculate_correlation_matrix(positions)
        
        # Risk scores by category
        market_risk = self._calculate_portfolio_market_risk(positions)
        liquidity_risk = await self._calculate_portfolio_liquidity_risk(positions)
        smart_contract_risk = await self._calculate_portfolio_contract_risk(positions)
        operational_risk = 0.2  # Baseline operational risk
        
        # Overall portfolio risk
        overall_portfolio_risk = (
            market_risk * 0.35 +
            liquidity_risk * 0.25 +
            smart_contract_risk * 0.25 +
            operational_risk * 0.15
        )
        
        risk_level = self._get_portfolio_risk_level(overall_portfolio_risk)
        
        # Assess individual positions
        position_risks = []
        for position in positions:
            position_risk = await self.assess_position_risk(position)
            position_risks.append(position_risk)
        
        # Generate portfolio recommendations
        recommendations = self._get_portfolio_recommendations(
            overall_portfolio_risk, concentration_ratio, max_drawdown
        )
        
        # Generate alerts
        alerts = self._get_portfolio_alerts(
            portfolio_var_95, total_value, concentration_ratio, position_risks
        )
        
        return PortfolioRisk(
            total_value=total_value,
            positions_count=len(positions),
            portfolio_var_95=portfolio_var_95,
            portfolio_var_99=portfolio_var_99,
            portfolio_beta=portfolio_beta,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            concentration_ratio=concentration_ratio,
            correlation_matrix=correlation_matrix,
            market_risk=market_risk,
            liquidity_risk=liquidity_risk,
            smart_contract_risk=smart_contract_risk,
            operational_risk=operational_risk,
            overall_portfolio_risk=overall_portfolio_risk,
            risk_level=risk_level,
            position_risks=position_risks,
            recommendations=recommendations,
            alerts=alerts
        )
    
    def _generate_mock_returns(self, asset: str, periods: int) -> np.ndarray:
        """Generate mock historical returns for testing"""
        # Different volatilities for different assets
        volatility_map = {
            "ETH/USDC": 0.04,
            "BTC/USDC": 0.03,
            "MATIC/USDC": 0.06,
            "LINK/ETH": 0.05,
            "market": 0.035
        }
        
        volatility = volatility_map.get(asset, 0.05)
        mean_return = 0.0002  # Small positive drift
        
        returns = np.random.normal(mean_return, volatility, periods)
        return returns
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        if len(returns) == 0:
            return 0.0
        
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return abs(np.min(drawdown))
    
    def _calculate_beta(self, asset_returns: np.ndarray, market_returns: np.ndarray) -> float:
        """Calculate beta relative to market"""
        if len(asset_returns) < 2 or len(market_returns) < 2:
            return 1.0
        
        covariance = np.cov(asset_returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)
        
        if market_variance == 0:
            return 1.0
        
        return covariance / market_variance
    
    def _calculate_portfolio_returns(self, positions: List[Dict]) -> np.ndarray:
        """Calculate portfolio returns based on positions"""
        total_value = sum(pos.get("size", 0) * pos.get("current_price", 0) for pos in positions)
        
        if total_value == 0:
            return np.array([])
        
        # Weight positions by value
        weights = []
        position_returns = []
        
        for pos in positions:
            value = pos.get("size", 0) * pos.get("current_price", 0)
            weight = value / total_value
            weights.append(weight)
            
            returns = self._generate_mock_returns(pos.get("token_pair", ""), 100)
            position_returns.append(returns)
        
        # Calculate weighted portfolio returns
        portfolio_returns = np.zeros(100)
        for i, (weight, returns) in enumerate(zip(weights, position_returns)):
            portfolio_returns += weight * returns
        
        return portfolio_returns
    
    def _calculate_portfolio_beta(self, positions: List[Dict]) -> float:
        """Calculate portfolio beta"""
        total_value = sum(pos.get("size", 0) * pos.get("current_price", 0) for pos in positions)
        
        if total_value == 0:
            return 1.0
        
        portfolio_beta = 0
        market_returns = self._generate_mock_returns("market", 100)
        
        for pos in positions:
            value = pos.get("size", 0) * pos.get("current_price", 0)
            weight = value / total_value
            
            asset_returns = self._generate_mock_returns(pos.get("token_pair", ""), 100)
            asset_beta = self._calculate_beta(asset_returns, market_returns)
            portfolio_beta += weight * asset_beta
        
        return portfolio_beta
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        excess_returns = np.mean(returns) - (risk_free_rate / 365)  # Daily risk-free rate
        return excess_returns / np.std(returns)
    
    def _calculate_sortino_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio"""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = np.mean(returns) - (risk_free_rate / 365)
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_deviation = np.std(downside_returns)
        if downside_deviation == 0:
            return 0.0
        
        return excess_returns / downside_deviation
    
    def _calculate_concentration_ratio(self, position_values: List[float]) -> float:
        """Calculate Herfindahl-Hirschman Index for concentration"""
        if not position_values or sum(position_values) == 0:
            return 0.0
        
        total_value = sum(position_values)
        weights = [value / total_value for value in position_values]
        hhi = sum(weight ** 2 for weight in weights)
        
        return hhi
    
    async def _calculate_correlation_matrix(self, positions: List[Dict]) -> Dict[str, Dict[str, float]]:
        """Calculate correlation matrix between positions"""
        if len(positions) < 2:
            return {}
        
        correlation_matrix = {}
        token_pairs = [pos.get("token_pair", "") for pos in positions]
        
        for i, pair1 in enumerate(token_pairs):
            correlation_matrix[pair1] = {}
            returns1 = self._generate_mock_returns(pair1, 100)
            
            for j, pair2 in enumerate(token_pairs):
                if i == j:
                    correlation_matrix[pair1][pair2] = 1.0
                else:
                    returns2 = self._generate_mock_returns(pair2, 100)
                    if len(returns1) > 1 and len(returns2) > 1:
                        corr = np.corrcoef(returns1, returns2)[0, 1]
                        correlation_matrix[pair1][pair2] = corr if not np.isnan(corr) else 0.0
                    else:
                        correlation_matrix[pair1][pair2] = 0.0
        
        return correlation_matrix
    
    def _calculate_portfolio_market_risk(self, positions: List[Dict]) -> float:
        """Calculate portfolio market risk score"""
        if not positions:
            return 0.0
        
        total_risk = 0
        total_value = sum(pos.get("size", 0) * pos.get("current_price", 0) for pos in positions)
        
        for pos in positions:
            value = pos.get("size", 0) * pos.get("current_price", 0)
            weight = value / total_value if total_value > 0 else 0
            
            # Mock volatility-based risk score
            token_pair = pos.get("token_pair", "")
            volatility = self._get_token_volatility(token_pair)
            position_risk = min(volatility * 5, 1.0)  # Scale to 0-1
            
            total_risk += weight * position_risk
        
        return min(total_risk, 1.0)
    
    async def _calculate_portfolio_liquidity_risk(self, positions: List[Dict]) -> float:
        """Calculate portfolio liquidity risk score"""
        if not positions:
            return 0.0
        
        total_risk = 0
        total_value = sum(pos.get("size", 0) * pos.get("current_price", 0) for pos in positions)
        
        for pos in positions:
            value = pos.get("size", 0) * pos.get("current_price", 0)
            weight = value / total_value if total_value > 0 else 0
            
            token_pair = pos.get("token_pair", "")
            liquidity_analysis = await self.liquidity_analyzer.analyze_liquidity_risk(token_pair, value)
            position_liquidity_risk = liquidity_analysis["liquidity_risk_score"]
            
            total_risk += weight * position_liquidity_risk
        
        return min(total_risk, 1.0)
    
    async def _calculate_portfolio_contract_risk(self, positions: List[Dict]) -> float:
        """Calculate portfolio smart contract risk score"""
        if not positions:
            return 0.0
        
        # Mock protocol exposure for positions
        protocol_exposures = {}
        total_value = sum(pos.get("size", 0) * pos.get("current_price", 0) for pos in positions)
        
        for pos in positions:
            value = pos.get("size", 0) * pos.get("current_price", 0)
            # Mock protocol assignment based on token pair
            protocol = self._get_mock_protocol(pos.get("token_pair", ""))
            
            if protocol not in protocol_exposures:
                protocol_exposures[protocol] = 0
            protocol_exposures[protocol] += value
        
        total_contract_risk = 0
        for protocol, exposure in protocol_exposures.items():
            weight = exposure / total_value if total_value > 0 else 0
            protocol_analysis = await self.contract_analyzer.analyze_protocol_risk(protocol)
            protocol_risk = protocol_analysis["risk_score"]
            
            total_contract_risk += weight * protocol_risk
        
        return min(total_contract_risk, 1.0)
    
    def _get_token_volatility(self, token_pair: str) -> float:
        """Get mock volatility for token pair"""
        volatility_map = {
            "ETH/USDC": 0.04,
            "BTC/USDC": 0.03,
            "MATIC/USDC": 0.06,
            "LINK/ETH": 0.05,
            "AVAX/USDC": 0.07,
            "SOL/USDC": 0.08
        }
        return volatility_map.get(token_pair, 0.05)
    
    def _get_mock_protocol(self, token_pair: str) -> str:
        """Get mock protocol for token pair"""
        protocol_map = {
            "ETH/USDC": "uniswap_v3",
            "BTC/USDC": "sushiswap",
            "MATIC/USDC": "quickswap",
            "LINK/ETH": "uniswap_v3",
            "AVAX/USDC": "trader_joe"
        }
        return protocol_map.get(token_pair, "uniswap_v3")
    
    def _get_position_risk_level(self, risk_score: float) -> RiskLevel:
        """Convert position risk score to risk level"""
        if risk_score < 0.2:
            return RiskLevel.VERY_LOW
        elif risk_score < 0.4:
            return RiskLevel.LOW
        elif risk_score < 0.6:
            return RiskLevel.MEDIUM
        elif risk_score < 0.8:
            return RiskLevel.HIGH
        else:
            return RiskLevel.VERY_HIGH
    
    def _get_portfolio_risk_level(self, risk_score: float) -> RiskLevel:
        """Convert portfolio risk score to risk level"""
        if risk_score < 0.15:
            return RiskLevel.VERY_LOW
        elif risk_score < 0.3:
            return RiskLevel.LOW
        elif risk_score < 0.5:
            return RiskLevel.MEDIUM
        elif risk_score < 0.7:
            return RiskLevel.HIGH
        else:
            return RiskLevel.VERY_HIGH
    
    def _get_position_recommendations(self, risk_score: float, var_95: float, 
                                   current_value: float, volatility: float) -> List[str]:
        """Generate position-specific recommendations"""
        recommendations = []
        
        if risk_score > 0.8:
            recommendations.extend([
                "Consider closing position due to very high risk",
                "If keeping, reduce position size significantly",
                "Set tight stop-loss orders"
            ])
        elif risk_score > 0.6:
            recommendations.extend([
                "High risk position - monitor closely",
                "Consider reducing position size by 50%",
                "Implement strict risk management"
            ])
        elif risk_score > 0.4:
            recommendations.extend([
                "Moderate risk - maintain current monitoring",
                "Consider setting stop-loss at 10% below entry",
                "Monitor market conditions"
            ])
        else:
            recommendations.append("Low risk position - maintain current strategy")
        
        # VaR-based recommendations
        var_percentage = (var_95 / current_value) * 100 if current_value > 0 else 0
        if var_percentage > 10:
            recommendations.append(f"High VaR ({var_percentage:.1f}%) - consider hedging")
        
        # Volatility-based recommendations
        if volatility > 0.1:  # >10% daily volatility
            recommendations.append("High volatility detected - use smaller position sizes")
        
        return recommendations
    
    def _get_position_warnings(self, risk_score: float, var_95: float, 
                             current_value: float, liquidity_analysis: Dict) -> List[str]:
        """Generate position-specific warnings"""
        warnings = []
        
        if risk_score > 0.9:
            warnings.append("CRITICAL: Extremely high risk position")
        elif risk_score > 0.7:
            warnings.append("WARNING: High risk position requires immediate attention")
        
        # VaR warnings
        var_percentage = (var_95 / current_value) * 100 if current_value > 0 else 0
        if var_percentage > 15:
            warnings.append(f"WARNING: Very high VaR ({var_percentage:.1f}%)")
        
        # Liquidity warnings
        if liquidity_analysis["liquidity_risk_score"] > 0.7:
            warnings.append("WARNING: Poor liquidity conditions")
        
        if liquidity_analysis["estimated_slippage"] > 0.05:
            warnings.append(f"WARNING: High expected slippage ({liquidity_analysis['estimated_slippage']*100:.1f}%)")
        
        return warnings
    
    def _get_portfolio_recommendations(self, risk_score: float, concentration_ratio: float, 
                                     max_drawdown: float) -> List[str]:
        """Generate portfolio-level recommendations"""
        recommendations = []
        
        if risk_score > 0.7:
            recommendations.extend([
                "Portfolio risk is very high - consider reducing overall exposure",
                "Implement hedging strategies",
                "Review and reduce position sizes"
            ])
        elif risk_score > 0.5:
            recommendations.extend([
                "Moderate to high portfolio risk",
                "Consider diversifying into lower-risk assets",
                "Monitor correlations between positions"
            ])
        
        # Concentration recommendations
        if concentration_ratio > 0.5:  # High concentration
            recommendations.extend([
                "Portfolio is highly concentrated",
                "Consider diversifying across more assets",
                "Reduce largest position sizes"
            ])
        elif concentration_ratio > 0.3:
            recommendations.append("Moderate concentration - consider additional diversification")
        
        # Drawdown recommendations
        if max_drawdown > 0.15:  # >15% drawdown
            recommendations.extend([
                "High historical drawdown detected",
                "Review risk management strategies",
                "Consider implementing dynamic hedging"
            ])
        
        return recommendations
    
    def _get_portfolio_alerts(self, portfolio_var_95: float, total_value: float, 
                            concentration_ratio: float, position_risks: List[PositionRisk]) -> List[str]:
        """Generate portfolio alerts"""
        alerts = []
        
        # VaR alerts
        var_percentage = (portfolio_var_95 / total_value) * 100 if total_value > 0 else 0
        if var_percentage > self.portfolio_limits.get("var_limit_95", 5) * 100:
            alerts.append(f"ALERT: Portfolio VaR exceeds limit ({var_percentage:.1f}%)")
        
        # Concentration alerts
        if concentration_ratio > 0.6:
            alerts.append("ALERT: High portfolio concentration risk")
        
        # Position-specific alerts
        critical_positions = [pr for pr in position_risks if pr.risk_level == RiskLevel.VERY_HIGH]
        if critical_positions:
            alerts.append(f"ALERT: {len(critical_positions)} positions at critical risk level")
        
        return alerts
    
    def _create_empty_portfolio_risk(self) -> PortfolioRisk:
        """Create empty portfolio risk assessment"""
        return PortfolioRisk(
            total_value=0.0,
            positions_count=0,
            portfolio_var_95=0.0,
            portfolio_var_99=0.0,
            portfolio_beta=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            concentration_ratio=0.0,
            correlation_matrix={},
            market_risk=0.0,
            liquidity_risk=0.0,
            smart_contract_risk=0.0,
            operational_risk=0.0,
            overall_portfolio_risk=0.0,
            risk_level=RiskLevel.VERY_LOW,
            position_risks=[],
            recommendations=["No active positions"],
            alerts=[]
        )
    
    async def check_risk_limits(self, portfolio_risk: PortfolioRisk) -> Dict[str, Any]:
        """Check if portfolio exceeds risk limits"""
        violations = []
        warnings = []
        
        # Check VaR limits
        var_percentage = (portfolio_risk.portfolio_var_95 / portfolio_risk.total_value) * 100 if portfolio_risk.total_value > 0 else 0
        if var_percentage > self.portfolio_limits.get("var_limit_95", 5) * 100:
            violations.append({
                "type": "var_limit",
                "current": var_percentage,
                "limit": self.portfolio_limits.get("var_limit_95", 5) * 100,
                "severity": "high"
            })
        
        # Check concentration limits
        if portfolio_risk.concentration_ratio > 0.6:
            violations.append({
                "type": "concentration",
                "current": portfolio_risk.concentration_ratio,
                "limit": 0.6,
                "severity": "medium"
            })
        
        # Check drawdown limits
        if portfolio_risk.max_drawdown > self.portfolio_limits.get("max_drawdown", 0.15):
            violations.append({
                "type": "drawdown",
                "current": portfolio_risk.max_drawdown,
                "limit": self.portfolio_limits.get("max_drawdown", 0.15),
                "severity": "high"
            })
        
        # Check individual position limits
        for position_risk in portfolio_risk.position_risks:
            position_percentage = (position_risk.current_value / portfolio_risk.total_value) if portfolio_risk.total_value > 0 else 0
            if position_percentage > self.position_limits.get("max_position_size", 0.2):
                violations.append({
                    "type": "position_size",
                    "position_id": position_risk.position_id,
                    "current": position_percentage,
                    "limit": self.position_limits.get("max_position_size", 0.2),
                    "severity": "medium"
                })
        
        return {
            "violations": violations,
            "warnings": warnings,
            "risk_limit_status": "violated" if violations else "compliant",
            "total_violations": len(violations),
            "requires_action": len([v for v in violations if v["severity"] == "high"]) > 0
        }
    
    async def suggest_risk_reductions(self, portfolio_risk: PortfolioRisk) -> List[Dict[str, Any]]:
        """Suggest specific actions to reduce portfolio risk"""
        suggestions = []
        
        # If overall risk is too high
        if portfolio_risk.overall_portfolio_risk > 0.7:
            suggestions.append({
                "action": "reduce_overall_exposure",
                "description": "Reduce total portfolio exposure by 25-30%",
                "priority": "high",
                "expected_impact": "Significant risk reduction"
            })
        
        # If concentration is too high
        if portfolio_risk.concentration_ratio > 0.5:
            largest_positions = sorted(
                portfolio_risk.position_risks,
                key=lambda x: x.current_value,
                reverse=True
            )[:3]
            
            for pos in largest_positions:
                suggestions.append({
                    "action": "reduce_position_size",
                    "position_id": pos.position_id,
                    "description": f"Reduce {pos.token_pair} position by 30-50%",
                    "priority": "medium",
                    "expected_impact": "Lower concentration risk"
                })
        
        # If high-risk positions exist
        high_risk_positions = [
            pr for pr in portfolio_risk.position_risks 
            if pr.risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]
        ]
        
        for pos in high_risk_positions:
            suggestions.append({
                "action": "implement_hedging",
                "position_id": pos.position_id,
                "description": f"Hedge {pos.token_pair} position or set tighter stop-loss",
                "priority": "high" if pos.risk_level == RiskLevel.VERY_HIGH else "medium",
                "expected_impact": "Reduced position-specific risk"
            })
        
        # If correlations are too high
        high_correlations = []
        for pair1, correlations in portfolio_risk.correlation_matrix.items():
            for pair2, corr in correlations.items():
                if pair1 != pair2 and abs(corr) > 0.8:
                    high_correlations.append((pair1, pair2, corr))
        
        if high_correlations:
            suggestions.append({
                "action": "diversify_correlations",
                "description": "Reduce positions in highly correlated assets",
                "priority": "medium",
                "expected_impact": "Improved diversification"
            })
        
        return suggestions
    
    def get_risk_summary(self, portfolio_risk: PortfolioRisk) -> Dict[str, Any]:
        """Get comprehensive risk summary"""
        return {
            "timestamp": datetime.now().isoformat(),
            "portfolio_overview": {
                "total_value": portfolio_risk.total_value,
                "positions_count": portfolio_risk.positions_count,
                "overall_risk_level": portfolio_risk.risk_level.value,
                "overall_risk_score": portfolio_risk.overall_portfolio_risk
            },
            "key_metrics": {
                "var_95_pct": (portfolio_risk.portfolio_var_95 / portfolio_risk.total_value) * 100 if portfolio_risk.total_value > 0 else 0,
                "max_drawdown_pct": portfolio_risk.max_drawdown * 100,
                "concentration_ratio": portfolio_risk.concentration_ratio,
                "portfolio_beta": portfolio_risk.portfolio_beta,
                "sharpe_ratio": portfolio_risk.sharpe_ratio
            },
            "risk_breakdown": {
                "market_risk": portfolio_risk.market_risk,
                "liquidity_risk": portfolio_risk.liquidity_risk,
                "smart_contract_risk": portfolio_risk.smart_contract_risk,
                "operational_risk": portfolio_risk.operational_risk
            },
            "high_risk_positions": len([
                pr for pr in portfolio_risk.position_risks 
                if pr.risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]
            ]),
            "alerts_count": len(portfolio_risk.alerts),
            "recommendations_count": len(portfolio_risk.recommendations)
        }