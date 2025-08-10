"""
Basic functionality tests for DeFi AI Trading Bot
"""

import pytest
import asyncio
from unittest.mock import Mock, patch
import json

from src.agents.market_intelligence import MarketIntelligenceAgent, ChainbaseAnalyticsTool, PriceAggregatorTool
from src.agents.risk_assessment import RiskAssessmentAgent, SmartContractAnalyzerTool
from src.agents.execution_manager import ExecutionManagerAgent, DEXAggregatorTool
from src.cli.main import TradingBotOrchestrator


class TestMarketIntelligenceAgent:
    """Test Market Intelligence Agent functionality"""

    @pytest.fixture
    def mock_llm(self):
        """Mock LLM for testing"""
        mock = Mock()
        mock.generate = Mock(return_value="Mock AI response")
        return mock

    @pytest.fixture 
    def market_agent(self, mock_llm):
        """Create market intelligence agent for testing"""
        return MarketIntelligenceAgent(llm=mock_llm)

    @pytest.mark.asyncio
    async def test_chainbase_analytics_tool(self):
        """Test Chainbase analytics tool"""
        tool = ChainbaseAnalyticsTool()
        
        result = await tool.execute(
            chain="ethereum",
            analysis_type="whale_movements",
            timeframe="24h"
        )
        
        assert result["chain"] == "ethereum"
        assert result["analysis"] == "whale_movements"
        assert "large_transactions" in result
        assert "summary" in result

    @pytest.mark.asyncio
    async def test_price_aggregator_tool(self):
        """Test price aggregation tool"""
        tool = PriceAggregatorTool()
        
        result = await tool.execute(
            token_symbol="ETH",
            vs_currency="USD"
        )
        
        assert result["token"] == "ETH"
        assert result["vs_currency"] == "USD"
        assert "prices" in result
        assert "average_price" in result
        assert "arbitrage_opportunities" in result

    def test_agent_initialization(self, market_agent):
        """Test agent initializes correctly"""
        assert market_agent.name == "market_intelligence_agent"
        assert len(market_agent.available_tools.tools) >= 3
        assert market_agent.max_steps == 15


class TestRiskAssessmentAgent:
    """Test Risk Assessment Agent functionality"""

    @pytest.fixture
    def mock_llm(self):
        mock = Mock()
        mock.generate = Mock(return_value="Mock risk analysis")
        return mock

    @pytest.fixture
    def risk_agent(self, mock_llm):
        return RiskAssessmentAgent(llm=mock_llm)

    @pytest.mark.asyncio
    async def test_smart_contract_analyzer(self):
        """Test smart contract security analysis"""
        tool = SmartContractAnalyzerTool()
        
        result = await tool.execute(
            contract_address="0x1234567890abcdef",
            chain="ethereum",
            analysis_depth="comprehensive"
        )
        
        assert result["contract_address"] == "0x1234567890abcdef"
        assert result["chain"] == "ethereum"
        assert "security_checks" in result
        assert "overall_risk_score" in result
        assert "audit_status" in result
        
        # Verify risk score is within valid range
        assert 0 <= result["overall_risk_score"] <= 10

    def test_risk_limits(self, risk_agent):
        """Test risk management limits"""
        limits = risk_agent.get_risk_limits()
        
        assert "max_portfolio_risk" in limits
        assert "max_position_size" in limits
        assert "max_drawdown" in limits
        
        # Verify reasonable default limits
        assert limits["max_portfolio_risk"] <= 0.05  # Max 5% risk
        assert limits["max_position_size"] <= 0.2    # Max 20% position


class TestExecutionManagerAgent:
    """Test Execution Manager Agent functionality"""

    @pytest.fixture
    def mock_llm(self):
        mock = Mock()
        mock.generate = Mock(return_value="Mock execution response")
        return mock

    @pytest.fixture
    def execution_agent(self, mock_llm):
        return ExecutionManagerAgent(llm=mock_llm)

    @pytest.mark.asyncio
    async def test_dex_aggregator_tool(self):
        """Test DEX aggregation functionality"""
        tool = DEXAggregatorTool()
        
        result = await tool.execute(
            token_in="ETH",
            token_out="USDC", 
            amount_in=1.0,
            chain="ethereum",
            slippage_tolerance=0.01
        )
        
        assert "trade_id" in result
        assert result["status"] == "completed"
        assert result["chain"] == "ethereum"
        assert result["input"]["token"] == "ETH"
        assert result["output"]["token"] == "USDC"
        assert "execution_details" in result
        assert "fees" in result

    def test_execution_status(self, execution_agent):
        """Test execution agent status reporting"""
        status = execution_agent.get_execution_status()
        
        assert status["agent_name"] == "execution_manager_agent"
        assert status["status"] == "active"
        assert "capabilities" in status
        assert "supported_chains" in status
        assert "execution_statistics" in status
        
        # Verify supported chains
        supported_chains = status["supported_chains"]
        expected_chains = ["ethereum", "polygon", "bsc", "arbitrum"]
        for chain in expected_chains:
            assert chain in supported_chains


class TestTradingBotOrchestrator:
    """Test main trading bot orchestrator"""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing"""
        return {
            "llm_provider": "openai",
            "model_name": "gpt-4-turbo",
            "trading_config": {
                "risk_management": {
                    "max_portfolio_risk": 0.02,
                    "max_position_size": 0.1
                }
            }
        }

    @patch('src.cli.main.ChatBot')
    @patch('src.cli.main.TradingBotOrchestrator.load_config')
    def test_orchestrator_initialization(self, mock_load_config, mock_chatbot, mock_config):
        """Test orchestrator initializes correctly"""
        mock_load_config.return_value = mock_config
        mock_chatbot.return_value = Mock()
        
        orchestrator = TradingBotOrchestrator()
        
        assert orchestrator.config == mock_config
        assert orchestrator.market_agent is not None
        assert orchestrator.risk_agent is not None
        assert orchestrator.execution_agent is not None

    @patch('src.cli.main.Path.exists')
    @patch('builtins.open')
    def test_load_config(self, mock_open, mock_exists):
        """Test configuration loading"""
        mock_exists.return_value = True
        mock_config = {"test": "config"}
        mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(mock_config)
        
        orchestrator = TradingBotOrchestrator()
        config = orchestrator.load_config()
        
        # Config loading is mocked, so we test the method exists and returns dict
        assert isinstance(config, dict)


class TestIntegration:
    """Integration tests for agent interactions"""

    @pytest.fixture
    def mock_llm(self):
        mock = Mock()
        mock.generate = Mock(return_value="Integrated AI response")
        return mock

    @pytest.fixture
    def agents(self, mock_llm):
        """Create all agents for integration testing"""
        return {
            'market': MarketIntelligenceAgent(llm=mock_llm),
            'risk': RiskAssessmentAgent(llm=mock_llm),
            'execution': ExecutionManagerAgent(llm=mock_llm)
        }

    @pytest.mark.asyncio
    async def test_trading_workflow_simulation(self, agents):
        """Test simulated end-to-end trading workflow"""
        market_agent = agents['market']
        risk_agent = agents['risk']
        execution_agent = agents['execution']

        # 1. Market analysis
        chainbase_tool = market_agent.available_tools.get_tool("chainbase_analytics")
        market_data = await chainbase_tool.execute("ethereum", "whale_movements")
        
        assert market_data["chain"] == "ethereum"
        
        # 2. Risk assessment
        contract_tool = risk_agent.available_tools.get_tool("smart_contract_analyzer")
        risk_data = await contract_tool.execute("0x1234567890abcdef", "ethereum")
        
        assert risk_data["chain"] == "ethereum"
        assert 0 <= risk_data["overall_risk_score"] <= 10
        
        # 3. Trade execution (if risk is acceptable)
        if risk_data["overall_risk_score"] < 7:
            dex_tool = execution_agent.available_tools.get_tool("dex_aggregator")
            trade_result = await dex_tool.execute("ETH", "USDC", 1.0, "ethereum")
            
            assert trade_result["status"] == "completed"
            assert "trade_id" in trade_result

    def test_agent_tool_availability(self, agents):
        """Test that all agents have required tools"""
        market_agent = agents['market']
        risk_agent = agents['risk']
        execution_agent = agents['execution']
        
        # Market agent tools
        market_tools = [tool.name for tool in market_agent.available_tools.tools]
        expected_market_tools = ["chainbase_analytics", "price_aggregator", "dex_monitor"]
        for tool in expected_market_tools:
            assert tool in market_tools
        
        # Risk agent tools  
        risk_tools = [tool.name for tool in risk_agent.available_tools.tools]
        expected_risk_tools = ["smart_contract_analyzer", "volatility_predictor", "portfolio_optimizer"]
        for tool in expected_risk_tools:
            assert tool in risk_tools
        
        # Execution agent tools
        execution_tools = [tool.name for tool in execution_agent.available_tools.tools]
        expected_execution_tools = ["dex_aggregator", "position_manager", "gas_optimizer"]
        for tool in expected_execution_tools:
            assert tool in execution_tools


class TestErrorHandling:
    """Test error handling and edge cases"""

    @pytest.mark.asyncio
    async def test_invalid_chain_handling(self):
        """Test handling of invalid blockchain networks"""
        tool = ChainbaseAnalyticsTool()
        
        # Should handle gracefully even with invalid chain
        result = await tool.execute(
            chain="invalid_chain",
            analysis_type="whale_movements"
        )
        
        # Tool should return some result even for invalid inputs
        assert "chain" in result
        assert result["chain"] == "invalid_chain"

    @pytest.mark.asyncio
    async def test_zero_amount_trade(self):
        """Test handling of zero/negative trade amounts"""
        tool = DEXAggregatorTool()
        
        result = await tool.execute(
            token_in="ETH",
            token_out="USDC",
            amount_in=0,  # Zero amount
            chain="ethereum"
        )
        
        # Should handle edge case gracefully
        assert "trade_id" in result

    @pytest.mark.asyncio
    async def test_high_slippage_tolerance(self):
        """Test behavior with unreasonably high slippage"""
        tool = DEXAggregatorTool()
        
        result = await tool.execute(
            token_in="ETH",
            token_out="USDC", 
            amount_in=1.0,
            chain="ethereum",
            slippage_tolerance=0.5  # 50% slippage (unreasonable)
        )
        
        assert result["execution_details"]["slippage_tolerance"] == 0.5
        assert "actual_slippage" in result["execution_details"]


class TestPerformance:
    """Test performance and timing"""

    @pytest.mark.asyncio
    async def test_tool_execution_timing(self):
        """Test that tools complete within reasonable time"""
        import time
        
        tool = PriceAggregatorTool()
        
        start_time = time.time()
        result = await tool.execute("ETH")
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Should complete within 5 seconds (generous limit for testing)
        assert execution_time < 5.0
        assert result is not None

    @pytest.mark.asyncio
    async def test_multiple_concurrent_tools(self):
        """Test concurrent tool execution"""
        tools = [
            ChainbaseAnalyticsTool().execute("ethereum", "whale_movements"),
            PriceAggregatorTool().execute("ETH"),
            SmartContractAnalyzerTool().execute("0x1234567890abcdef", "ethereum")
        ]
        
        start_time = asyncio.get_event_loop().time()
        results = await asyncio.gather(*tools)
        end_time = asyncio.get_event_loop().time()
        
        # Concurrent execution should be faster than sequential
        execution_time = end_time - start_time
        assert execution_time < 10.0  # Should complete within 10 seconds
        assert len(results) == 3
        assert all(result is not None for result in results)


# Utility functions for testing
def create_mock_position(token="ETH", amount=1.0, entry_price=2500.0):
    """Create mock position data for testing"""
    return {
        "token": token,
        "amount": amount,
        "entry_price": entry_price,
        "current_price": entry_price * 1.02,  # 2% gain
        "status": "active"
    }


def create_mock_trade_params(token_in="ETH", token_out="USDC", amount=1.0):
    """Create mock trade parameters for testing"""
    return {
        "token_in": token_in,
        "token_out": token_out,
        "amount": amount,
        "chain": "ethereum",
        "slippage_tolerance": 0.01
    }


# Configuration for pytest
pytest_plugins = []

if __name__ == "__main__":
    pytest.main([__file__, "-v"])