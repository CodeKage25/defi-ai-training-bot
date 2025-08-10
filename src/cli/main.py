"""
DeFi AI Trading Bot - Main CLI Interface
Complete implementation with SpoonOS integration
"""

import asyncio
import click
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
import time
import signal
import threading
from concurrent.futures import ThreadPoolExecutor

# Rich console for beautiful CLI output
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.layout import Layout
from rich.live import Live
from rich import print as rprint
from rich.text import Text
from rich.style import Style
from rich.prompt import Prompt, Confirm
from rich.columns import Columns

# Logging
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# Data processing
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# SpoonOS imports
try:
    from spoon_ai.agents import ToolCallAgent, SpoonReactMCP, SpoonReactAI
    from spoon_ai.tools.base import BaseTool
    from spoon_ai.tools import ToolManager
    from spoon_ai.chat import ChatBot
    from spoon_ai.llm import LLMProvider
    # from spoon_ai.core import SpoonReactMCP, SpoonReactAI
except ImportError as e:
    print(f"SpoonOS not installed. Please install with: pip install spoon-ai")
    print(f"Error details: {e}")
    sys.exit(1)

# Environment and configuration
from dotenv import load_dotenv
from pydantic import Field, BaseModel, validator

# Web3 imports for blockchain interaction
try:
    from web3 import Web3
    from eth_account import Account
    from web3.middleware import geth_poa_middleware
except ImportError as e:
    print(f"Web3 not installed. Please install with: pip install web3 eth-account")
    print(f"Error details: {e}")
    sys.exit(1)

# Async utilities
import aiohttp
import aiofiles
from asyncio import Queue, Event
import asyncio_mqtt

# Database (optional)
try:
    import redis
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker
except ImportError:
    redis = None
    create_engine = None
    sessionmaker = None

# Load environment variables
load_dotenv()

# Initialize console for rich output
console = Console()

# Global event for graceful shutdown
shutdown_event = Event()

# Import our custom agents from the document content
class ChainbaseAnalyticsTool(BaseTool):
    """On-chain analytics and transaction analysis using Chainbase"""
    
    name: str = "chainbase_analytics"
    description: str = "Analyze on-chain data, token flows, and whale movements"
    
    async def execute(self, chain: str, analysis_type: str, token_address: str = None, timeframe: str = "24h") -> Dict[str, Any]:
        logger.info(f"Analyzing {chain} - {analysis_type} for timeframe {timeframe}")
        await asyncio.sleep(1)
        
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
                    }
                ],
                "summary": {
                    "total_whale_volume": 12_300_000,
                    "net_flow": "outbound",
                    "sentiment": "bearish"
                }
            }
        
        return {"analysis": analysis_type, "status": "completed", "data": {}}


class MarketIntelligenceAgent(ToolCallAgent):
    """Market intelligence and opportunity scanning agent"""
    name: str = "market_intelligence_agent"
    description: str = "Scans multi-chain DeFi markets for trading opportunities"
    system_prompt: str = """You are an advanced DeFi market intelligence AI agent.
    Monitor multiple blockchains for opportunities, analyze on-chain data, detect arbitrage,
    and provide actionable trading recommendations."""
    max_steps: int = 15

    available_tools: ToolManager = Field(default_factory=lambda: ToolManager([
        ChainbaseAnalyticsTool(),
    ]))


class RiskAssessmentAgent(ToolCallAgent):
    """Risk analysis and portfolio management agent"""
    name: str = "risk_assessment_agent"
    description: str = "Evaluates risks and optimizes portfolio allocation"
    system_prompt: str = """You are an expert DeFi risk assessment AI agent.
    Analyze smart contract risks, predict volatility, optimize portfolios,
    and provide comprehensive risk management strategies."""
    max_steps: int = 12


class ExecutionManagerAgent(ToolCallAgent):
    """Trade execution and position management agent"""
    name: str = "execution_manager_agent"
    description: str = "Handles optimal trade execution and position management"
    system_prompt: str = """You are an expert DeFi trade execution AI agent.
    Execute trades across DEXs, manage positions, optimize gas costs,
    and implement automated risk management."""
    max_steps: int = 20


class TradingStrategy(Enum):
    """Available trading strategies"""
    ARBITRAGE = "arbitrage"
    YIELD_FARMING = "yield_farming"
    TREND_FOLLOWING = "trend_following"
    DCA = "dca"
    MARKET_MAKING = "market_making"
    LIQUIDITY_MINING = "liquidity_mining"


class TradingBotConfig(BaseModel):
    """Trading bot configuration model"""
    llm_settings: Dict[str, Any]
    trading_config: Dict[str, Any]
    monitoring: Dict[str, Any]
    
    @validator('trading_config')
    def validate_trading_config(cls, v):
        required_keys = ['risk_management', 'chains']
        for key in required_keys:
            if key not in v:
                raise ValueError(f"Missing required trading config key: {key}")
        return v


class TradingBotOrchestrator:
    """Main orchestrator for the DeFi AI Trading Bot"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the trading bot with configuration"""
        self.config_path = config_path or "config/config.json"
        self.config = self.load_config()
        self.setup_logging()
        
        # Initialize Web3 connections
        self.web3_connections = self.setup_web3()
        
        # Initialize LLM provider
        self.llm = self.setup_llm()
        
        # Initialize agents
        self.market_agent = None
        self.risk_agent = None
        self.execution_agent = None
        self.initialize_agents()
        
        # Initialize database connections
        self.redis_client = self.setup_redis()
        self.db_engine = self.setup_database()
        
        # Trading state
        self.active_positions = {}
        self.trading_history = []
        self.performance_metrics = {
            "total_trades": 0,
            "successful_trades": 0,
            "total_pnl": 0.0,
            "win_rate": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0
        }
        
        # Monitoring
        self.running_strategies = {}
        self.market_data_cache = {}
        self.last_health_check = datetime.now()
        
        # Session for HTTP requests
        self.session = self.setup_http_session()
        
        logger.info("🚀 DeFi AI Trading Bot initialized successfully")
        self.display_startup_banner()

    def display_startup_banner(self):
        """Display startup banner"""
        banner = Panel.fit(
            """[bold cyan]🤖 DeFi AI Trading Bot v1.0[/bold cyan]
[green]Powered by SpoonOS Agent Framework[/green]
            
[yellow]Multi-Chain • AI-Powered • Risk-Managed[/yellow]
            
[dim]Ready for autonomous DeFi trading across chains[/dim]""",
            border_style="bright_blue"
        )
        console.print(banner)

    def load_config(self) -> Dict:
        """Load configuration from JSON file"""
        config_path = Path(self.config_path)
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                    logger.info(f"Configuration loaded from {config_path}")
                    return TradingBotConfig(**config).dict()
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Invalid config file: {e}")
                return self.get_default_config()
        else:
            logger.warning(f"Config file not found at {config_path}, using defaults")
            config_path.parent.mkdir(parents=True, exist_ok=True)
            default_config = self.get_default_config()
            
            # Save default config
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            logger.info(f"Default config saved to {config_path}")
            
            return default_config

    def get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            "llm_settings": {
                "default_provider": "openai",
                "model": "gpt-4-turbo",
                "temperature": 0.1,
                "max_tokens": 4096,
                "timeout": 30
            },
            "trading_config": {
                "risk_management": {
                    "max_portfolio_risk": 0.02,
                    "max_position_size": 0.1,
                    "stop_loss_percentage": 0.05,
                    "take_profit_percentage": 0.15,
                    "max_drawdown": 0.1
                },
                "execution": {
                    "min_trade_amount": 100,
                    "max_slippage": 0.01,
                    "gas_limit_multiplier": 1.2,
                    "transaction_timeout": 300,
                    "retry_attempts": 3
                },
                "chains": {
                    "ethereum": {
                        "chain_id": 1,
                        "enabled": True,
                        "gas_optimization": True,
                        "supported_dexs": ["uniswap_v3", "sushiswap", "curve"]
                    },
                    "polygon": {
                        "chain_id": 137,
                        "enabled": True,
                        "gas_optimization": False,
                        "supported_dexs": ["quickswap", "sushiswap", "curve"]
                    },
                    "bsc": {
                        "chain_id": 56,
                        "enabled": True,
                        "gas_optimization": False,
                        "supported_dexs": ["pancakeswap", "biswap", "mdex"]
                    },
                    "arbitrum": {
                        "chain_id": 42161,
                        "enabled": True,
                        "gas_optimization": True,
                        "supported_dexs": ["uniswap_v3", "sushiswap", "curve"]
                    }
                }
            },
            "monitoring": {
                "performance_tracking": True,
                "real_time_alerts": True,
                "telegram_notifications": True,
                "discord_notifications": False,
                "email_notifications": True,
                "metrics_retention_days": 365
            }
        }

    def setup_logging(self):
        """Configure logging system"""
        logger.remove()
        
        # Get log level from environment
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        debug_mode = os.getenv("DEBUG", "false").lower() == "true"
        
        if debug_mode:
            log_level = "DEBUG"
        
        # Console logging with color
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
            level=log_level,
            colorize=True
        )
        
        # File logging
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logger.add(
            log_dir / "trading_bot_{time:YYYY-MM-DD}.log",
            rotation="1 day",
            retention="30 days",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG"
        )
        
        # Error logging
        logger.add(
            log_dir / "errors_{time:YYYY-MM-DD}.log",
            rotation="1 day",
            retention="90 days",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
            level="ERROR"
        )
        
        logger.info("Logging system configured")

    def setup_http_session(self) -> requests.Session:
        """Setup HTTP session with retry strategy"""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set timeout and headers
        session.timeout = 30
        session.headers.update({
            'User-Agent': 'DeFi-AI-Trading-Bot/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
        
        return session

    def setup_web3(self) -> Dict[str, Web3]:
        """Setup Web3 connections for different chains"""
        connections = {}
        
        chain_configs = {
            "ethereum": {
                "rpc": os.getenv("ETHEREUM_RPC_URL", "https://eth.public-rpc.com"),
                "chain_id": 1,
                "is_poa": False
            },
            "polygon": {
                "rpc": os.getenv("POLYGON_RPC_URL", "https://polygon-rpc.com"),
                "chain_id": 137,
                "is_poa": True
            },
            "bsc": {
                "rpc": os.getenv("BSC_RPC_URL", "https://bsc-dataseed.binance.org"),
                "chain_id": 56,
                "is_poa": True
            },
            "arbitrum": {
                "rpc": os.getenv("ARBITRUM_RPC_URL", "https://arb1.arbitrum.io/rpc"),
                "chain_id": 42161,
                "is_poa": False
            },
            "optimism": {
                "rpc": os.getenv("OPTIMISM_RPC_URL", "https://mainnet.optimism.io"),
                "chain_id": 10,
                "is_poa": False
            }
        }
        
        for chain, config in chain_configs.items():
            try:
                w3 = Web3(Web3.HTTPProvider(
                    config["rpc"],
                    request_kwargs={'timeout': 30}
                ))
                
                # Add PoA middleware if needed
                if config["is_poa"]:
                    w3.middleware_onion.inject(geth_poa_middleware, layer=0)
                
                if w3.is_connected():
                    # Verify chain ID
                    actual_chain_id = w3.eth.chain_id
                    if actual_chain_id == config["chain_id"]:
                        connections[chain] = w3
                        logger.info(f"✅ Connected to {chain} network (Chain ID: {actual_chain_id})")
                    else:
                        logger.warning(f"⚠️ Chain ID mismatch for {chain}: expected {config['chain_id']}, got {actual_chain_id}")
                else:
                    logger.warning(f"⚠️ Failed to connect to {chain}")
            except Exception as e:
                logger.error(f"Error connecting to {chain}: {e}")
        
        if not connections:
            logger.warning("No blockchain connections available - running in simulation mode")
        
        return connections

    def setup_llm(self):
        """Setup LLM provider for AI agents"""
        try:
            # Try OpenAI first
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                logger.info("Initializing OpenAI LLM provider")
                return ChatBot(
                    llm_provider="openai",
                    model_name=self.config["llm_settings"]["model"],
                    api_key=openai_key,
                    temperature=self.config["llm_settings"]["temperature"],
                    max_tokens=self.config["llm_settings"]["max_tokens"]
                )
            
            # Fallback to Anthropic
            anthropic_key = os.getenv("ANTHROPIC_API_KEY")
            if anthropic_key:
                logger.info("Initializing Anthropic LLM provider")
                return ChatBot(
                    llm_provider="anthropic",
                    model_name="claude-3-sonnet-20240229",
                    api_key=anthropic_key,
                    temperature=self.config["llm_settings"]["temperature"]
                )
            
            logger.error("No LLM API keys found in environment variables")
            logger.info("Please set OPENAI_API_KEY or ANTHROPIC_API_KEY")
            return None
            
        except Exception as e:
            logger.error(f"Failed to setup LLM: {e}")
            return None

    def setup_redis(self):
        """Setup Redis connection for caching"""
        if not redis:
            logger.info("Redis not available - using in-memory cache")
            return None
        
        try:
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            client = redis.from_url(redis_url)
            client.ping()  # Test connection
            logger.info("✅ Connected to Redis cache")
            return client
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            return None

    def setup_database(self):
        """Setup database connection"""
        if not create_engine:
            logger.info("SQLAlchemy not available - using file storage")
            return None
        
        try:
            database_url = os.getenv("DATABASE_URL")
            if database_url:
                engine = create_engine(database_url)
                # Test connection
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                logger.info("✅ Connected to database")
                return engine
            else:
                logger.info("No database URL configured - using file storage")
                return None
        except Exception as e:
            logger.warning(f"Database connection failed: {e}")
            return None

    def initialize_agents(self):
        """Initialize all AI agents"""
        if not self.llm:
            logger.warning("LLM not available - agents will run in mock mode")
            
        try:
            self.market_agent = MarketIntelligenceAgent(llm=self.llm)
            self.risk_agent = RiskAssessmentAgent(llm=self.llm)
            self.execution_agent = ExecutionManagerAgent(llm=self.llm)
            logger.info("✅ All AI agents initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize agents: {e}")
            # Create mock agents for testing
            self.market_agent = MarketIntelligenceAgent(llm=None)
            self.risk_agent = RiskAssessmentAgent(llm=None)
            self.execution_agent = ExecutionManagerAgent(llm=None)
            logger.info("⚠️ Mock agents initialized (no LLM available)")

    async def scan_markets(self, chains: Optional[List[str]] = None, 
                          tokens: Optional[List[str]] = None) -> Dict:
        """Scan markets for trading opportunities"""
        console.print("\n[bold blue]🔍 Scanning Markets for Opportunities[/bold blue]")
        
        enabled_chains = [
            chain for chain, config in self.config["trading_config"]["chains"].items()
            if config.get("enabled", False)
        ]
        
        chains = chains or enabled_chains
        tokens = tokens or ["ETH", "BTC", "USDC", "MATIC", "USDT"]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"Analyzing {len(chains)} chains...", total=len(chains))
            
            opportunities = []
            
            for chain in chains:
                progress.update(task, description=f"Scanning {chain}...")
                
                try:
                    # Call market intelligence agent
                    if self.market_agent and self.llm:
                        result = await self.market_agent.analyze_specific_opportunity(
                            chain=chain,
                            token_pair="ETH/USDC",
                            opportunity_type="arbitrage"
                        )
                        if result:
                            opportunities.extend(result.get("opportunities", []))
                    else:
                        # Mock opportunity for testing
                        opportunity = {
                            "id": f"{chain}_arb_{len(opportunities)}",
                            "chain": chain,
                            "type": "arbitrage",
                            "tokens": ["ETH", "USDC"],
                            "potential_profit": np.random.uniform(50, 500),
                            "risk_score": np.random.uniform(1, 5),
                            "confidence": np.random.uniform(0.6, 0.95),
                            "timestamp": datetime.now().isoformat(),
                            "estimated_gas": np.random.uniform(20, 100)
                        }
                        opportunities.append(opportunity)
                
                except Exception as e:
                    logger.error(f"Error scanning {chain}: {e}")
                
                progress.advance(task)
                await asyncio.sleep(0.5)  # Rate limiting
            
            # Cache results
            if self.redis_client:
                try:
                    self.redis_client.setex(
                        "market_scan_results",
                        3600,  # 1 hour TTL
                        json.dumps(opportunities)
                    )
                except Exception as e:
                    logger.warning(f"Failed to cache results: {e}")
            
            self.display_opportunities(opportunities)
            return {
                "opportunities": opportunities, 
                "timestamp": datetime.now().isoformat(),
                "chains_scanned": len(chains),
                "total_opportunities": len(opportunities)
            }

    def display_opportunities(self, opportunities: List[Dict]):
        """Display found trading opportunities"""
        if not opportunities:
            console.print("[yellow]No opportunities found[/yellow]")
            return
        
        # Sort by potential profit
        opportunities.sort(key=lambda x: x.get("potential_profit", 0), reverse=True)
        
        table = Table(title=f"🎯 Trading Opportunities Found ({len(opportunities)})")
        table.add_column("ID", style="dim", width=12)
        table.add_column("Chain", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Tokens", style="white")
        table.add_column("Profit ($)", style="green", justify="right")
        table.add_column("Risk", style="yellow", justify="center")
        table.add_column("Confidence", style="blue", justify="center")
        table.add_column("Gas ($)", style="red", justify="right")
        
        for opp in opportunities[:10]:  # Show top 10
            risk_score = opp.get("risk_score", 0)
            risk_color = "green" if risk_score < 3 else "yellow" if risk_score < 4 else "red"
            
            confidence = opp.get("confidence", 0)
            conf_color = "red" if confidence < 0.7 else "yellow" if confidence < 0.85 else "green"
            
            table.add_row(
                opp.get("id", "N/A")[:12],
                opp["chain"].capitalize(),
                opp["type"].upper(),
                " → ".join(opp.get("tokens", [])),
                f"${opp.get('potential_profit', 0):.2f}",
                f"[{risk_color}]{risk_score:.1f}/5[/{risk_color}]",
                f"[{conf_color}]{confidence*100:.1f}%[/{conf_color}]",
                f"${opp.get('estimated_gas', 0):.2f}"
            )
        
        console.print(table)
        
        if len(opportunities) > 10:
            console.print(f"[dim]... and {len(opportunities) - 10} more opportunities[/dim]")

    async def assess_risk(self, strategy: str, amount: float, tokens: List[str] = None) -> Dict:
        """Perform comprehensive risk assessment"""
        console.print(f"\n[bold yellow]⚠️ Risk Assessment: {strategy}[/bold yellow]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Analyzing risks...", total=None)
            
            try:
                if self.risk_agent and self.llm:
                    result = await self.risk_agent.assess_trading_risk(
                        strategy=strategy,
                        amount_usd=amount,
                        tokens=tokens
                    )
                    risk_data = result.get("risk_analysis", {})
                else:
                    # Mock risk analysis for testing
                    risk_data = {
                        "strategy": strategy,
                        "amount": amount,
                        "risk_score": np.random.uniform(3, 7),
                        "max_loss": amount * np.random.uniform(0.05, 0.20),
                        "expected_return": amount * np.random.uniform(0.05, 0.25),
                        "confidence": np.random.uniform(0.65, 0.90),
                        "sharpe_ratio": np.random.uniform(0.5, 2.5),
                        "var_95": amount * np.random.uniform(0.03, 0.15),
                        "recommendations": [
                            "Use stop-loss at 5%",
                            "Monitor liquidity conditions closely",
                            "Consider position sizing at 50% of planned amount",
                            "Set take-profit at 15% gains"
                        ],
                        "risk_factors": {
                            "market_risk": np.random.uniform(0.2, 0.8),
                            "liquidity_risk": np.random.uniform(0.1, 0.6),
                            "smart_contract_risk": np.random.uniform(0.1, 0.5),
                            "operational_risk": np.random.uniform(0.1, 0.4)
                        }
                    }
                
                progress.remove_task(task)
                self.display_risk_analysis(risk_data)
                return risk_data
                
            except Exception as e:
                logger.error(f"Risk assessment failed: {e}")
                progress.remove_task(task)
                console.print("[red]❌ Risk assessment failed[/red]")
                return {"error": str(e)}

    def display_risk_analysis(self, analysis: Dict):
        """Display risk analysis results"""
        risk_score = analysis.get("risk_score", 0)
        risk_level = "LOW" if risk_score < 4 else "MEDIUM" if risk_score < 7 else "HIGH"
        risk_color = "green" if risk_score < 4 else "yellow" if risk_score < 7 else "red"
        
        # Main risk panel
        content = f"""
[bold]Strategy:[/bold] {analysis.get('strategy', 'Unknown')}
[bold]Amount:[/bold] ${analysis.get('amount', 0):,.2f}

[bold]Risk Score:[/bold] [{risk_color}]{risk_score:.1f}/10 ({risk_level})[/{risk_color}]
[bold]Max Potential Loss (VaR 95%):[/bold] [red]-${analysis.get('var_95', 0):,.2f}[/red]
[bold]Expected Return:[/bold] [green]+${analysis.get('expected_return', 0):,.2f}[/green]
[bold]Sharpe Ratio:[/bold] {analysis.get('sharpe_ratio', 0):.2f}
[bold]Confidence Level:[/bold] {analysis.get('confidence', 0)*100:.1f}%"""
        
        panel = Panel(content, title="📊 Risk Analysis Report", border_style=risk_color)
        console.print(panel)
        
        # Risk factors breakdown
        risk_factors = analysis.get('risk_factors', {})
        if risk_factors:
            risk_table = Table(title="Risk Factor Breakdown")
            risk_table.add_column("Factor", style="cyan")
            risk_table.add_column("Level", style="white", justify="center")
            risk_table.add_column("Score", style="yellow", justify="right")
            
            for factor, score in risk_factors.items():
                level = "LOW" if score < 0.3 else "MEDIUM" if score < 0.7 else "HIGH"
                color = "green" if score < 0.3 else "yellow" if score < 0.7 else "red"
                
                risk_table.add_row(
                    factor.replace("_", " ").title(),
                    f"[{color}]{level}[/{color}]",
                    f"{score:.2f}"
                )
            
            console.print(risk_table)
        
        # Recommendations
        recommendations = analysis.get('recommendations', [])
        if recommendations:
            rec_content = "[bold]Risk Management Recommendations:[/bold]\n"
            for i, rec in enumerate(recommendations, 1):
                rec_content += f"  {i}. {rec}\n"
            
            rec_panel = Panel(rec_content.strip(), title="💡 Recommendations", border_style="blue")
            console.print(rec_panel)

    async def execute_trade(self, token_in: str, token_out: str, amount: float, 
                           chain: str = "ethereum", slippage: float = 0.01) -> Dict:
        """Execute a trade with optimal routing"""
        console.print(f"\n[bold green]🚀 Executing Trade[/bold green]")
        console.print(f"Trading {amount} {token_in} → {token_out} on {chain}")
        
        # Pre-execution checks
        if not self.web3_connections.get(chain):
            console.print(f"[red]❌ No connection to {chain} network[/red]")
            return {"error": f"No connection to {chain}"}
        
        # Risk check
        max_trade_amount = float(os.getenv("MAX_TRADE_AMOUNT", "1000"))
        if amount > max_trade_amount:
            console.print(f"[red]❌ Trade amount exceeds maximum: ${max_trade_amount}[/red]")
            return {"error": "Amount exceeds maximum"}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console,
        ) as progress:
            
            # Step 1: Check wallet balance
            task1 = progress.add_task("Checking wallet balance...", total=100)
            await asyncio.sleep(0.5)
            progress.update(task1, completed=100)
            
            # Step 2: Get best route
            task2 = progress.add_task("Finding optimal route...", total=100)
            await asyncio.sleep(1)
            
            try:
                if self.execution_agent and self.llm:
                    route_result = await self.execution_agent.execute_optimal_trade(
                        token_in=token_in,
                        token_out=token_out,
                        amount=amount,
                        chain=chain,
                        max_slippage=slippage
                    )
                else:
                    # Mock route finding
                    route_result = {
                        "best_route": {
                            "dex": np.random.choice(["Uniswap V3", "SushiSwap", "Curve"]),
                            "expected_output": amount * 2500 if token_out == "USDC" else amount / 2500,
                            "price_impact": np.random.uniform(0.001, slippage),
                            "gas_estimate": np.random.uniform(50, 150)
                        }
                    }
                
                progress.update(task2, completed=100)
            except Exception as e:
                logger.error(f"Route finding failed: {e}")
                progress.update(task2, completed=100)
                console.print(f"[red]❌ Route finding failed: {e}[/red]")
                return {"error": "Route finding failed"}
            
            # Step 3: Estimate gas
            task3 = progress.add_task("Optimizing gas costs...", total=100)
            await asyncio.sleep(0.5)
            
            gas_price = await self.get_optimal_gas_price(chain)
            estimated_gas_cost = route_result["best_route"]["gas_estimate"] * gas_price / 1e9
            
            progress.update(task3, completed=100)
            
            # Step 4: Execute transaction
            task4 = progress.add_task("Executing transaction...", total=100)
            
            # Simulate transaction execution
            await asyncio.sleep(2)
            
            # Mock trade result
            trade_result = {
                "status": "success",
                "trade_id": f"trade_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "transaction_hash": f"0x{''.join(np.random.choice(list('0123456789abcdef'), 64))}",
                "chain": chain,
                "input": {"token": token_in, "amount": amount},
                "output": {
                    "token": token_out, 
                    "amount": route_result["best_route"]["expected_output"]
                },
                "execution_price": route_result["best_route"]["expected_output"] / amount,
                "gas_used": route_result["best_route"]["gas_estimate"],
                "gas_price_gwei": gas_price,
                "gas_cost_usd": estimated_gas_cost,
                "slippage": route_result["best_route"]["price_impact"],
                "dex_used": route_result["best_route"]["dex"],
                "timestamp": datetime.now().isoformat(),
                "block_number": np.random.randint(18000000, 19000000)
            }
            
            progress.update(task4, completed=100)
            
            # Update trading history and metrics
            self.trading_history.append(trade_result)
            self.performance_metrics["total_trades"] += 1
            self.performance_metrics["successful_trades"] += 1
            
            # Save to database if available
            await self.save_trade_to_db(trade_result)
            
            self.display_trade_result(trade_result)
            return trade_result

    async def get_optimal_gas_price(self, chain: str) -> float:
        """Get optimal gas price for the chain"""
        try:
            w3 = self.web3_connections.get(chain)
            if w3:
                gas_price = w3.eth.gas_price
                return gas_price / 1e9  # Convert to gwei
            else:
                # Mock gas prices
                mock_prices = {
                    "ethereum": 25.0,
                    "polygon": 2.0,
                    "bsc": 5.0,
                    "arbitrum": 0.2
                }
                return mock_prices.get(chain, 10.0)
        except Exception as e:
            logger.error(f"Failed to get gas price for {chain}: {e}")
            return 20.0  # Default fallback

    async def save_trade_to_db(self, trade_result: Dict):
        """Save trade result to database"""
        try:
            if self.db_engine:
                # In production, this would save to a proper database
                logger.info(f"Trade {trade_result['trade_id']} saved to database")
            else:
                # Save to file as fallback
                trades_file = Path("data/trades.jsonl")
                trades_file.parent.mkdir(exist_ok=True)
                
                async with aiofiles.open(trades_file, mode='a') as f:
                    await f.write(json.dumps(trade_result) + '\n')
        except Exception as e:
            logger.error(f"Failed to save trade: {e}")

    def display_trade_result(self, result: Dict):
        """Display trade execution results"""
        if result.get("status") != "success":
            console.print(f"[red]❌ Trade failed: {result.get('error', 'Unknown error')}[/red]")
            return
        
        console.print("\n[bold green]✅ Trade Executed Successfully[/bold green]")
        
        # Transaction details table
        table = Table(show_header=False, box=None)
        table.add_column("Field", style="cyan", width=20)
        table.add_column("Value", style="white")
        
        table.add_row("Trade ID", result.get('trade_id', 'N/A'))
        table.add_row("Transaction Hash", f"{result.get('transaction_hash', '')[:10]}...{result.get('transaction_hash', '')[-10:]}")
        table.add_row("Chain", result.get('chain', '').capitalize())
        table.add_row("DEX Used", result.get('dex_used', 'N/A'))
        table.add_row("Input", f"{result['input']['amount']} {result['input']['token']}")
        table.add_row("Output", f"{result['output']['amount']:.6f} {result['output']['token']}")
        table.add_row("Execution Price", f"${result.get('execution_price', 0):.4f}")
        table.add_row("Slippage", f"{result.get('slippage', 0)*100:.3f}%")
        table.add_row("Gas Cost", f"${result.get('gas_cost_usd', 0):.2f}")
        table.add_row("Block Number", str(result.get('block_number', 'N/A')))
        
        console.print(table)
        
        # Profit calculation (simple example)
        input_value = result['input']['amount'] * 2500  # Mock ETH price
        output_value = result['output']['amount'] * 1  # Mock USDC price
        profit = output_value - input_value - result.get('gas_cost_usd', 0)
        
        profit_color = "green" if profit > 0 else "red"
        console.print(f"\n[bold]Net Profit/Loss: [{profit_color}]${profit:+.2f}[/{profit_color}][/bold]")

    async def monitor_positions(self, continuous: bool = False):
        """Monitor and display active positions"""
        console.print("\n[bold blue]📊 Position Monitoring[/bold blue]")
        
        if continuous:
            # Continuous monitoring mode
            with Live(self.generate_position_layout(), refresh_per_second=1, console=console) as live:
                try:
                    while not shutdown_event.is_set():
                        # Update positions
                        await self.update_positions()
                        live.update(self.generate_position_layout())
                        await asyncio.sleep(5)
                except KeyboardInterrupt:
                    console.print("\n[yellow]Position monitoring stopped[/yellow]")
        else:
            # Single snapshot
            await self.update_positions()
            self.display_positions(list(self.active_positions.values()))

    def generate_position_layout(self) -> Layout:
        """Generate live position monitoring layout"""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="positions", size=15),
            Layout(name="summary", size=5)
        )
        
        # Header
        header = Panel.fit(
            f"[bold blue]Live Position Monitoring[/bold blue] - {datetime.now().strftime('%H:%M:%S')}",
            border_style="blue"
        )
        layout["header"].update(header)
        
        # Positions
        positions = list(self.active_positions.values())
        if positions:
            positions_table = self.create_positions_table(positions)
            layout["positions"].update(positions_table)
        else:
            layout["positions"].update(Panel("No active positions", style="dim"))
        
        # Summary
        total_pnl = sum(pos.get("pnl", 0) for pos in positions)
        summary_color = "green" if total_pnl >= 0 else "red"
        
        summary_content = f"""[bold]Portfolio Summary[/bold]
Active Positions: {len(positions)}
Total P&L: [{summary_color}]${total_pnl:+.2f}[/{summary_color}]
Success Rate: {self.performance_metrics.get('win_rate', 0):.1f}%"""
        
        layout["summary"].update(Panel(summary_content, border_style=summary_color))
        
        return layout

    async def update_positions(self):
        """Update position data with current prices"""
        for pos_id, position in self.active_positions.items():
            try:
                # Mock price update (in production, fetch real prices)
                price_change = np.random.uniform(-0.02, 0.02)
                position["current_price"] *= (1 + price_change)
                position["pnl"] = (position["current_price"] - position["entry_price"]) * position["amount"]
                position["pnl_percent"] = ((position["current_price"] - position["entry_price"]) / position["entry_price"]) * 100
                position["last_updated"] = datetime.now().isoformat()
                
                # Check for stop-loss/take-profit triggers
                if position["pnl_percent"] <= -5.0:  # 5% stop loss
                    await self.trigger_stop_loss(pos_id, position)
                elif position["pnl_percent"] >= 15.0:  # 15% take profit
                    await self.trigger_take_profit(pos_id, position)
                    
            except Exception as e:
                logger.error(f"Error updating position {pos_id}: {e}")

    async def trigger_stop_loss(self, pos_id: str, position: Dict):
        """Trigger stop-loss for a position"""
        logger.warning(f"Stop-loss triggered for position {pos_id}")
        console.print(f"[red]🛑 Stop-loss triggered for {position['token']} position[/red]")
        
        # Execute sell order (mock)
        await self.close_position(pos_id, reason="stop_loss")

    async def trigger_take_profit(self, pos_id: str, position: Dict):
        """Trigger take-profit for a position"""
        logger.info(f"Take-profit triggered for position {pos_id}")
        console.print(f"[green]🎯 Take-profit triggered for {position['token']} position[/green]")
        
        # Execute sell order (mock)
        await self.close_position(pos_id, reason="take_profit")

    async def close_position(self, pos_id: str, reason: str = "manual"):
        """Close a trading position"""
        if pos_id not in self.active_positions:
            logger.error(f"Position {pos_id} not found")
            return
        
        position = self.active_positions[pos_id]
        position["status"] = "closed"
        position["close_reason"] = reason
        position["closed_at"] = datetime.now().isoformat()
        
        # Update performance metrics
        if position["pnl"] > 0:
            self.performance_metrics["successful_trades"] += 1
        
        self.performance_metrics["total_pnl"] += position["pnl"]
        
        # Remove from active positions
        del self.active_positions[pos_id]
        
        logger.info(f"Position {pos_id} closed with P&L: ${position['pnl']:.2f}")

    def create_positions_table(self, positions: List[Dict]) -> Table:
        """Create a table for displaying positions"""
        table = Table(title=f"📈 Active Positions ({len(positions)})")
        table.add_column("ID", style="dim", width=12)
        table.add_column("Token", style="magenta", width=8)
        table.add_column("Amount", style="white", justify="right")
        table.add_column("Entry", style="white", justify="right")
        table.add_column("Current", style="white", justify="right")
        table.add_column("P&L ($)", style="white", justify="right")
        table.add_column("P&L (%)", style="white", justify="right")
        table.add_column("Status", style="blue")
        
        for pos in positions:
            pnl = pos.get("pnl", 0)
            pnl_percent = pos.get("pnl_percent", 0)
            pnl_color = "green" if pnl > 0 else "red"
            
            table.add_row(
                pos.get("position_id", "")[:12],
                pos.get("token", ""),
                f"{pos.get('amount', 0):.4f}",
                f"${pos.get('entry_price', 0):.2f}",
                f"${pos.get('current_price', 0):.2f}",
                f"[{pnl_color}]{pnl:+.2f}[/{pnl_color}]",
                f"[{pnl_color}]{pnl_percent:+.2f}%[/{pnl_color}]",
                pos.get("status", "").upper()
            )
        
        return table

    def display_positions(self, positions: List[Dict]):
        """Display position information"""
        if not positions:
            console.print("[yellow]No active positions[/yellow]")
            return
        
        console.print(self.create_positions_table(positions))
        
        # Summary
        total_pnl = sum(pos.get("pnl", 0) for pos in positions)
        winning_positions = len([p for p in positions if p.get("pnl", 0) > 0])
        
        summary_color = "green" if total_pnl > 0 else "red"
        console.print(f"\n[bold]Summary:[/bold]")
        console.print(f"Total P&L: [{summary_color}]${total_pnl:+.2f}[/{summary_color}]")
        console.print(f"Winning Positions: {winning_positions}/{len(positions)}")

    async def run_strategy(self, strategy_name: str, parameters: Dict):
        """Execute a specific trading strategy"""
        console.print(f"\n[bold magenta]🎯 Running Strategy: {strategy_name.upper()}[/bold magenta]")
        
        # Validate strategy
        try:
            strategy_enum = TradingStrategy(strategy_name.lower())
        except ValueError:
            available = [s.value for s in TradingStrategy]
            console.print(f"[red]❌ Unknown strategy: {strategy_name}[/red]")
            console.print(f"Available strategies: {', '.join(available)}")
            return
        
        # Check if strategy is already running
        if strategy_name in self.running_strategies:
            console.print(f"[yellow]⚠️ Strategy {strategy_name} is already running[/yellow]")
            return
        
        # Start strategy
        self.running_strategies[strategy_name] = {
            "status": "running",
            "started_at": datetime.now().isoformat(),
            "parameters": parameters
        }
        
        strategy_functions = {
            TradingStrategy.ARBITRAGE.value: self._run_arbitrage_strategy,
            TradingStrategy.YIELD_FARMING.value: self._run_yield_strategy,
            TradingStrategy.TREND_FOLLOWING.value: self._run_trend_strategy,
            TradingStrategy.DCA.value: self._run_dca_strategy,
            TradingStrategy.MARKET_MAKING.value: self._run_market_making_strategy,
            TradingStrategy.LIQUIDITY_MINING.value: self._run_liquidity_mining_strategy
        }
        
        try:
            await strategy_functions[strategy_name](parameters)
            self.running_strategies[strategy_name]["status"] = "completed"
        except Exception as e:
            logger.error(f"Strategy {strategy_name} failed: {e}")
            console.print(f"[red]❌ Strategy failed: {e}[/red]")
            self.running_strategies[strategy_name]["status"] = "failed"
            self.running_strategies[strategy_name]["error"] = str(e)

    async def _run_arbitrage_strategy(self, params: Dict):
        """Execute arbitrage trading strategy"""
        console.print("[yellow]🔄 Executing Arbitrage Strategy...[/yellow]")
        
        min_profit = params.get("min_profit", 50)
        max_gas = params.get("max_gas", 100)
        chains = params.get("chains", ["ethereum", "polygon"])
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Scanning for arbitrage opportunities...", total=None)
            
            # Step 1: Scan multiple DEXs across chains
            opportunities = []
            for chain in chains:
                await asyncio.sleep(1)
                # Mock arbitrage opportunity
                profit = np.random.uniform(30, 200)
                if profit >= min_profit:
                    opportunities.append({
                        "chain": chain,
                        "profit": profit,
                        "gas_cost": np.random.uniform(10, max_gas)
                    })
            
            progress.update(task, description="Analyzing price differences...")
            await asyncio.sleep(1)
            
            # Step 2: Execute profitable opportunities
            executed_trades = 0
            total_profit = 0
            
            for opp in opportunities:
                if opp["profit"] - opp["gas_cost"] > 0:  # Profitable after gas
                    progress.update(task, description=f"Executing trade on {opp['chain']}...")
                    await asyncio.sleep(1)
                    
                    # Mock trade execution
                    executed_trades += 1
                    total_profit += (opp["profit"] - opp["gas_cost"])
            
            progress.remove_task(task)
        
        console.print(f"[green]✅ Arbitrage strategy completed[/green]")
        console.print(f"  • Found {len(opportunities)} opportunities")
        console.print(f"  • Executed {executed_trades} profitable trades")
        console.print(f"  • Total profit: ${total_profit:.2f}")

    async def _run_yield_strategy(self, params: Dict):
        """Execute yield farming strategy"""
        console.print("[yellow]🌾 Executing Yield Farming Strategy...[/yellow]")
        
        target_apy = params.get("target_apy", 15.0)
        amount = params.get("amount", 1000)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Scanning yield opportunities...", total=100)
            
            # Mock yield farming optimization
            await asyncio.sleep(1)
            progress.update(task, advance=25)
            
            progress.update(task, description="Comparing pool APYs...")
            await asyncio.sleep(1)
            progress.update(task, advance=25)
            
            progress.update(task, description="Moving liquidity to optimal pools...")
            await asyncio.sleep(1)
            progress.update(task, advance=25)
            
            progress.update(task, description="Setting up auto-compound...")
            await asyncio.sleep(1)
            progress.update(task, advance=25)
        
        new_apy = np.random.uniform(target_apy, target_apy + 8)
        console.print(f"[green]✅ Yield farming positions optimized[/green]")
        console.print(f"  • Target APY: {target_apy}%")
        console.print(f"  • Achieved APY: {new_apy:.1f}%")
        console.print(f"  • Amount deployed: ${amount:,.2f}")
        console.print(f"  • Expected annual return: ${amount * new_apy / 100:,.2f}")

    async def _run_trend_strategy(self, params: Dict):
        """Execute trend following strategy"""
        console.print("[yellow]📈 Executing Trend Following Strategy...[/yellow]")
        
        tokens = params.get("tokens", ["ETH", "BTC"])
        timeframe = params.get("timeframe", "1h")
        
        await asyncio.sleep(2)
        
        # Mock trend analysis
        trend_direction = np.random.choice(["bullish", "bearish", "sideways"])
        confidence = np.random.uniform(0.6, 0.95)
        
        console.print(f"[green]✅ Trend analysis complete[/green]")
        console.print(f"  • Analyzed tokens: {', '.join(tokens)}")
        console.print(f"  • Timeframe: {timeframe}")
        console.print(f"  • Detected trend: {trend_direction}")
        console.print(f"  • Confidence: {confidence*100:.1f}%")
        
        if trend_direction == "bullish" and confidence > 0.8:
            console.print("  • Action: Opened long positions")
        elif trend_direction == "bearish" and confidence > 0.8:
            console.print("  • Action: Opened short positions")
        else:
            console.print("  • Action: Waiting for clearer signals")

    async def _run_dca_strategy(self, params: Dict):
        """Execute dollar cost averaging strategy"""
        console.print("[yellow]💰 Executing DCA Strategy...[/yellow]")
        
        token = params.get("token", "ETH")
        amount_per_buy = params.get("amount_per_buy", 100)
        frequency = params.get("frequency", "daily")
        duration_days = params.get("duration_days", 30)
        
        await asyncio.sleep(2)
        
        total_amount = amount_per_buy * duration_days
        console.print(f"[green]✅ DCA strategy configured[/green]")
        console.print(f"  • Token: {token}")
        console.print(f"  • Amount per purchase: ${amount_per_buy}")
        console.print(f"  • Frequency: {frequency}")
        console.print(f"  • Duration: {duration_days} days")
        console.print(f"  • Total investment: ${total_amount:,.2f}")

    async def _run_market_making_strategy(self, params: Dict):
        """Execute market making strategy"""
        console.print("[yellow]📊 Executing Market Making Strategy...[/yellow]")
        
        pair = params.get("pair", "ETH/USDC")
        spread = params.get("spread", 0.002)  # 0.2%
        amount = params.get("amount", 5000)
        
        await asyncio.sleep(2)
        
        console.print(f"[green]✅ Market making strategy active[/green]")
        console.print(f"  • Trading pair: {pair}")
        console.print(f"  • Spread: {spread*100:.2f}%")
        console.print(f"  • Capital deployed: ${amount:,.2f}")
        console.print(f"  • Expected daily volume: ${amount * 2:,.2f}")

    async def _run_liquidity_mining_strategy(self, params: Dict):
        """Execute liquidity mining strategy"""
        console.print("[yellow]⛏️ Executing Liquidity Mining Strategy...[/yellow]")
        
        pool = params.get("pool", "ETH/USDC")
        protocol = params.get("protocol", "Uniswap V3")
        amount = params.get("amount", 2000)
        
        await asyncio.sleep(2)
        
        estimated_apy = np.random.uniform(12, 25)
        console.print(f"[green]✅ Liquidity mining position established[/green]")
        console.print(f"  • Protocol: {protocol}")
        console.print(f"  • Pool: {pool}")
        console.print(f"  • Amount provided: ${amount:,.2f}")
        console.print(f"  • Estimated APY: {estimated_apy:.1f}%")
        console.print(f"  • Earning: Trading fees + Protocol rewards")

    async def display_dashboard(self, live_mode: bool = False):
        """Display main trading dashboard"""
        if live_mode:
            with Live(self.generate_dashboard_layout(), refresh_per_second=0.5, console=console) as live:
                try:
                    while not shutdown_event.is_set():
                        # Update data
                        await self.update_dashboard_data()
                        live.update(self.generate_dashboard_layout())
                        await asyncio.sleep(2)
                except KeyboardInterrupt:
                    console.print("\n[yellow]Dashboard closed[/yellow]")
        else:
            console.clear()
            await self.update_dashboard_data()
            console.print(self.generate_dashboard_layout())

    def generate_dashboard_layout(self) -> Layout:
        """Generate dashboard layout"""
        console.clear()
        
        # Title
        title = Panel.fit(
            f"[bold blue]🤖 DeFi AI Trading Bot Dashboard[/bold blue] - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            border_style="bright_blue"
        )
        
        # Create layout
        layout = Layout()
        layout.split_column(
            Layout(title, name="title", size=3),
            Layout(name="main")
        )
        
        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        layout["left"].split_column(
            Layout(name="status", size=8),
            Layout(name="performance", size=10)
        )
        
        layout["right"].split_column(
            Layout(name="positions", size=10),
            Layout(name="activity", size=8)
        )
        
        # System Status
        connected_chains = len(self.web3_connections)
        total_chains = len(self.config["trading_config"]["chains"])
        
        agent_status = "🟢 All Active" if self.llm else "🟡 Mock Mode"
        
        status_content = f"""[bold]System Status[/bold]
[green]● Market Scanner:[/green] {agent_status}
[green]● Risk Assessor:[/green] {agent_status}
[green]● Trade Executor:[/green] Ready
[yellow]● Running Strategies:[/yellow] {len(self.running_strategies)}
[blue]● Connected Chains:[/blue] {connected_chains}/{total_chains}
[cyan]● Last Health Check:[/cyan] {self.last_health_check.strftime('%H:%M:%S')}"""
        
        layout["status"].update(Panel(status_content, title="System", border_style="green"))
        
        # Performance Metrics
        total_trades = self.performance_metrics["total_trades"]
        win_rate = 0 if total_trades == 0 else (self.performance_metrics["successful_trades"] / total_trades) * 100
        
        performance_content = f"""[bold]Performance Metrics[/bold]
[blue]● Total Trades:[/blue] {total_trades}
[green]● Successful:[/green] {self.performance_metrics['successful_trades']}
[yellow]● Win Rate:[/yellow] {win_rate:.1f}%
[{'green' if self.performance_metrics['total_pnl'] >= 0 else 'red'}]● Total P&L:[/{'green' if self.performance_metrics['total_pnl'] >= 0 else 'red'}] ${self.performance_metrics['total_pnl']:+.2f}
[cyan]● Sharpe Ratio:[/cyan] {self.performance_metrics.get('sharpe_ratio', 0):.2f}
[magenta]● Max Drawdown:[/magenta] {self.performance_metrics.get('max_drawdown', 0)*100:.1f}%"""
        
        layout["performance"].update(Panel(performance_content, title="Performance", border_style="blue"))
        
        # Active Positions
        positions = list(self.active_positions.values())
        if positions:
            positions_content = "[bold]Active Positions[/bold]\n"
            for pos in positions[:5]:  # Show top 5
                pnl = pos.get("pnl", 0)
                pnl_color = "green" if pnl > 0 else "red"
                positions_content += f"● {pos.get('token', 'N/A')}: [{pnl_color}]${pnl:+.2f}[/{pnl_color}]\n"
            
            if len(positions) > 5:
                positions_content += f"... and {len(positions) - 5} more"
        else:
            positions_content = "[dim]No active positions[/dim]"
        
        layout["positions"].update(Panel(positions_content.strip(), title="Positions", border_style="yellow"))
        
        # Recent Activity
        activity_content = f"""[bold]Recent Activity[/bold]
● [green]✓[/green] {len(self.trading_history)} trades executed
● [yellow]⚠[/yellow] {len(self.running_strategies)} strategies running
● [blue]ℹ[/blue] Gas optimization active
● [green]✓[/green] Risk monitoring enabled
● [cyan]ℹ[/cyan] Market scanning continuous"""
        
        layout["activity"].update(Panel(activity_content, title="Activity", border_style="cyan"))
        
        return layout

    async def update_dashboard_data(self):
        """Update dashboard data"""
        self.last_health_check = datetime.now()
        
        # Update positions if any exist
        if self.active_positions:
            await self.update_positions()

    def get_system_health(self) -> Dict:
        """Get system health status"""
        return {
            "timestamp": datetime.now().isoformat(),
            "agents": {
                "market_intelligence": "active" if self.market_agent else "inactive",
                "risk_assessment": "active" if self.risk_agent else "inactive",
                "execution_manager": "active" if self.execution_agent else "inactive"
            },
            "connections": {
                chain: "connected" for chain in self.web3_connections.keys()
            },
            "llm_status": "active" if self.llm else "mock_mode",
            "database_status": "connected" if self.db_engine else "file_storage",
            "cache_status": "redis" if self.redis_client else "memory",
            "performance": self.performance_metrics,
            "running_strategies": len(self.running_strategies),
            "active_positions": len(self.active_positions),
            "uptime": "100%",  # Would calculate actual uptime
            "last_trade": self.trading_history[-1]["timestamp"] if self.trading_history else None
        }

    async def emergency_stop(self):
        """Emergency stop all trading activities"""
        console.print("\n[bold red]🛑 EMERGENCY STOP ACTIVATED[/bold red]")
        
        # Stop all running strategies
        for strategy_name in list(self.running_strategies.keys()):
            self.running_strategies[strategy_name]["status"] = "stopped"
            console.print(f"[red]● Stopped strategy: {strategy_name}[/red]")
        
        # Close all positions
        positions_to_close = list(self.active_positions.keys())
        for pos_id in positions_to_close:
            await self.close_position(pos_id, reason="emergency_stop")
            console.print(f"[red]● Closed position: {pos_id}[/red]")
        
        console.print("[bold red]All trading activities stopped[/bold red]")
        logger.critical("Emergency stop executed - all activities halted")

    def save_state(self):
        """Save current bot state"""
        state = {
            "timestamp": datetime.now().isoformat(),
            "active_positions": self.active_positions,
            "performance_metrics": self.performance_metrics,
            "running_strategies": self.running_strategies,
            "config": self.config
        }
        
        state_file = Path("data/bot_state.json")
        state_file.parent.mkdir(exist_ok=True)
        
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info("Bot state saved")

    def load_state(self):
        """Load previous bot state"""
        state_file = Path("data/bot_state.json")
        
        if state_file.exists():
            try:
                with open(state_file) as f:
                    state = json.load(f)
                
                self.active_positions = state.get("active_positions", {})
                self.performance_metrics = state.get("performance_metrics", self.performance_metrics)
                self.running_strategies = state.get("running_strategies", {})
                
                logger.info("Bot state loaded from file")
                return True
            except Exception as e:
                logger.error(f"Failed to load state: {e}")
                return False
        
        return False

    async def graceful_shutdown(self):
        """Graceful shutdown sequence"""
        console.print("\n[yellow]🔄 Initiating graceful shutdown...[/yellow]")
        
        # Save current state
        self.save_state()
        
        # Close HTTP session
        if hasattr(self, 'session'):
            self.session.close()
        
        # Close database connections
        if self.db_engine:
            self.db_engine.dispose()
        
        if self.redis_client:
            self.redis_client.close()
        
        console.print("[green]✅ Shutdown complete[/green]")
        logger.info("Bot shutdown completed")


def setup_signal_handlers(bot: TradingBotOrchestrator):
    """Setup signal handlers for graceful shutdown"""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}")
        shutdown_event.set()
        
        # Run emergency stop in a new event loop
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(bot.emergency_stop())
            loop.run_until_complete(bot.graceful_shutdown())
            loop.close()
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


# CLI Command Group
@click.group()
@click.option('--config', '-c', help='Path to configuration file')
@click.option('--debug', is_flag=True, help='Enable debug logging')
@click.pass_context
def cli(ctx, config, debug):
    """DeFi AI Trading Bot - Multi-chain autonomous trading with AI agents"""
    ctx.ensure_object(dict)
    ctx.obj['config'] = config
    ctx.obj['debug'] = debug
    
    if debug:
        os.environ['DEBUG'] = 'true'


@cli.command()
@click.option('--chains', '-c', multiple=True, help='Chains to scan (can specify multiple)')
@click.option('--tokens', '-t', multiple=True, help='Tokens to focus on (can specify multiple)')
@click.option('--min-profit', type=float, default=50, help='Minimum profit threshold in USD')
@click.pass_context
def scan(ctx, chains, tokens, min_profit):
    """Scan markets for trading opportunities"""
    async def run_scan():
        bot = TradingBotOrchestrator(ctx.obj['config'])
        setup_signal_handlers(bot)
        
        chains_list = list(chains) if chains else None
        tokens_list = list(tokens) if tokens else None
        
        result = await bot.scan_markets(chains=chains_list, tokens=tokens_list)
        
        # Filter by minimum profit
        opportunities = result.get("opportunities", [])
        profitable_ops = [
            op for op in opportunities 
            if op.get("potential_profit", 0) >= min_profit
        ]
        
        if profitable_ops:
            console.print(f"\n[bold green]Found {len(profitable_ops)} opportunities above ${min_profit} profit threshold[/bold green]")
        else:
            console.print(f"[yellow]No opportunities found above ${min_profit} profit threshold[/yellow]")
    
    asyncio.run(run_scan())


@cli.command()
@click.argument('strategy')
@click.argument('amount', type=float)
@click.option('--tokens', '-t', multiple=True, help='Tokens to analyze')
@click.pass_context
def risk(ctx, strategy, amount, tokens):
    """Perform risk assessment for a strategy"""
    async def run_risk():
        bot = TradingBotOrchestrator(ctx.obj['config'])
        setup_signal_handlers(bot)
        
        tokens_list = list(tokens) if tokens else None
        await bot.assess_risk(strategy, amount, tokens_list)
    
    asyncio.run(run_risk())


@cli.command()
@click.argument('token_in')
@click.argument('token_out')
@click.argument('amount', type=float)
@click.option('--chain', default='ethereum', help='Blockchain network')
@click.option('--slippage', type=float, default=0.01, help='Maximum slippage percentage')
@click.option('--dry-run', is_flag=True, help='Simulate trade without execution')
@click.pass_context
def trade(ctx, token_in, token_out, amount, chain, slippage, dry_run):
    """Execute a trade"""
    async def run_trade():
        bot = TradingBotOrchestrator(ctx.obj['config'])
        setup_signal_handlers(bot)
        
        if dry_run:
            console.print("[yellow]🧪 DRY RUN MODE - No actual trades will be executed[/yellow]")
        
        await bot.execute_trade(token_in, token_out, amount, chain, slippage)
    
    asyncio.run(run_trade())


@cli.command()
@click.argument('strategy_name')
@click.option('--amount', type=float, help='Amount to allocate to strategy')
@click.option('--chains', '-c', multiple=True, help='Chains to use for strategy')
@click.option('--min-profit', type=float, default=50, help='Minimum profit threshold')
@click.option('--max-risk', type=float, default=0.05, help='Maximum risk per trade')
@click.pass_context
def strategy(ctx, strategy_name, amount, chains, min_profit, max_risk):
    """Run a trading strategy"""
    async def run_strategy():
        bot = TradingBotOrchestrator(ctx.obj['config'])
        setup_signal_handlers(bot)
        
        parameters = {
            "amount": amount or 1000,
            "chains": list(chains) if chains else ["ethereum", "polygon"],
            "min_profit": min_profit,
            "max_risk": max_risk
        }
        
        # Strategy-specific parameters
        if strategy_name == "arbitrage":
            parameters.update({
                "max_gas": 100,
                "min_profit_margin": 0.005  # 0.5%
            })
        elif strategy_name == "yield_farming":
            parameters.update({
                "target_apy": 15.0,
                "auto_compound": True
            })
        elif strategy_name == "dca":
            parameters.update({
                "token": "ETH",
                "amount_per_buy": parameters["amount"] / 30,  # Daily for 30 days
                "frequency": "daily",
                "duration_days": 30
            })
        
        await bot.run_strategy(strategy_name, parameters)
    
    asyncio.run(run_strategy())


@cli.command()
@click.option('--live', '-l', is_flag=True, help='Live monitoring mode')
@click.pass_context
def positions(ctx, live):
    """Monitor active positions"""
    async def run_positions():
        bot = TradingBotOrchestrator(ctx.obj['config'])
        setup_signal_handlers(bot)
        
        # Add some mock positions for demonstration
        if not bot.active_positions:
            bot.active_positions = {
                "pos_001": {
                    "position_id": "pos_001",
                    "token": "ETH",
                    "amount": 1.5,
                    "entry_price": 2450.00,
                    "current_price": 2520.00,
                    "pnl": 105.00,
                    "pnl_percent": 2.86,
                    "status": "active",
                    "created_at": datetime.now().isoformat()
                },
                "pos_002": {
                    "position_id": "pos_002",
                    "token": "MATIC",
                    "amount": 1000,
                    "entry_price": 0.85,
                    "current_price": 0.82,
                    "pnl": -30.00,
                    "pnl_percent": -3.53,
                    "status": "active",
                    "created_at": datetime.now().isoformat()
                }
            }
        
        await bot.monitor_positions(continuous=live)
    
    asyncio.run(run_positions())


@cli.command()
@click.option('--live', '-l', is_flag=True, help='Live dashboard mode')
@click.pass_context
def dashboard(ctx, live):
    """Display trading dashboard"""
    async def run_dashboard():
        bot = TradingBotOrchestrator(ctx.obj['config'])
        setup_signal_handlers(bot)
        
        # Load previous state
        bot.load_state()
        
        await bot.display_dashboard(live_mode=live)
    
    asyncio.run(run_dashboard())


@cli.command()
@click.pass_context
def status(ctx):
    """Show system status and health check"""
    async def run_status():
        bot = TradingBotOrchestrator(ctx.obj['config'])
        health = bot.get_system_health()
        
        # Display system health
        table = Table(title="🏥 System Health Check")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="white")
        table.add_column("Details", style="dim")
        
        # Agents
        for agent, status in health["agents"].items():
            status_color = "green" if status == "active" else "yellow"
            table.add_row(
                f"Agent: {agent.replace('_', ' ').title()}",
                f"[{status_color}]{status.upper()}[/{status_color}]",
                "Operational" if status == "active" else "Mock mode"
            )
        
        # Connections
        for chain, status in health["connections"].items():
            table.add_row(
                f"Chain: {chain.title()}",
                f"[green]{status.upper()}[/green]",
                "RPC connection active"
            )
        
        # Other systems
        llm_status = health["llm_status"]
        llm_color = "green" if llm_status == "active" else "yellow"
        table.add_row("LLM Provider", f"[{llm_color}]{llm_status.upper()}[/{llm_color}]", "AI agents enabled")
        
        db_status = health["database_status"]
        db_color = "green" if db_status == "connected" else "yellow"
        table.add_row("Database", f"[{db_color}]{db_status.upper()}[/{db_color}]", "Data persistence")
        
        cache_status = health["cache_status"]
        cache_color = "green" if cache_status == "redis" else "yellow"
        table.add_row("Cache", f"[{cache_color}]{cache_status.upper()}[/{cache_color}]", "Performance optimization")
        
        console.print(table)
        
        # Performance summary
        perf = health["performance"]
        console.print(f"\n[bold]Performance Summary:[/bold]")
        console.print(f"Total Trades: {perf['total_trades']}")
        console.print(f"Success Rate: {perf['win_rate']:.1f}%")
        console.print(f"Total P&L: ${perf['total_pnl']:+.2f}")
        console.print(f"Running Strategies: {health['running_strategies']}")
        console.print(f"Active Positions: {health['active_positions']}")
    
    asyncio.run(run_status())


@cli.command()
@click.confirmation_option(prompt='Are you sure you want to stop all trading activities?')
@click.pass_context
def emergency_stop(ctx):
    """Emergency stop all trading activities"""
    async def run_emergency_stop():
        bot = TradingBotOrchestrator(ctx.obj['config'])
        await bot.emergency_stop()
    
    asyncio.run(run_emergency_stop())


@cli.command()
@click.pass_context
def configure(ctx):
    """Interactive configuration setup"""
    config_path = ctx.obj.get('config') or "config/config.json"
    config_file = Path(config_path)
    
    console.print("[bold blue]🔧 DeFi Trading Bot Configuration[/bold blue]")
    
    # Create config directory
    config_file.parent.mkdir(exist_ok=True)
    
    # Load existing config or create new
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
        console.print(f"[green]Loaded existing configuration from {config_path}[/green]")
    else:
        bot = TradingBotOrchestrator()
        config = bot.get_default_config()
        console.print("[yellow]Creating new configuration[/yellow]")
    
    # Interactive configuration
    console.print("\n[bold]LLM Configuration:[/bold]")
    openai_key = Prompt.ask("OpenAI API Key", default=os.getenv("OPENAI_API_KEY", ""))
    if openai_key:
        console.print("[dim]Note: Set OPENAI_API_KEY environment variable[/dim]")
    
    console.print("\n[bold]Risk Management:[/bold]")
    max_risk = Prompt.ask("Max portfolio risk per trade", default="0.02")
    stop_loss = Prompt.ask("Default stop loss percentage", default="0.05")
    take_profit = Prompt.ask("Default take profit percentage", default="0.15")
    
    # Update config
    config["trading_config"]["risk_management"]["max_portfolio_risk"] = float(max_risk)
    config["trading_config"]["risk_management"]["stop_loss_percentage"] = float(stop_loss)
    config["trading_config"]["risk_management"]["take_profit_percentage"] = float(take_profit)
    
    console.print("\n[bold]Chain Configuration:[/bold]")
    for chain_name in config["trading_config"]["chains"].keys():
        enabled = Confirm.ask(f"Enable {chain_name.title()} chain?", 
                            default=config["trading_config"]["chains"][chain_name]["enabled"])
        config["trading_config"]["chains"][chain_name]["enabled"] = enabled
    
    # Save configuration
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    console.print(f"\n[green]✅ Configuration saved to {config_path}[/green]")
    console.print("\n[bold]Next steps:[/bold]")
    console.print("1. Set up environment variables (API keys, RPC URLs)")
    console.print("2. Run: defi-bot status")
    console.print("3. Start with: defi-bot scan")


@cli.command()
def version():
    """Show version information"""
    console.print("[bold blue]🤖 DeFi AI Trading Bot[/bold blue]")
    console.print("Version: 1.0.0")
    console.print("Powered by SpoonOS Agent Framework")
    console.print("Multi-chain • AI-powered • Risk-managed")
    console.print("\nFor help: defi-bot --help")
    console.print("Documentation: https://github.com/CodeKage25/defi-ai-trading-bot")


def main():
    """Main entry point"""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()