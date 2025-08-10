"""
DeFi AI Trading Bot - Quick Start Script
"""

import asyncio
import os
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich import print as rprint
import json

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from cli.main import TradingBotOrchestrator

console = Console()

def check_environment():
    """Check if environment is properly configured"""
    console.print("\n[bold blue]🔍 Checking Environment Configuration[/bold blue]")
    
    required_vars = [
        "OPENAI_API_KEY",
        "PRIVATE_KEY",
        "ETHEREUM_RPC_URL"
    ]
    
    missing_vars = []
    configured_vars = []
    
    for var in required_vars:
        if os.getenv(var):
            configured_vars.append(var)
        else:
            missing_vars.append(var)
    
    # Display status table
    table = Table(title="Environment Configuration Status")
    table.add_column("Variable", style="cyan")
    table.add_column("Status", style="white")
    table.add_column("Value", style="dim")
    
    for var in required_vars:
        if var in configured_vars:
            value = os.getenv(var)
            # Mask sensitive values
            if "KEY" in var or "PRIVATE" in var:
                display_value = f"{value[:8]}...{value[-4:]}" if len(value) > 12 else "****"
            else:
                display_value = value[:50] + "..." if len(value) > 50 else value
            table.add_row(var, "[green]✅ Set[/green]", display_value)
        else:
            table.add_row(var, "[red]❌ Missing[/red]", "Not configured")
    
    console.print(table)
    
    if missing_vars:
        console.print(f"\n[yellow]⚠️  Missing required environment variables: {', '.join(missing_vars)}[/yellow]")
        console.print("\n[dim]Please configure them in your .env file or as environment variables.[/dim]")
        return False
    else:
        console.print("\n[green]✅ All required environment variables are configured![/green]")
        return True

def display_welcome():
    """Display welcome message and bot info"""
    welcome_text = """
[bold blue]🤖 Welcome to DeFi AI Trading Bot![/bold blue]

This intelligent multi-chain trading bot uses SpoonOS Agent Framework to:
• 🧠 Analyze market opportunities across multiple chains
• ⚠️  Assess risks using advanced AI models
• 🚀 Execute optimal trades with MEV protection
• 📊 Manage positions with automated risk controls

[bold]Supported Features:[/bold]
• Multi-chain support (Ethereum, Polygon, BSC, Arbitrum)
• AI-powered market intelligence and risk assessment
• DEX aggregation for optimal trade execution
• Automated position management and portfolio optimization
• Real-time gas optimization and cost minimization

[dim]Note: This is experimental software. Always test with small amounts first![/dim]
"""
    
    panel = Panel(welcome_text, title="DeFi AI Trading Bot", border_style="blue", padding=1)
    console.print(panel)

def interactive_setup():
    """Interactive setup for first-time users"""
    console.print("\n[bold yellow]🔧 Interactive Setup[/bold yellow]")
    
    # Check if config exists
    config_path = Path("config/config.json")
    if not config_path.exists():
        console.print("[yellow]No configuration file found. Let's create one![/yellow]")
        
        # Basic configuration prompts
        llm_provider = Prompt.ask(
            "Choose LLM provider", 
            choices=["openai", "anthropic"], 
            default="openai"
        )
        
        model_name = "gpt-4-turbo" if llm_provider == "openai" else "claude-3-5-sonnet-20241022"
        
        risk_tolerance = Prompt.ask(
            "Choose risk tolerance",
            choices=["conservative", "moderate", "aggressive"],
            default="moderate"
        )
        
        # Create basic config
        config = {
            "llm_provider": llm_provider,
            "model_name": model_name,
            "trading_config": {
                "risk_management": {
                    "max_portfolio_risk": 0.01 if risk_tolerance == "conservative" else 0.02 if risk_tolerance == "moderate" else 0.03,
                    "max_position_size": 0.05 if risk_tolerance == "conservative" else 0.1 if risk_tolerance == "moderate" else 0.15
                }
            }
        }
        
        # Save config
        config_path.parent.mkdir(exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        console.print(f"[green]✅ Configuration saved to {config_path}[/green]")
    
    return True

async def demo_market_scan():
    """Run a demo market scan"""
    console.print("\n[bold green]🔍 Running Demo Market Scan[/bold green]")
    
    try:
        bot = TradingBotOrchestrator()
        
        # Run market scan
        console.print("Scanning Ethereum and Polygon for ETH and USDC opportunities...")
        result = await bot.scan_markets(
            chains=["ethereum", "polygon"],
            tokens=["ETH", "USDC"]
        )
        
        console.print("[green]✅ Demo market scan completed successfully![/green]")
        return True
        
    except Exception as e:
        console.print(f"[red]❌ Demo failed: {e}[/red]")
        console.print("\n[dim]This might be due to missing API keys or network issues.[/dim]")
        return False

async def demo_risk_assessment():
    """Run a demo risk assessment"""
    console.print("\n[bold yellow]⚠️  Running Demo Risk Assessment[/bold yellow]")
    
    try:
        bot = TradingBotOrchestrator()
        
        # Run risk assessment
        console.print("Assessing risk for ETH/USDC arbitrage strategy with $1000...")
        result = await bot.assess_risk("ETH/USDC arbitrage", 1000.0)
        
        console.print("[green]✅ Demo risk assessment completed successfully![/green]")
        return True
        
    except Exception as e:
        console.print(f"[red]❌ Demo failed: {e}[/red]")
        return False

def show_usage_examples():
    """Show common usage examples"""
    console.print("\n[bold blue]📖 Usage Examples[/bold blue]")
    
    examples = [
        ("Scan markets", "defi-bot scan -c ethereum -c polygon -t ETH -t USDC"),
        ("Assess risk", "defi-bot risk 'arbitrage strategy' 1000"),
        ("Execute trade", "defi-bot trade ETH USDC 1.0 --chain ethereum"),
        ("Monitor positions", "defi-bot positions"),
        ("Run strategy", "defi-bot strategy arbitrage --amount 1000"),
        ("Launch dashboard", "defi-bot dashboard")
    ]
    
    table = Table(title="Common Commands")
    table.add_column("Action", style="cyan")
    table.add_column("Command", style="green")
    
    for action, command in examples:
        table.add_row(action, command)
    
    console.print(table)

async def main():
    """Main startup function"""
    console.clear()
    
    # Welcome message
    display_welcome()
    
    # Environment check
    env_ok = check_environment()
    
    if not env_ok:
        console.print("\n[red]⚠️  Environment not properly configured.[/red]")
        console.print("Please check the README.md for setup instructions.")
        
        if Confirm.ask("\nWould you like to see the setup guide?"):
            console.print("""
[bold]Quick Setup Guide:[/bold]

1. Copy the environment template:
   [green]cp config/.env.example .env[/green]

2. Edit .env with your API keys:
   [green]OPENAI_API_KEY=sk-your-openai-key[/green]
   [green]PRIVATE_KEY=0x1234567890abcdef...[/green]
   [green]ETHEREUM_RPC_URL=https://eth-mainnet.alchemyapi.io/v2/your-key[/green]

3. Run this script again:
   [green]python start_bot.py[/green]
""")
        return
    
    # Interactive setup
    interactive_setup()
    
    # Main menu
    while True:
        console.print("\n[bold blue]🤖 What would you like to do?[/bold blue]")
        
        choice = Prompt.ask(
            "Choose an option",
            choices=["scan", "risk", "demo", "examples", "cli", "quit"],
            default="scan"
        )
        
        if choice == "scan":
            await demo_market_scan()
        elif choice == "risk":
            await demo_risk_assessment()
        elif choice == "demo":
            console.print("\n[bold green]🎬 Running Full Demo[/bold green]")
            await demo_market_scan()
            await demo_risk_assessment()
        elif choice == "examples":
            show_usage_examples()
        elif choice == "cli":
            console.print("\n[bold blue]🖥️  Launching CLI Mode[/bold blue]")
            console.print("Use 'defi-bot --help' to see all available commands.")
            console.print("Example: [green]defi-bot scan -c ethereum -t ETH[/green]")
            break
        elif choice == "quit":
            console.print("\n[yellow]👋 Thanks for trying DeFi AI Trading Bot![/yellow]")
            console.print("[dim]Remember to always test with small amounts first.[/dim]")
            break
        
        if choice in ["scan", "risk", "demo"]:
            if not Confirm.ask("\nWould you like to try something else?", default=True):
                break

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]👋 Goodbye![/yellow]")
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")