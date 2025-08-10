"""
Web Interface - FastAPI-based web dashboard and API
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import asyncio
import json
from pathlib import Path
import os
import uuid

from fastapi import FastAPI, WebSocket, HTTPException, Depends, BackgroundTasks, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn
from pydantic import BaseModel, Field
from loguru import logger
import websockets
from starlette.websockets import WebSocketDisconnect

from strategies import StrategyManager, create_strategy, TradeSignal
from risk import RiskManager, PortfolioRisk, PositionRisk
from execution import ExecutionEngine, OrderRequest, OrderType, ExecutionStrategy
from cli.main import TradingBotOrchestrator


# Pydantic models for API
class TradeRequest(BaseModel):
    token_in: str
    token_out: str
    amount_in: float
    chain: str = "ethereum"
    slippage_tolerance: float = 0.01
    execution_strategy: str = "balanced"
    order_type: str = "market"


class StrategyRequest(BaseModel):
    strategy_type: str
    parameters: Dict[str, Any]
    enabled: bool = True


class RiskAssessmentRequest(BaseModel):
    strategy: str
    amount: float
    tokens: Optional[List[str]] = None


class WebSocketMessage(BaseModel):
    type: str
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)


class DashboardData(BaseModel):
    portfolio_summary: Dict[str, Any]
    active_positions: List[Dict[str, Any]]
    recent_trades: List[Dict[str, Any]]
    performance_metrics: Dict[str, Any]
    market_opportunities: List[Dict[str, Any]]
    system_status: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)


# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.client_data: Dict[str, Dict] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.client_data[client_id] = {"websocket": websocket, "subscriptions": set()}
        logger.info(f"WebSocket client {client_id} connected")
    
    def disconnect(self, websocket: WebSocket, client_id: str):
        self.active_connections.remove(websocket)
        if client_id in self.client_data:
            del self.client_data[client_id]
        logger.info(f"WebSocket client {client_id} disconnected")
    
    async def send_personal_message(self, message: dict, client_id: str):
        if client_id in self.client_data:
            websocket = self.client_data[client_id]["websocket"]
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
    
    async def broadcast(self, message: dict, subscription_type: str = None):
        disconnected = []
        
        for client_id, client_info in self.client_data.items():
            # Check if client is subscribed to this type of message
            if subscription_type and subscription_type not in client_info["subscriptions"]:
                continue
            
            websocket = client_info["websocket"]
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error broadcasting to {client_id}: {e}")
                disconnected.append(websocket)
        
        # Clean up disconnected clients
        for ws in disconnected:
            if ws in self.active_connections:
                self.active_connections.remove(ws)


# Create FastAPI app
app = FastAPI(
    title="DeFi AI Trading Bot API",
    description="Advanced AI-powered DeFi trading bot with multi-chain support",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer(auto_error=False)

# Global instances
trading_bot: Optional[TradingBotOrchestrator] = None
connection_manager = ConnectionManager()
templates = Jinja2Templates(directory="templates")

# Static files
static_path = Path("static")
if static_path.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")


def get_trading_bot():
    """Dependency to get trading bot instance"""
    global trading_bot
    if trading_bot is None:
        trading_bot = TradingBotOrchestrator()
    return trading_bot


async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API token (simplified)"""
    if not credentials:
        return None
    
    
    api_key = os.getenv("API_KEY", "demo-key")
    if credentials.credentials == api_key:
        return {"user_id": "demo_user"}
    
    return None


# HTML Routes
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page"""
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/trading", response_class=HTMLResponse)
async def trading_interface(request: Request):
    """Trading interface page"""
    return templates.TemplateResponse("trading.html", {"request": request})


@app.get("/portfolio", response_class=HTMLResponse)
async def portfolio_page(request: Request):
    """Portfolio management page"""
    return templates.TemplateResponse("portfolio.html", {"request": request})


@app.get("/strategies", response_class=HTMLResponse)
async def strategies_page(request: Request):
    """Strategy management page"""
    return templates.TemplateResponse("strategies.html", {"request": request})


@app.get("/analytics", response_class=HTMLResponse)
async def analytics_page(request: Request):
    """Analytics and performance page"""
    return templates.TemplateResponse("analytics.html", {"request": request})


# API Routes
@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/api/status")
async def get_system_status(bot: TradingBotOrchestrator = Depends(get_trading_bot)):
    """Get comprehensive system status"""
    try:
        health_status = bot.get_system_health()
        return JSONResponse(content=health_status)
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/dashboard")
async def get_dashboard_data(bot: TradingBotOrchestrator = Depends(get_trading_bot)) -> DashboardData:
    """Get dashboard data"""
    try:
        # Get portfolio summary
        portfolio_summary = {
            "total_value": 25000.50,  # Mock data
            "daily_pnl": 125.75,
            "total_pnl": 2847.25,
            "positions_count": len(bot.active_positions),
            "active_strategies": len(bot.running_strategies)
        }
        
        # Get active positions
        active_positions = [
            {
                "id": pos_id,
                "token_pair": pos.get("token", "ETH") + "/USDC",
                "size": pos.get("amount", 0),
                "entry_price": pos.get("entry_price", 0),
                "current_price": pos.get("current_price", 0),
                "pnl": pos.get("pnl", 0),
                "pnl_percentage": pos.get("pnl_percentage", 0),
                "strategy": pos.get("strategy", "manual")
            }
            for pos_id, pos in bot.active_positions.items()
        ]
        
        # Get recent trades
        recent_trades = bot.trading_history[-10:] if bot.trading_history else []
        
        # Performance metrics
        performance_metrics = bot.performance_metrics
        
        # Market opportunities (mock)
        market_opportunities = [
            {
                "id": "arb_001",
                "type": "arbitrage",
                "tokens": ["ETH", "USDC"],
                "chain": "ethereum",
                "profit_potential": 145.50,
                "confidence": 0.87,
                "risk_level": "medium"
            },
            {
                "id": "yield_002",
                "type": "yield_farming",
                "tokens": ["USDC", "USDT"],
                "chain": "polygon",
                "profit_potential": 89.25,
                "confidence": 0.92,
                "risk_level": "low"
            }
        ]
        
        # System status
        system_status = bot.get_system_health()
        
        return DashboardData(
            portfolio_summary=portfolio_summary,
            active_positions=active_positions,
            recent_trades=recent_trades,
            performance_metrics=performance_metrics,
            market_opportunities=market_opportunities,
            system_status=system_status
        )
        
    except Exception as e:
        logger.error(f"Error getting dashboard data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/trades/execute")
async def execute_trade(
    trade_request: TradeRequest,
    background_tasks: BackgroundTasks,
    bot: TradingBotOrchestrator = Depends(get_trading_bot),
    auth = Depends(verify_token)
):
    """Execute a trade"""
    try:
        # Execute trade in background
        result = await bot.execute_trade(
            token_in=trade_request.token_in,
            token_out=trade_request.token_out,
            amount=trade_request.amount_in,
            chain=trade_request.chain,
            slippage=trade_request.slippage_tolerance
        )
        
        # Broadcast trade update to connected clients
        await connection_manager.broadcast({
            "type": "trade_executed",
            "data": result,
            "timestamp": datetime.now().isoformat()
        }, "trades")
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Error executing trade: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/trades/history")
async def get_trade_history(
    limit: int = 50,
    offset: int = 0,
    bot: TradingBotOrchestrator = Depends(get_trading_bot)
):
    """Get trade history"""
    try:
        history = bot.trading_history[offset:offset+limit]
        return {
            "trades": history,
            "total": len(bot.trading_history),
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        logger.error(f"Error getting trade history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/positions")
async def get_positions(bot: TradingBotOrchestrator = Depends(get_trading_bot)):
    """Get active positions"""
    try:
        positions = [
            {
                "position_id": pos_id,
                "token": pos.get("token", ""),
                "amount": pos.get("amount", 0),
                "entry_price": pos.get("entry_price", 0),
                "current_price": pos.get("current_price", 0),
                "pnl": pos.get("pnl", 0),
                "pnl_percentage": pos.get("pnl_percentage", 0),
                "created_at": pos.get("created_at", datetime.now().isoformat())
            }
            for pos_id, pos in bot.active_positions.items()
        ]
        
        return {"positions": positions}
    except Exception as e:
        logger.error(f"Error getting positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/positions/{position_id}/close")
async def close_position(
    position_id: str,
    bot: TradingBotOrchestrator = Depends(get_trading_bot),
    auth = Depends(verify_token)
):
    """Close a specific position"""
    try:
        if position_id not in bot.active_positions:
            raise HTTPException(status_code=404, detail="Position not found")
        
        # Close position logic would go here
        position = bot.active_positions[position_id]
        final_pnl = position.get("pnl", 0)
        
        # Remove from active positions
        del bot.active_positions[position_id]
        
        # Update metrics
        bot.performance_metrics["total_pnl"] += final_pnl
        
        # Broadcast position update
        await connection_manager.broadcast({
            "type": "position_closed",
            "data": {
                "position_id": position_id,
                "final_pnl": final_pnl
            },
            "timestamp": datetime.now().isoformat()
        }, "positions")
        
        return {
            "success": True,
            "position_id": position_id,
            "final_pnl": final_pnl
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error closing position: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/strategies/create")
async def create_trading_strategy(
    strategy_request: StrategyRequest,
    bot: TradingBotOrchestrator = Depends(get_trading_bot),
    auth = Depends(verify_token)
):
    """Create and start a new trading strategy"""
    try:
        strategy_id = f"strategy_{uuid.uuid4().hex[:8]}"
        
        # Create strategy
        strategy = create_strategy(strategy_request.strategy_type, strategy_request.parameters)
        
        if not strategy:
            raise HTTPException(status_code=400, detail="Invalid strategy type or parameters")
        
        # Add to running strategies
        bot.running_strategies[strategy_id] = {
            "strategy": strategy,
            "status": "active" if strategy_request.enabled else "paused",
            "created_at": datetime.now().isoformat(),
            "parameters": strategy_request.parameters
        }
        
        # Broadcast strategy update
        await connection_manager.broadcast({
            "type": "strategy_created",
            "data": {
                "strategy_id": strategy_id,
                "type": strategy_request.strategy_type,
                "status": "active" if strategy_request.enabled else "paused"
            },
            "timestamp": datetime.now().isoformat()
        }, "strategies")
        
        return {
            "success": True,
            "strategy_id": strategy_id,
            "message": f"Strategy {strategy_request.strategy_type} created successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/strategies")
async def get_strategies(bot: TradingBotOrchestrator = Depends(get_trading_bot)):
    """Get all active strategies"""
    try:
        strategies = []
        for strategy_id, strategy_info in bot.running_strategies.items():
            strategies.append({
                "strategy_id": strategy_id,
                "type": strategy_info.get("strategy", {}).get("name", "unknown"),
                "status": strategy_info.get("status", "unknown"),
                "created_at": strategy_info.get("created_at", ""),
                "parameters": strategy_info.get("parameters", {}),
                "performance": strategy_info.get("performance", {})
            })
        
        return {"strategies": strategies}
    except Exception as e:
        logger.error(f"Error getting strategies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/strategies/{strategy_id}/toggle")
async def toggle_strategy(
    strategy_id: str,
    bot: TradingBotOrchestrator = Depends(get_trading_bot),
    auth = Depends(verify_token)
):
    """Toggle strategy on/off"""
    try:
        if strategy_id not in bot.running_strategies:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        strategy_info = bot.running_strategies[strategy_id]
        current_status = strategy_info.get("status", "paused")
        new_status = "paused" if current_status == "active" else "active"
        
        strategy_info["status"] = new_status
        
        # Broadcast strategy update
        await connection_manager.broadcast({
            "type": "strategy_toggled",
            "data": {
                "strategy_id": strategy_id,
                "status": new_status
            },
            "timestamp": datetime.now().isoformat()
        }, "strategies")
        
        return {
            "success": True,
            "strategy_id": strategy_id,
            "status": new_status
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error toggling strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/strategies/{strategy_id}")
async def delete_strategy(
    strategy_id: str,
    bot: TradingBotOrchestrator = Depends(get_trading_bot),
    auth = Depends(verify_token)
):
    """Delete a strategy"""
    try:
        if strategy_id not in bot.running_strategies:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        # Stop and remove strategy
        del bot.running_strategies[strategy_id]
        
        # Broadcast strategy deletion
        await connection_manager.broadcast({
            "type": "strategy_deleted",
            "data": {"strategy_id": strategy_id},
            "timestamp": datetime.now().isoformat()
        }, "strategies")
        
        return {
            "success": True,
            "strategy_id": strategy_id,
            "message": "Strategy deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/risk/assess")
async def assess_risk(
    risk_request: RiskAssessmentRequest,
    bot: TradingBotOrchestrator = Depends(get_trading_bot)
):
    """Perform risk assessment"""
    try:
        risk_analysis = await bot.assess_risk(
            strategy=risk_request.strategy,
            amount=risk_request.amount,
            tokens=risk_request.tokens
        )
        
        return JSONResponse(content=risk_analysis)
        
    except Exception as e:
        logger.error(f"Error assessing risk: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/market/scan")
async def scan_market_opportunities(
    chains: Optional[str] = None,
    tokens: Optional[str] = None,
    min_profit: float = 50,
    bot: TradingBotOrchestrator = Depends(get_trading_bot)
):
    """Scan for market opportunities"""
    try:
        chains_list = chains.split(",") if chains else None
        tokens_list = tokens.split(",") if tokens else None
        
        opportunities = await bot.scan_markets(
            chains=chains_list,
            tokens=tokens_list
        )
        
        # Filter by minimum profit
        filtered_opportunities = [
            opp for opp in opportunities.get("opportunities", [])
            if opp.get("potential_profit", 0) >= min_profit
        ]
        
        return {
            "opportunities": filtered_opportunities,
            "total_found": len(opportunities.get("opportunities", [])),
            "filtered_count": len(filtered_opportunities),
            "scan_timestamp": opportunities.get("timestamp")
        }
        
    except Exception as e:
        logger.error(f"Error scanning market: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analytics/performance")
async def get_performance_analytics(
    timeframe: str = "24h",
    bot: TradingBotOrchestrator = Depends(get_trading_bot)
):
    """Get performance analytics"""
    try:
        # Mock performance data - in production, calculate from actual data
        performance_data = {
            "timeframe": timeframe,
            "total_return": 5.2,  # percentage
            "sharpe_ratio": 1.85,
            "max_drawdown": 3.1,
            "win_rate": 78.5,
            "total_trades": 45,
            "profitable_trades": 35,
            "average_trade_duration": 4.5,  # hours
            "best_performing_strategy": "arbitrage",
            "worst_performing_strategy": "trend_following",
            "daily_pnl": [
                {"date": "2024-01-01", "pnl": 125.50},
                {"date": "2024-01-02", "pnl": 89.25},
                {"date": "2024-01-03", "pnl": 234.75},
                {"date": "2024-01-04", "pnl": -45.20},
                {"date": "2024-01-05", "pnl": 167.80}
            ],
            "strategy_performance": {
                "arbitrage": {"return": 8.5, "trades": 20, "win_rate": 85},
                "yield_farming": {"return": 6.2, "trades": 15, "win_rate": 93},
                "trend_following": {"return": 1.8, "trades": 10, "win_rate": 60}
            }
        }
        
        return JSONResponse(content=performance_data)
        
    except Exception as e:
        logger.error(f"Error getting performance analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/emergency-stop")
async def emergency_stop(
    bot: TradingBotOrchestrator = Depends(get_trading_bot),
    auth = Depends(verify_token)
):
    """Emergency stop all trading activities"""
    try:
        await bot.emergency_stop()
        
        # Broadcast emergency stop to all clients
        await connection_manager.broadcast({
            "type": "emergency_stop",
            "data": {"message": "All trading activities stopped"},
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "success": True,
            "message": "Emergency stop executed - all trading activities halted"
        }
        
    except Exception as e:
        logger.error(f"Error executing emergency stop: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket endpoint
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time updates"""
    await connection_manager.connect(websocket, client_id)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle client subscriptions
            if message.get("type") == "subscribe":
                subscriptions = message.get("data", {}).get("subscriptions", [])
                if client_id in connection_manager.client_data:
                    connection_manager.client_data[client_id]["subscriptions"].update(subscriptions)
                
                await connection_manager.send_personal_message({
                    "type": "subscription_confirmed",
                    "data": {"subscriptions": list(connection_manager.client_data[client_id]["subscriptions"])},
                    "timestamp": datetime.now().isoformat()
                }, client_id)
            
            elif message.get("type") == "unsubscribe":
                subscriptions = message.get("data", {}).get("subscriptions", [])
                if client_id in connection_manager.client_data:
                    for sub in subscriptions:
                        connection_manager.client_data[client_id]["subscriptions"].discard(sub)
            
            elif message.get("type") == "ping":
                await connection_manager.send_personal_message({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                }, client_id)
                
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket, client_id)
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
        connection_manager.disconnect(websocket, client_id)


# Background task to broadcast real-time updates
async def broadcast_updates():
    """Background task to broadcast periodic updates"""
    while True:
        try:
            # Broadcast portfolio updates every 10 seconds
            if connection_manager.active_connections:
                global trading_bot
                if trading_bot:
                    dashboard_data = await get_dashboard_data(trading_bot)
                    
                    await connection_manager.broadcast({
                        "type": "dashboard_update",
                        "data": dashboard_data.dict(),
                        "timestamp": datetime.now().isoformat()
                    }, "dashboard")
            
            await asyncio.sleep(10)
            
        except Exception as e:
            logger.error(f"Error broadcasting updates: {e}")
            await asyncio.sleep(10)


# Start background tasks
@app.on_event("startup")
async def startup_event():
    """Initialize app on startup"""
    logger.info("Starting DeFi AI Trading Bot Web Interface")
    
    # Initialize trading bot
    global trading_bot
    trading_bot = TradingBotOrchestrator()
    
    # Start background update broadcaster
    asyncio.create_task(broadcast_updates())
    
    logger.info("Web interface started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down DeFi AI Trading Bot Web Interface")
    
    # Graceful shutdown
    if trading_bot:
        await trading_bot.graceful_shutdown()


# Custom error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.now().isoformat()
        }
    )


# Utility function to run the web server
def run_web_server(host: str = "0.0.0.0", port: int = 8000, debug: bool = False):
    """Run the web server"""
    logger.info(f"Starting web server on {host}:{port}")
    
    uvicorn.run(
        "web_interface:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info" if debug else "warning"
    )


if __name__ == "__main__":
    run_web_server(debug=True)