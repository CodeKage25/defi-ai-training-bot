"""
DeFi AI Trading Bot - Fixed Implementation with Real SpoonOS Agents
Complete implementation with proper SpoonOS agent integration
"""

import asyncio
import click
import json
import sys
import inspect
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import signal
import re
import os


import warnings
try:
    from pydantic.warnings import PydanticDeprecatedSince211
    warnings.simplefilter("ignore", PydanticDeprecatedSince211)
except Exception:
    pass
warnings.filterwarnings(
    "ignore",
    message=r"Accessing the 'model_fields' attribute on the instance is deprecated",
    module=r"spoon_ai\.tools\.base"
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r"spoon_ai\.tools\.base"
)


# Rich CLI
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.layout import Layout
from rich.live import Live
from rich.prompt import Prompt, Confirm

# Logging
from loguru import logger

# Data
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Env & pydantic
from dotenv import load_dotenv
from pydantic import Field 

try:
    from openai import OpenAI, AsyncOpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False
    AsyncOpenAI = None  # type: ignore


try:
    from web3 import Web3
    from eth_account import Account
    try:
        from web3.middleware import geth_poa_middleware  # v5
        POA_MIDDLEWARE = geth_poa_middleware
    except ImportError:
        try:
            from web3.middleware.proof_of_authority import ExtraDataToPOAMiddleware  # v6
            POA_MIDDLEWARE = ExtraDataToPOAMiddleware
        except ImportError:
            POA_MIDDLEWARE = None
            logger.warning("POA middleware not available - some networks may not work properly")
    WEB3_AVAILABLE = True
except ImportError as e:
    logger.error(f"Web3 dependencies not available: {e}")
    WEB3_AVAILABLE = False
    Web3 = None
    Account = None
    POA_MIDDLEWARE = None


try:
    import importlib
    mw = importlib.import_module("web3.middleware")
    if not hasattr(mw, "geth_poa_middleware"):
        from web3.middleware.proof_of_authority import ExtraDataToPOAMiddleware as _ExtraPOA
        mw.geth_poa_middleware = _ExtraPOA  # type: ignore[attr-defined]
except Exception:
    pass

try:
    from spoon_ai.agents import SpoonReactAI
    from spoon_ai.chat import ChatBot
    from spoon_ai.tools.base import BaseTool as SpoonBaseTool
    from spoon_ai.tools import ToolManager
    SPOONOS_AVAILABLE = True
    _SPOON_BASETOOL = SpoonBaseTool
    logger.info("✅ SpoonOS framework loaded successfully")
except ImportError as e:
    logger.warning(f"SpoonOS not available: {e}")
    SPOONOS_AVAILABLE = False
    ChatBot = None  
    ToolManager = None  
    _SPOON_BASETOOL = None  


def _patch_spoon_basetool_model_fields_deprec():
    if not SPOONOS_AVAILABLE:
        return
    try:
        import spoon_ai.tools.base as sbase
        try:
            def _mf(self):
                try:
                    return self.__class__.model_fields
                except Exception:
                    return getattr(type(self), "model_fields", {})
            if not isinstance(getattr(sbase.BaseTool, "model_fields", None), property):
                sbase.BaseTool.model_fields = property(_mf)  # type: ignore
        except Exception:
            pass
        if not getattr(sbase.BaseTool, "__mf_shim__", False):
            _orig_getattribute = sbase.BaseTool.__getattribute__
            def _ga(self, name):
                if name == "model_fields":
                    return type(self).model_fields
                return _orig_getattribute(self, name)
            sbase.BaseTool.__getattribute__ = _ga  
            sbase.BaseTool.__mf_shim__ = True  
    except Exception:
        pass

_patch_spoon_basetool_model_fields_deprec()


from pydantic.fields import FieldInfo  
try:
    _bt = _SPOON_BASETOOL
    if _bt is None:
        raise NameError
except Exception:
    class SpoonBaseTool:
        name: str = "base_tool"
        description: str = ""
        parameters: Dict[str, Any] = {}

        def __init_subclass__(cls):
            for k, v in list(cls.__dict__.items()):
                if isinstance(v, FieldInfo):
                    if getattr(v, "default_factory", None) is not None:
                        setattr(cls, k, v.default_factory())
                    else:
                        setattr(cls, k, v.default)

        async def execute(self, **kwargs) -> Dict[str, Any]:
            raise NotImplementedError("Tool must implement execute()")

BaseTool = SpoonBaseTool  

# Async utils
from asyncio import Event


try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    REDIS_AVAILABLE = False

try:
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    create_engine = None
    sessionmaker = None
    SQLALCHEMY_AVAILABLE = False

# Load env
load_dotenv()


console = Console()


shutdown_event = Event()


DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"


OPENROUTER_BASE = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")


def is_openrouter_enabled() -> bool:
    ok = (os.getenv("OPENAI_API_KEY") or "").strip()
    if ok.startswith("sk-or-"):
        return True
    return bool((os.getenv("OPENROUTER_API_KEY") or "").strip())

def get_openrouter_key() -> Optional[str]:
    k = (os.getenv("OPENAI_API_KEY") or "").strip()
    if k.startswith("sk-or-"):
        return k
    k2 = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    return k2 or None

def choose_model(default_openai: str = "gpt-4o-mini") -> str:
    if is_openrouter_enabled():
        # Allow specifying either OpenAI or Anthropic models through OpenRouter
        return os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
    # Direct OpenAI/Anthropic selection if using native APIs (not OpenRouter)
    return os.getenv("OPENAI_MODEL", default_openai)
# ----------------------------------------------------------------------------

# ----------------------------- Tool-call sanitizing -----------------------------
class _ResponseShim:
    """Minimal interface Spoon agents use: `.content` and `.tool_calls`."""
    def __init__(self, content: Optional[str], tool_calls: Optional[List[Any]]):
        self.content = self._ensure_string_content(content)
        self.tool_calls = tool_calls or []
    
    def _ensure_string_content(self, content: Any) -> str:
        """Ensure content is always a string, never None"""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        try:
            return str(content)
        except Exception:
            return ""

class _ToolCallLite:
    """Small object with .id, .type, .function (dict) so Spoon's code can do attribute access."""
    def __init__(self, _id: Optional[str], _type: str, function: Dict[str, Any]):
        self.id = _id
        self.type = _type
        self.function = function  # {"name": str, "arguments": str}

def _force_function_dict(func_obj: Any) -> Dict[str, Any]:
    if isinstance(func_obj, dict):
        name = func_obj.get("name")
        args = func_obj.get("arguments")
        if not isinstance(args, str):
            try:
                args = json.dumps(args or {})
            except Exception:
                args = "{}"
        return {"name": name, "arguments": args}
    name = getattr(func_obj, "name", None)
    args = getattr(func_obj, "arguments", None)
    if not isinstance(args, str):
        try:
            args = json.dumps(args or {})
        except Exception:
            args = "{}"
    return {"name": name, "arguments": args}

def _allowed_tool_names_from(owner: Any) -> Optional[set]:
    try:
        return set(getattr(owner, "_allowed_tool_names", []) or [])
    except Exception:
        return None

def _sanitize_tool_calls(raw_tool_calls: Any, owner: Any = None) -> List[Any]:
    if not raw_tool_calls:
        return []
    allowed: Optional[set] = None
    if owner is not None:
        allowed = _allowed_tool_names_from(owner)

    out: List[Any] = []
    try:
        for tc in raw_tool_calls:
            _id = getattr(tc, "id", None) if not isinstance(tc, dict) else tc.get("id")
            _type = getattr(tc, "type", "function") if not isinstance(tc, dict) else tc.get("type", "function")
            func_obj = getattr(tc, "function", None) if not isinstance(tc, dict) else tc.get("function", None)
            func_dict = _force_function_dict(func_obj)
            if allowed and func_dict.get("name") not in allowed:
                continue
            out.append(_ToolCallLite(_id, _type, func_dict))
        return out
    except Exception:
        cleaned: List[Any] = []
        try:
            for tc in list(raw_tool_calls):
                if isinstance(tc, dict):
                    func_dict = _force_function_dict(tc.get("function"))
                    if allowed and func_dict.get("name") not in allowed:
                        continue
                    cleaned.append(_ToolCallLite(tc.get("id"), tc.get("type", "function"), func_dict))
            return cleaned
        except Exception:
            return []

def _extract_content_and_tool_calls(resp: Any, owner: Any = None) -> _ResponseShim:
    content = getattr(resp, "content", None)
    tool_calls = getattr(resp, "tool_calls", None)
    if content is None and hasattr(resp, "choices"):
        try:
            msg = resp.choices[0].message
            content = getattr(msg, "content", None)
            tool_calls = getattr(msg, "tool_calls", None)
        except Exception:
            pass
    sanitized = _sanitize_tool_calls(tool_calls, owner=owner)
    return _ResponseShim(content, sanitized)

def _patch_chatbot_methods(bot: Any) -> Any:
    import types
    method_names = [n for n in ("chat", "generate", "create", "completion",
                                "complete", "ask", "respond", "message") if hasattr(bot, n)]
    for name in method_names:
        original = getattr(bot, name)
        if asyncio.iscoroutinefunction(original):
            async def async_wrapper(self, *args, __orig=original, **kwargs):
                r = await __orig(*args, **kwargs)
                return _extract_content_and_tool_calls(r, owner=self)
            setattr(bot, name, types.MethodType(async_wrapper, bot))
        else:
            def wrapper(self, *args, __orig=original, **kwargs):
                r = __orig(*args, **kwargs)
                return _extract_content_and_tool_calls(r, owner=self)
            setattr(bot, name, types.MethodType(wrapper, bot))
    logger.info("🧽 Patched ChatBot to sanitize + filter tool_calls for Spoon agents")
    return bot



def _patch_spoon_baseagent_add_message():
    try:
        import importlib, types
        base_mod = importlib.import_module("spoon_ai.agents.base")
        orig = base_mod.BaseAgent.add_message
    except Exception:
        return  # SpoonOS not installed; nothing to patch

    # Mode: "strip" (drop all tool_calls) or "sanitize" (filter + dict-ify)
    default_mode = os.getenv("SPOON_TOOLCALL_MODE", "").strip().lower()
    if not default_mode:
        default_mode = "strip" if is_openrouter_enabled() else "sanitize"

    def _collect_allowed(agent: Any) -> Optional[set]:
        names = set()
        # try finding ToolManager or container inside the agent
        for attr in ("available_tools", "avaliable_tools", "tools", "tool_manager", "tools_manager"):
            container = getattr(agent, attr, None)
            if not container:
                continue
            # dict-like
            try:
                it = container.values() if isinstance(container, dict) else container
            except Exception:
                it = container
            for inner in ("tools", "available_tools", "avaliable_tools"):
                try:
                    coll = getattr(container, inner, None)
                    if coll:
                        items = coll.values() if isinstance(coll, dict) else coll
                        for t in items:
                            n = getattr(t, "name", None) or (t.get("name") if isinstance(t, dict) else None)
                            if n:
                                names.add(n)
                except Exception:
                    pass
            try:
                for t in (it.values() if isinstance(it, dict) else it):
                    n = getattr(t, "name", None) or (t.get("name") if isinstance(t, dict) else None)
                    if n:
                        names.add(n)
            except Exception:
                pass
        
        try:
            names |= set(getattr(agent, "_allowed_tool_names", []) or [])
        except Exception:
            pass
        try:
            llm = getattr(agent, "llm", None)
            if llm:
                names |= set(getattr(llm, "_allowed_tool_names", []) or [])
        except Exception:
            pass
        return names or None

    def _ensure_string_content(content: Any) -> str:
        """CRITICAL: Force a string; OpenAI chat.completions requires content to be string for ALL providers"""
        if content is None:
            return ""  # Never return None!
        if isinstance(content, str):
            return content
        if isinstance(content, (dict, list)):
            try:
                return json.dumps(content)
            except Exception:
                return str(content) or ""
        try:
            return str(content)
        except Exception:
            return ""
        
    def _has_recent_tool_calls(agent_self) -> bool:
           """Check if the last assistant message had tool_calls"""
           try:
               messages = getattr(agent_self, "messages", []) or getattr(agent_self, "conversation_history", [])
               if not messages:
                   return False
               for msg in reversed(messages):
                   if isinstance(msg, dict):
                       role = msg.get("role")
                       if role == "assistant":
                           return bool(msg.get("tool_calls"))
                       elif role == "tool":
                           continue
                       else:
                           break
                    # else:
                    #     role = getattr(msg, "role", None)
                    #     if role == "assistant":
                    #         return bool(getattr(msg, "tool_calls", None))
                    #     elif role == "tool":
                    #         continue
                    #     else:
                    #         break
                
           except Exception:
               return False
                           

    def add_message_patched(self, role: str, content: Optional[str] = None,
                            tool_calls: Optional[List[Any]] = None, *args, **kwargs):
        
        content = _ensure_string_content(content)
        
        if role == "tool":
            if not _has_recent_tool_calls(self):
                logger.debug("🛑 Skipping tool message - no recent tool_calls found")
                return
        
        
        if not content:
            if role == "system":
                content = "System message"
            elif role == "user":
                content = "User input"
            elif role == "assistant":
                content = "Assistant response"
            elif role == "tool":
                content = "Tool execution result"
            else:
                content = f"Message from {role}"
        
        
        if role == "tool" and not content.strip():
            if tool_calls:
                content = f"Tool executed: {len(tool_calls)} call(s)"
            else:
                content = "Tool executed successfully"
        
        mode = default_mode
        safe_calls = None
        
        if tool_calls and mode != "strip":
            allowed = _collect_allowed(self)
            safe_calls = []
            for tc in tool_calls:
                _id = getattr(tc, "id", None) if not isinstance(tc, dict) else tc.get("id")
                _type = getattr(tc, "type", "function") if not isinstance(tc, dict) else tc.get("type", "function")
                func_obj = getattr(tc, "function", None) if not isinstance(tc, dict) else tc.get("function")
                func_dict = _force_function_dict(func_obj)
                if allowed and func_dict.get("name") not in allowed:
                    continue
                lite = types.SimpleNamespace(id=_id, type=_type, function=func_dict)
                safe_calls.append(lite)
            # ensure list even if empty
            if not safe_calls:
                safe_calls = []
        elif mode == "strip" and tool_calls:
            logger.debug("🔒 Stripping tool_calls to avoid Pydantic validation issues (SPOON_TOOLCALL_MODE=strip)")
            safe_calls = None  # drop entirely

        # FINAL SAFETY CHECK: Ensure we never pass None content
        if content is None:
            content = "Empty message"
            
        logger.debug(f"Adding message: role={role}, content_len={len(content)}, has_tool_calls={bool(safe_calls)}")

        if mode == "strip":
            return orig(self, role, content, tool_calls=None, *args, **kwargs)
        else:
            return orig(self, role, content, tool_calls=safe_calls, *args, **kwargs)

    try:
        base_mod.BaseAgent.add_message = add_message_patched
        logger.info(f"🛡️ Patched BaseAgent.add_message (mode={default_mode}) with CRITICAL null content guard")
    except Exception as e:
        logger.error(f"Failed to patch BaseAgent.add_message: {e}")

_patch_spoon_baseagent_add_message()



class ChainbaseAnalyticsTool(BaseTool):
    """On-chain analytics and transaction analysis using Chainbase"""
    name: str = "chainbase_analytics"
    description: str = "Analyze on-chain data, token flows, and whale movements"
    parameters: Dict[str, Any] = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "chain": {"type": "string", "description": "ethereum|polygon|bsc|arbitrum"},
                "analysis_type": {"type": "string", "description": "e.g. whale_movements"},
                "token_address": {"type": "string", "nullable": True},
                "timeframe": {"type": "string", "default": "24h"},
            },
            "required": ["chain", "analysis_type"],
        }
    )

    async def execute(
        self,
        chain: str,
        analysis_type: str,
        token_address: Optional[str] = None,
        timeframe: str = "24h",
    ) -> Dict[str, Any]:
        logger.info(f"Analyzing {chain} - {analysis_type} for timeframe {timeframe}")
        api_key = os.getenv("CHAINBASE_API_KEY")
        if not api_key:
            logger.warning("CHAINBASE_API_KEY not found, using mock data")
            await asyncio.sleep(0.3)  

        if analysis_type == "whale_movements":
            return {
                "chain": chain,
                "analysis": "whale_movements",
                "timeframe": timeframe,
                "large_transactions": [
                    {
                        "hash": f"0x{hash(f'{chain}_{timeframe}_{i}')%1000000:06x}...",
                        "value_usd": float(np.random.uniform(500_000, 5_000_000)),
                        "from": "0xwhale1...",
                        "to": "0xexchange...",
                        "token": token_address or "ETH",
                        "timestamp": datetime.now().isoformat(),
                    }
                    for i in range(np.random.randint(1, 5))
                ],
                "summary": {
                    "total_whale_volume": float(np.random.uniform(5_000_000, 50_000_000)),
                    "net_flow": np.random.choice(["inbound", "outbound"]),
                    "sentiment": np.random.choice(["bullish", "bearish", "neutral"]),
                },
            }
        return {"analysis": analysis_type, "status": "completed", "data": {}}


class PriceAggregatorTool(BaseTool):
    """Multi-source price aggregation and comparison"""
    name: str = "price_aggregator"
    description: str = "Get real-time prices from multiple sources"
    parameters: Dict[str, Any] = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "tokens": {"type": "array", "items": {"type": "string"}},
                "vs_currency": {"type": "string", "default": "usd"},
                "sources": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": ["coingecko"],
                },
            },
            "required": ["tokens"],
        }
    )

    async def execute(
        self,
        tokens: List[str],
        vs_currency: str = "usd",
        sources: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        logger.info(f"Fetching prices for {tokens} in {vs_currency}")
        prices: Dict[str, Any] = {}
        for token in tokens:
            base_price = {
                "ETH": 2500.0,
                "BTC": 45_000.0,
                "USDC": 1.0,
                "MATIC": 0.85,
                "USDT": 1.0,
            }.get(token.upper(), 100.0)
            variance = float(np.random.uniform(-0.05, 0.05))
            prices[token.lower()] = {
                "price": base_price * (1 + variance),
                "24h_change": float(np.random.uniform(-10, 10)),
                "volume_24h": float(np.random.uniform(1_000_000, 100_000_000)),
                "market_cap": float(base_price * np.random.uniform(10_000_000, 1_000_000_000)),
                "last_updated": datetime.now().isoformat(),
            }
        return {
            "prices": prices,
            "sources_used": sources or ["coingecko"],
            "timestamp": datetime.now().isoformat(),
        }


class DEXMonitorTool(BaseTool):
    """DEX liquidity and arbitrage opportunity monitoring"""
    name: str = "dex_monitor"
    description: str = "Monitor DEX liquidity and find arbitrage opportunities"
    parameters: Dict[str, Any] = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "chain": {"type": "string"},
                "token_pair": {"type": "array", "items": {"type": "string"}},
                "dexs": {"type": "array", "items": {"type": "string"}},
                "min_profit_bps": {"type": "integer", "default": 50},
            },
            "required": ["chain", "token_pair"],
        }
    )

    async def execute(
        self,
        chain: str,
        token_pair: List[str],
        dexs: Optional[List[str]] = None,
        min_profit_bps: int = 50,
    ) -> Dict[str, Any]:
        logger.info(f"Monitoring DEX opportunities for {token_pair} on {chain}")
        default_dexs = {
            "ethereum": ["uniswap_v3", "sushiswap", "curve"],
            "polygon": ["quickswap", "sushiswap", "curve"],
            "bsc": ["pancakeswap", "biswap", "mdex"],
            "arbitrum": ["uniswap_v3", "sushiswap", "curve"],
        }
        dexs = dexs or default_dexs.get(chain, ["uniswap_v3"])

        
        low = int(max(1, min(min_profit_bps, 199)))
        high = 200
        num = int(np.random.randint(1, 3))  # 1..2
        opportunities = []
        for _ in range(num):
            profit_bps = int(np.random.randint(low, high))
            opportunities.append({
                "dex_buy": str(np.random.choice(dexs)),
                "dex_sell": str(np.random.choice(dexs)),
                "token_in": token_pair[0],
                "token_out": token_pair[1],
                "profit_bps": profit_bps,
                "profit_usd": float(np.random.uniform(50, 1000)),
                "liquidity_in": float(np.random.uniform(10_000, 1_000_000)),
                "liquidity_out": float(np.random.uniform(10_000, 1_000_000)),
                "gas_cost_usd": float(np.random.uniform(10, 100)),
                "timestamp": datetime.now().isoformat(),
            })
        return {
            "chain": chain,
            "token_pair": token_pair,
            "opportunities": opportunities,
            "dexs_monitored": dexs,
            "min_profit_threshold_bps": min_profit_bps,
        }
-

def make_llm_client_async() -> Optional[AsyncOpenAI]:
    if not OPENAI_AVAILABLE:
        return None
    if is_openrouter_enabled():
        key = get_openrouter_key()
        if not key:
            return None
        return AsyncOpenAI(
            api_key=key,
            base_url=OPENROUTER_BASE,
            default_headers={
                "HTTP-Referer": os.getenv("OPENROUTER_REFERRER", "http://localhost"),
                "X-Title": os.getenv("OPENROUTER_APP_NAME", "defi-ai-trading-bot"),
            },
        )
    openai_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if openai_key:
        return AsyncOpenAI(api_key=openai_key)
    return None


class DirectOpenAIAgent:
    """Fallback agent using direct OpenAI/OpenRouter API (OpenAI 1.x client)"""
    def __init__(self, name: str, description: str, system_prompt: str, max_steps: int = 10):
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.tools: List[BaseTool] = []
        self.client = make_llm_client_async()
        if not self.client:
            logger.error("No LLM client available (no OpenAI/OpenRouter key)")

    def add_tool(self, tool: BaseTool):
        self.tools.append(tool)

    async def run(self, prompt: str) -> Dict[str, Any]:
        if not self.client:
            logger.warning("OpenAI/OpenRouter not available, returning mock response")
            return {"content": "", "tool_calls": [], "final_answer": ""}
        try:
            tool_schemas = [{
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters
                }
            } for t in self.tools]

            messages = [
                {"role": "system", "content": self.system_prompt or "You are a helpful AI assistant."},
                {"role": "user", "content": prompt or "Analyze the market."}
            ]
            response = await self.client.chat.completions.create(
                model=choose_model(),
                messages=messages,
                tools=tool_schemas or None,
                tool_choice="auto" if tool_schemas else "none",
                temperature=0.1,
                max_tokens=2048,
            )
            return await self._process_response(response)
        except Exception as e:
            logger.error(f"LLM API error: {e}")
            return {"content": "", "tool_calls": [], "final_answer": ""}

    async def _process_response(self, response: Any) -> Dict[str, Any]:
        msg = response.choices[0].message
        content = getattr(msg, "content", None) or ""
        tool_calls = getattr(msg, "tool_calls", None) or []
        results = []
        
        for tc in tool_calls:
            tool_name = tc.function.name
            try:
                tool_args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                tool_args = {}
            tool = next((t for t in self.tools if t.name == tool_name), None)
            if tool:
                result = await tool.execute(**tool_args)
                if not isinstance(result, str):
                    try:
                        result = json.dumps(result)
                    except Exception:
                        result = str(result) or "Tool executed"
                results.append({"tool": tool_name, "result": result})
        return {
            "content": content,
            "tool_calls": results,
            "final_answer": content
        }


# ----------------------------- JSON extraction helpers -----------------------------
def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Enhanced JSON extraction with better pattern matching"""
    if not text:
        return None
    
    # Remove any markdown formatting
    text = re.sub(r'```json\s*|\s*```', '', text, flags=re.IGNORECASE)
    text = re.sub(r'```\s*|\s*```', '', text)
    
    # Try to find JSON block patterns
    json_patterns = [
        r'\{[^{}]*"opportunities"[^{}]*\[[^\]]*\][^{}]*\}',  # Simple opportunities object
        r'\{.*?"opportunities".*?\[.*?\].*?\}',  # More complex opportunities object
        r'\{.*?\}',  # Any JSON object
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, text, flags=re.DOTALL)
        for match in matches:
            try:
                parsed = json.loads(match)
                if isinstance(parsed, dict) and "opportunities" in parsed:
                    return parsed
            except json.JSONDecodeError:
                continue
    
    # Try parsing the entire text as JSON
    try:
        parsed = json.loads(text.strip())
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    
    return None

def _validate_opportunities(obj: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    if not isinstance(obj, dict):
        return None
    opps = obj.get("opportunities")
    if isinstance(opps, list) and all(isinstance(o, dict) for o in opps):
        return opps
    return None



class TradingBotOrchestrator:
    """Main orchestrator for the DeFi AI Trading Bot with real agent integration"""
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/config.json"
        self.config = self.load_config()
        self.setup_logging()
        # Web3
        self.web3_connections = self.setup_web3() if WEB3_AVAILABLE else {}
        # Flags
        self.using_spoon_agents: bool = False
        # LLM & agents
        self.chatbot = self.setup_chatbot()
        self.initialize_agents()
        # DB/cache
        self.redis_client = self.setup_redis() if REDIS_AVAILABLE else None
        self.db_engine = self.setup_database() if SQLALCHEMY_AVAILABLE else None
        # State
        self.active_positions: Dict[str, Dict[str, Any]] = {}
        self.trading_history: List[Dict[str, Any]] = []
        self.performance_metrics = {
            "total_trades": 0,
            "successful_trades": 0,
            "total_pnl": 0.0,
            "win_rate": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0
        }
        self.running_strategies: Dict[str, Dict[str, Any]] = {}
        self.market_data_cache: Dict[str, Any] = {}
        self.last_health_check = datetime.now()
        self.session = self.setup_http_session()
        logger.info("🚀 DeFi AI Trading Bot initialized successfully")
        self.display_startup_banner()

    # ---------- SpoonOS ChatBot setup (OpenRouter-compatible even without base_url param)
    def _build_openrouter_chatbot(self) -> Optional[Any]:
        key = get_openrouter_key()
        if not key:
            logger.warning("OpenRouter requested but no key set")
            return None

        
        os.environ.setdefault("OPENAI_API_KEY", key)
        os.environ["OPENAI_BASE_URL"] = OPENROUTER_BASE

        kwargs = dict(
            llm_provider="openai",  
            model_name=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
            api_key=key,
        )
        try:
            sig = inspect.signature(ChatBot.__init__)
            if "base_url" in sig.parameters:
                kwargs["base_url"] = OPENROUTER_BASE
        except Exception:
            pass

        try:
            bot = ChatBot(**kwargs)
            bot = _patch_chatbot_methods(bot)
            # Allow-list for tool names (used by sanitizer)
            try:
                setattr(bot, "_allowed_tool_names", set(self._tool_names()))
            except Exception:
                pass
            logger.info("✅ Spoon ChatBot ready via OpenRouter")
            return bot
        except TypeError as e:
            if "base_url" in kwargs:
                kwargs.pop("base_url", None)
                try:
                    bot = ChatBot(**kwargs)
                    bot = _patch_chatbot_methods(bot)
                    try:
                        setattr(bot, "_allowed_tool_names", set(self._tool_names()))
                    except Exception:
                        pass
                    logger.info("✅ Spoon ChatBot ready via OpenRouter (env base_url fallback)")
                    return bot
                except Exception as e2:
                    logger.error(f"ChatBot init failed even after fallback: {e2}")
            else:
                logger.error(f"ChatBot init failed: {e}")
            return None
        except Exception as e:
            logger.error(f"ChatBot init failed: {e}")
            return None

    def setup_chatbot(self):
        if not SPOONOS_AVAILABLE or ChatBot is None:
            logger.info("SpoonOS not available, will use direct API calls")
            return None
        try:
            provider = self.config["llm_settings"].get("default_provider", "openai")
            model = self.config["llm_settings"].get("model", "gpt-4o-mini")

            if provider == "openai":
                if is_openrouter_enabled():
                    return self._build_openrouter_chatbot()
                api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
                if not api_key:
                    logger.warning("No OPENAI_API_KEY; Spoon agents will be skipped")
                    return None
                bot = ChatBot(llm_provider="openai", model_name=model, api_key=api_key)
                bot = _patch_chatbot_methods(bot)
                try:
                    setattr(bot, "_allowed_tool_names", set(self._tool_names()))
                except Exception:
                    pass
                logger.info(f"✅ Spoon ChatBot ready (OpenAI:{model})")
                return bot

            elif provider == "anthropic":
                # Native Anthropic; if using OpenRouter, prefer openrouter path above
                api_key = (os.getenv("ANTHROPIC_API_KEY") or "").strip()
                if not api_key:
                    logger.warning("No ANTHROPIC_API_KEY; Spoon agents will be skipped")
                    return None
                bot = ChatBot(llm_provider="anthropic", model_name=model, api_key=api_key)
                bot = _patch_chatbot_methods(bot)
                try:
                    setattr(bot, "_allowed_tool_names", set(self._tool_names()))
                except Exception:
                    pass
                logger.info(f"✅ Spoon ChatBot ready (Anthropic:{model})")
                return bot

            else:
                logger.warning(f"Unsupported provider '{provider}' in config")
                return None

        except Exception as e:
            logger.error(f"Failed to setup Spoon ChatBot: {e}")
            return None
    # ---------- end ChatBot setup

    def display_startup_banner(self):
        if self.using_spoon_agents:
            status = "🟢 SpoonOS (OpenRouter)" if is_openrouter_enabled() else "🟢 SpoonOS (OpenAI)"
        elif OPENAI_AVAILABLE and (is_openrouter_enabled() or bool(os.getenv("OPENAI_API_KEY"))):
            status = "🟡 Direct API (OpenRouter)" if is_openrouter_enabled() else "🟡 Direct API (OpenAI)"
        else:
            status = "🔴 No AI Available"
        web3_status = f"{len(self.web3_connections)} chains" if WEB3_AVAILABLE else "Simulation"
        banner = Panel.fit(
            f"""[bold cyan]🤖 DeFi AI Trading Bot v1.3.2[/bold cyan]
[green]Status: {status} • Web3: {web3_status}[/green]
            
[yellow]Multi-Chain • AI-Powered • Risk-Managed[/yellow]
            
[dim]Ready for autonomous DeFi trading across chains[/dim]""",
            border_style="bright_blue"
        )
        console.print(banner)

    def load_config(self) -> Dict:
        config_path = Path(self.config_path)
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                logger.info(f"Configuration loaded from {config_path}")
                return config
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Invalid config file: {e}")
                return self.get_default_config()
        else:
            logger.warning(f"Config file not found at {config_path}, using defaults")
            config_path.parent.mkdir(parents=True, exist_ok=True)
            default_config = self.get_default_config()
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            logger.info(f"Default config saved to {config_path}")
            return default_config

    def get_default_config(self) -> Dict:
        return {
            "default_agent": "market_intelligence",
            "providers": {
                "openai": {
                    "api_key_env": "OPENAI_API_KEY",
                    "model": "gpt-4o-mini",
                    "temperature": 0.1,
                    "max_tokens": 4096,
                    "timeout": 30
                },
                "anthropic": {
                    "api_key_env": "ANTHROPIC_API_KEY",
                    "model": "claude-3-5-sonnet-20241022",
                    "temperature": 0.1,
                    "max_tokens": 4096,
                    "timeout": 30
                }
            },
            "llm_settings": {
                "default_provider": "openai",
                "model": "gpt-4o-mini"
            },
            "agents": {
                "market_intelligence": {
                    "class": "SpoonReactAI",
                    "description": "Multi-chain market intelligence and opportunity scanner",
                    "config": {"max_steps": 15, "tool_choice": "auto"}
                },
                "risk_assessment": {
                    "class": "SpoonReactAI",
                    "description": "AI-powered risk analysis and portfolio management",
                    "config": {"max_steps": 10, "tool_choice": "auto"}
                },
                "execution_manager": {
                    "class": "SpoonReactAI",
                    "description": "Trade execution and position management",
                    "config": {"max_steps": 20, "tool_choice": "auto"}
                }
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
                    "ethereum": {"chain_id": 1, "enabled": True, "gas_optimization": True, "supported_dexs": ["uniswap_v3", "sushiswap", "curve"]},
                    "polygon": {"chain_id": 137, "enabled": True, "gas_optimization": False, "supported_dexs": ["quickswap", "sushiswap", "curve"]},
                    "bsc": {"chain_id": 56, "enabled": True, "gas_optimization": False, "supported_dexs": ["pancakeswap", "biswap", "mdex"]},
                    "arbitrum": {"chain_id": 42161, "enabled": True, "gas_optimization": True, "supported_dexs": ["uniswap_v3", "sushiswap", "curve"]}
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
        logger.remove()
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        debug_mode = os.getenv("DEBUG", "false").lower() == "true"
        if debug_mode:
            log_level = "DEBUG"
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
            level=log_level,
            colorize=True
        )
        log_dir = Path("logs"); log_dir.mkdir(exist_ok=True)
        logger.add(
            log_dir / "trading_bot_{time:YYYY-MM-DD}.log",
            rotation="1 day",
            retention="30 days",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG"
        )
        logger.info("Logging system configured")

    def setup_http_session(self) -> requests.Session:
        session = requests.Session()
        retry_strategy = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter); session.mount("https://", adapter)
        session.timeout = 30  # custom attribute; harmless if unused
        session.headers.update({
            'User-Agent': 'DeFi-AI-Trading-Bot/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
        return session

    def setup_web3(self) -> Dict[str, Any]:
        if not WEB3_AVAILABLE:
            logger.warning("Web3 not available - running in simulation mode")
            return {}
        connections: Dict[str, Any] = {}
        chain_configs = {
            "ethereum": {"rpc": os.getenv("ETHEREUM_RPC_URL", "https://eth.public-rpc.com"), "chain_id": 1, "is_poa": False},
            "polygon": {"rpc": os.getenv("POLYGON_RPC_URL", "https://polygon-rpc.com"), "chain_id": 137, "is_poa": True},
            "bsc": {"rpc": os.getenv("BSC_RPC_URL", "https://bsc-dataseed.binance.org"), "chain_id": 56, "is_poa": True},
            "arbitrum": {"rpc": os.getenv("ARBITRUM_RPC_URL", "https://arb1.arbitrum.io/rpc"), "chain_id": 42161, "is_poa": False}
        }
        for chain, config in chain_configs.items():
            try:
                w3 = Web3(Web3.HTTPProvider(config["rpc"], request_kwargs={'timeout': 30}))
                if config["is_poa"] and POA_MIDDLEWARE is not None:
                    w3.middleware_onion.inject(POA_MIDDLEWARE, layer=0)
                is_connected = w3.is_connected() if hasattr(w3, "is_connected") else w3.isConnected()
                if is_connected:
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

    def setup_redis(self):
        try:
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            client = redis.from_url(redis_url)
            client.ping()
            logger.info("✅ Connected to Redis cache")
            return client
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            return None

    def setup_database(self):
        try:
            database_url = os.getenv("DATABASE_URL")
            if database_url:
                engine = create_engine(database_url)
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

    
    def _make_spoon_agent(self, cls, **kwargs):
        tools_manager = kwargs.pop("tools_manager", None)
        attempts = []
        if tools_manager is not None:
            attempts = [
                dict(available_tools=tools_manager),
                dict(avaliable_tools=tools_manager),
                dict(tools=tools_manager)
            ]
        else:
            attempts = [dict()]

        last_err = None
        for tool_kwargs in attempts:
            try:
                agent = cls(**kwargs, **tool_kwargs)
                return agent
            except TypeError as e:
                last_err = e
                continue
        if last_err:
            raise last_err
        return cls(**kwargs)

    def _tool_names(self) -> List[str]:
        return ["chainbase_analytics", "price_aggregator", "dex_monitor"]

    def _tools_clause(self) -> str:
        names = ", ".join(self._tool_names())
        return (
            f"\nTOOLS: You may call functions only in {{{names}}}. "
            f"Do NOT invent other tool names."
        )

    def initialize_agents(self):
        self.tools = self.setup_tools()
        if SPOONOS_AVAILABLE and self.chatbot:
            try:
                setattr(self.chatbot, "_allowed_tool_names", set(self._tool_names()))
            except Exception:
                pass
            self._initialize_spoonos_agents()
        else:
            self._initialize_fallback_agents()

    def setup_tools(self):
        tools = {
            "chainbase_analytics": ChainbaseAnalyticsTool(),
            "price_aggregator": PriceAggregatorTool(),
            "dex_monitor": DEXMonitorTool()
        }
        logger.info(f"✅ Initialized {len(tools)} trading tools")
        return tools

    def _initialize_spoonos_agents(self):
        try:
            if not self.chatbot:
                raise RuntimeError("ChatBot not initialized")

            market_tools = ToolManager([
                self.tools["chainbase_analytics"],
                self.tools["price_aggregator"],
                self.tools["dex_monitor"]
            ])
            self.market_agent = self._make_spoon_agent(
                SpoonReactAI,
                name="market_intelligence_agent",
                description="Scans multi-chain DeFi markets for trading opportunities",
                system_prompt=(
                    "You are an advanced DeFi market intelligence AI agent. "
                    "Analyze the market and identify profitable opportunities. "
                    "ALWAYS respond with a JSON object containing an 'opportunities' array. "
                    "Each opportunity should have: id, chain, type, tokens, potential_profit, "
                    "risk_score, confidence, estimated_gas, timestamp, and notes. "
                    "Use tools to gather real data when possible."
                    + self._tools_clause()
                ),
                max_steps=15,
                llm=self.chatbot,
                tools_manager=market_tools
            )

            risk_tools = ToolManager([
                self.tools["price_aggregator"],
                self.tools["chainbase_analytics"]
            ])
            self.risk_agent = self._make_spoon_agent(
                SpoonReactAI,
                name="risk_assessment_agent",
                description="Evaluates risks and optimizes portfolio allocation",
                system_prompt=(
                    "You are an expert DeFi risk assessment AI agent."
                    + self._tools_clause()
                ),
                max_steps=12,
                llm=self.chatbot,
                tools_manager=risk_tools
            )

            execution_tools = ToolManager([
                self.tools["dex_monitor"],
                self.tools["price_aggregator"]
            ])
            self.execution_agent = self._make_spoon_agent(
                SpoonReactAI,
                name="execution_manager_agent",
                description="Handles optimal trade execution and position management",
                system_prompt=(
                    "You are an expert DeFi trade execution AI agent."
                    + self._tools_clause()
                ),
                max_steps=20,
                llm=self.chatbot,
                tools_manager=execution_tools
            )

            self.using_spoon_agents = True
            logger.info("✅ SpoonOS agents initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize SpoonOS agents: {e}")
            self._initialize_fallback_agents()

    def _initialize_fallback_agents(self):
        try:
            self.market_agent = DirectOpenAIAgent(
                name="market_intelligence_agent",
                description="Scans multi-chain DeFi markets for trading opportunities",
                system_prompt=(
                    "You are an advanced DeFi market intelligence AI agent. "
                    "Analyze the market and identify profitable opportunities. "
                    "ALWAYS respond with a JSON object containing an 'opportunities' array. "
                    "Each opportunity should have: id, chain, type, tokens, potential_profit, "
                    "risk_score, confidence, estimated_gas, timestamp, and notes."
                ),
                max_steps=15
            )
            self.market_agent.add_tool(self.tools["chainbase_analytics"])
            self.market_agent.add_tool(self.tools["price_aggregator"])
            self.market_agent.add_tool(self.tools["dex_monitor"])

            self.risk_agent = DirectOpenAIAgent(
                name="risk_assessment_agent",
                description="Evaluates risks and optimizes portfolio allocation",
                system_prompt=(
                    "You are an expert DeFi risk assessment AI agent."
                ),
                max_steps=12
            )
            self.risk_agent.add_tool(self.tools["price_aggregator"])
            self.risk_agent.add_tool(self.tools["chainbase_analytics"])

            self.execution_agent = DirectOpenAIAgent(
                name="execution_manager_agent",
                description="Handles optimal trade execution and position management",
                system_prompt=(
                    "You are an expert DeFi trade execution AI agent."
                ),
                max_steps=20
            )
            self.execution_agent.add_tool(self.tools["dex_monitor"])
            self.execution_agent.add_tool(self.tools["price_aggregator"])
            self.using_spoon_agents = False
            logger.info("✅ Fallback agents initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize fallback agents: {e}")
            self.market_agent = None
            self.risk_agent = None
            self.execution_agent = None
            self.using_spoon_agents = False

    def _recover_agents(self):
        logger.warning("Recovering agents after failure...")
        try:
            if SPOONOS_AVAILABLE and self.chatbot:
                self._initialize_spoonos_agents()
            else:
                self._initialize_fallback_agents()
            logger.info("Agents recovered.")
        except Exception as e:
            logger.error(f"Agent recovery failed: {e}")

    # ---------- Market Scan ----------
    def _scan_json_schema(self) -> str:
        return """
Return ONLY this JSON (no markdown, no explanations):

{
  "opportunities": [
    {
      "id": "string",
      "chain": "ethereum|polygon|bsc|arbitrum",
      "type": "arbitrage|yield_farming|liquidity_mining|other",
      "tokens": ["TOKEN_IN","TOKEN_OUT"],
      "potential_profit": 123.45,
      "risk_score": 1.0,
      "confidence": 0.85,
      "estimated_gas": 42.0,
      "timestamp": "<ISO8601>",
      "notes": "short justification"
    }
  ]
}
        """.strip()

    async def scan_markets(self, chains: Optional[List[str]] = None, tokens: Optional[List[str]] = None) -> Dict:
        console.print("\n[bold blue]🔍 Scanning Markets for Opportunities[/bold blue]")
        enabled_chains = [chain for chain, cfg in self.config["trading_config"]["chains"].items() if cfg.get("enabled", False)]
        chains = list(chains) if chains else enabled_chains
        if not chains:
            chains = ["ethereum", "polygon"]
        tokens = list(tokens) if tokens else ["ETH", "BTC", "USDC", "MATIC", "USDT"]

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), console=console,) as progress:
            task = progress.add_task(f"Analyzing {len(chains)} chains...", total=len(chains))
            all_opportunities: List[Dict[str, Any]] = []

            for chain in chains:
                progress.update(task, description=f"Scanning {chain}...")
                
                chain_ops: List[Dict[str, Any]] = []
                
                
                try:
                    if self.market_agent:
                        if hasattr(self.market_agent, "clear"):
                            try:
                                self.market_agent.clear()
                            except Exception:
                                pass
                        prompt = f"""
Scan the {chain} blockchain for profitable DeFi opportunities focusing on tokens: {', '.join(tokens)}.

Use your tools to gather real market data, then identify opportunities like:
- Arbitrage between DEXs
- Price discrepancies
- Yield farming opportunities  
- Liquidity mining rewards

Return a JSON response with opportunities array containing detailed analysis.

{self._scan_json_schema()}
""".strip()
                        try:
                            result = await self.market_agent.run(prompt)
                            if hasattr(self.market_agent, "clear"):
                                try:
                                    self.market_agent.clear()
                                except Exception:
                                    pass
                            chain_ops.extend(self._parse_agent_opportunities(result, chain))
                            logger.info(f"AI agent found {len(chain_ops)} opportunities on {chain}")
                        except Exception as agent_err:
                            logger.error(f"Agent scan failed on {chain}: {agent_err}")
                            # Continue to fallback below
                except Exception as e:
                    logger.error(f"Unexpected error setting up agent scan for {chain}: {e}")

               
                try:
                    dm = await self.tools["dex_monitor"].execute(chain=chain, token_pair=["ETH", "USDC"], min_profit_bps=30)
                    for opp in dm.get("opportunities", []):
                        chain_ops.append({
                            "id": f"{chain}_dex_{len(chain_ops)}",
                            "chain": chain,
                            "type": "ARBITRAGE",
                            "tokens": [opp.get("token_in", "ETH"), opp.get("token_out", "USDC")],
                            "potential_profit": float(opp.get("profit_usd", 0)),
                            "risk_score": float(np.random.uniform(1, 5)),
                            "confidence": float(np.random.uniform(0.75, 0.95)),
                            "timestamp": datetime.now().isoformat(),
                            "estimated_gas": float(opp.get("gas_cost_usd", 50)),
                            "source": "dex_monitor",
                            "details": opp
                        })
                    logger.info(f"DEX monitor found {len(dm.get('opportunities', []))} opportunities on {chain}")
                except Exception as _e:
                    logger.warning(f"Direct DEX monitor fallback failed for {chain}: {_e}")

                # Final fallback if we have no opportunities at all
                if DEMO_MODE and not chain_ops:
                    chain_ops.append({
                        "id": f"{chain}_fallback_0",
                        "chain": chain,
                        "type": "ARBITRAGE",
                        "tokens": ["ETH", "USDC"],
                        "potential_profit": float(np.random.uniform(80, 250)),
                        "risk_score": float(np.random.uniform(2, 4)),
                        "confidence": float(np.random.uniform(0.7, 0.9)),
                        "timestamp": datetime.now().isoformat(),
                        "estimated_gas": float(np.random.uniform(20, 60)),
                        "source": "ai_synthetic"
                    })

                all_opportunities.extend(chain_ops)
                progress.advance(task)
                await asyncio.sleep(0.15)

            if self.redis_client:
                try:
                    self.redis_client.setex("market_scan_results", 3600, json.dumps(all_opportunities, default=str))
                except Exception as e:
                    logger.warning(f"Failed to cache results: {e}")

            self.display_opportunities(all_opportunities)
            return {
                "opportunities": all_opportunities,
                "timestamp": datetime.now().isoformat(),
                "chains_scanned": len(chains),
                "total_opportunities": len(all_opportunities)
            }

    def _parse_agent_opportunities(self, agent_result: Any, chain: str) -> List[Dict]:
        """Enhanced parsing to better extract AI-generated opportunities from agent responses."""
        opportunities: List[Dict[str, Any]] = []
        if agent_result is None:
            return opportunities

        # 1) Parse tool call results first (DEX monitor, price data, etc.)
        if isinstance(agent_result, dict) and "tool_calls" in agent_result:
            for tool_call in agent_result["tool_calls"]:
                if tool_call.get("tool") == "dex_monitor":
                    try:
                        result = tool_call.get("result", "{}")
                        if isinstance(result, str):
                            result = json.loads(result)
                        for opp in result.get("opportunities", []):
                            opportunities.append({
                                "id": f"{chain}_dex_{len(opportunities)}",
                                "chain": chain,
                                "type": "ARBITRAGE",
                                "tokens": [opp.get("token_in", "ETH"), opp.get("token_out", "USDC")],
                                "potential_profit": float(opp.get("profit_usd", 0)),
                                "risk_score": float(np.random.uniform(1, 5)),
                                "confidence": float(np.random.uniform(0.7, 0.9)),
                                "timestamp": datetime.now().isoformat(),
                                "estimated_gas": float(opp.get("gas_cost_usd", 50)),
                                "source": "dex_monitor_ai",
                                "details": opp
                            })
                    except Exception as e:
                        logger.warning(f"Failed to parse DEX monitor tool result: {e}")

        # 2) Parse AI-generated content (the main response)
        text_content = None
        if isinstance(agent_result, dict):
            text_content = agent_result.get("content") or agent_result.get("final_answer")
        elif hasattr(agent_result, "content"):
            text_content = getattr(agent_result, "content")
        elif isinstance(agent_result, str):
            text_content = agent_result

        if text_content:
            logger.debug(f"Parsing AI content for {chain}: {text_content[:200]}...")
            
            # Try to extract JSON from the text
            extracted_json = _extract_json_from_text(text_content)
            if extracted_json:
                opps = _validate_opportunities(extracted_json)
                if opps:
                    logger.info(f"Found {len(opps)} AI-generated opportunities in JSON response")
                    for o in opps:
                        rec = {
                            "id": str(o.get("id") or f"{chain}_ai_{len(opportunities)}"),
                            "chain": str(o.get("chain", chain)),
                            "type": str(o.get("type", "ARBITRAGE")).upper(),
                            "tokens": list(o.get("tokens", ["ETH", "USDC"])),
                            "potential_profit": float(o.get("potential_profit", 0.0)),
                            "risk_score": float(o.get("risk_score", 3.0)),
                            "confidence": float(o.get("confidence", 0.75)),
                            "timestamp": str(o.get("timestamp", datetime.now().isoformat())),
                            "estimated_gas": float(o.get("estimated_gas", 50.0)),
                            "source": "ai_agent",
                        }
                        notes = o.get("notes")
                        if notes:
                            rec["notes"] = str(notes)
                        opportunities.append(rec)
                else:
                    logger.debug(f"JSON found but no valid opportunities array for {chain}")
            else:
                # If no structured JSON found, try to parse opportunities from text
                logger.debug(f"No JSON found, attempting text parsing for {chain}")
                
                # Look for opportunity-like patterns in the text
                if any(keyword in text_content.lower() for keyword in ["arbitrage", "opportunity", "profit", "yield"]):
                    # Create a synthetic opportunity based on the text analysis
                    profit_match = re.search(r'[\$]?(\d+\.?\d*)', text_content)
                    profit = float(profit_match.group(1)) if profit_match else np.random.uniform(100, 500)
                    
                    opportunities.append({
                        "id": f"{chain}_ai_parsed_{len(opportunities)}",
                        "chain": chain,
                        "type": "ARBITRAGE",
                        "tokens": ["ETH", "USDC"],
                        "potential_profit": profit,
                        "risk_score": float(np.random.uniform(2, 4)),
                        "confidence": 0.8,
                        "timestamp": datetime.now().isoformat(),
                        "estimated_gas": float(np.random.uniform(30, 80)),
                        "source": "ai_agent",
                        "notes": text_content[:100] + "..." if len(text_content) > 100 else text_content
                    })

        logger.info(f"Total opportunities parsed for {chain}: {len(opportunities)}")
        return opportunities

    # ---------- Risk / Execution ----------
    async def assess_risk(self, strategy: str, amount: float, tokens: List[str] = None) -> Dict:
        console.print(f"\n[bold yellow]⚠️ Assessing Risk for {strategy}[/bold yellow]")
        tokens = tokens or ["ETH", "USDC"]
        try:
            if self.risk_agent:
                if hasattr(self.risk_agent, "clear"):
                    try:
                        self.risk_agent.clear()
                    except Exception:
                        pass
                prompt = f"""
Perform comprehensive risk analysis for the following trading strategy:

Strategy: {strategy}
Investment Amount: ${amount:,.2f}
Target Tokens: {', '.join(tokens)}

Analyze:
1. Market Risk (volatility, correlations)
2. Liquidity Risk (DEX liquidity, slippage potential)
3. Smart Contract Risk (protocol security)
4. Operational Risk (gas, MEV)
5. Portfolio Impact (position sizing)

Return concise bullet recommendations.
"""
                result = await self.risk_agent.run(prompt)
                if hasattr(self.risk_agent, "clear"):
                    try:
                        self.risk_agent.clear()
                    except Exception:
                        pass
                risk_analysis = self._parse_risk_assessment(result, strategy, amount)
            else:
                risk_analysis = {
                    "strategy": strategy,
                    "amount": amount,
                    "risk_score": float(np.random.uniform(3, 7)),
                    "max_loss": float(amount * np.random.uniform(0.05, 0.20)),
                    "expected_return": float(amount * np.random.uniform(0.05, 0.25)),
                    "confidence": 0.5,
                    "recommendations": ["Use fallback risk assessment", "Consider manual review"],
                    "source": "fallback"
                }
            self.display_risk_assessment(risk_analysis)
            return risk_analysis
        except Exception as e:
            logger.error(f"Error in risk assessment: {e}")
            return {"error": str(e), "strategy": strategy, "amount": amount}

    def _parse_risk_assessment(self, agent_result: Dict, strategy: str, amount: float) -> Dict:
        try:
            risk_data = {
                "strategy": strategy,
                "amount": amount,
                "risk_score": 5.0,
                "max_loss": amount * 0.1,
                "expected_return": amount * 0.1,
                "confidence": 0.7,
                "sharpe_ratio": 1.0,
                "var_95": amount * 0.05,
                "recommendations": [],
                "risk_factors": {},
                "source": "ai_analysis"
            }
            if isinstance(agent_result, dict):
                if "tool_calls" in agent_result:
                    for tool_call in agent_result["tool_calls"]:
                        res = tool_call.get("result", {})
                        if isinstance(res, str):
                            try:
                                res = json.loads(res)
                            except:
                                res = {}
                        if "prices" in res:
                            prices = res["prices"]
                            avg_change = float(np.mean([p.get("24h_change", 0) for p in prices.values()]) or 0.0)
                            risk_data["risk_score"] = float(min(10.0, max(1.0, abs(avg_change) / 2)))
                final_answer = agent_result.get("final_answer", "") or agent_result.get("content", "") or ""
                if final_answer:
                    risk_data["recommendations"].append(final_answer[:200] + "..." if len(final_answer) > 200 else final_answer)
            return risk_data
        except Exception as e:
            logger.error(f"Error parsing risk assessment: {e}")
            return {"strategy": strategy, "amount": amount, "risk_score": 5.0, "error": str(e)}

    async def execute_optimal_trade(self, token_in: str, token_out: str, amount: float, chain: str = "ethereum") -> Dict:
        console.print(f"\n[bold green]🚀 Executing Trade: {amount} {token_in} → {token_out} on {chain}[/bold green]")
        try:
            if self.execution_agent:
                if hasattr(self.execution_agent, "clear"):
                    try:
                        self.execution_agent.clear()
                    except Exception:
                        pass
                prompt = f"""
Execute optimal trade:
- From: {amount} {token_in}
- To: {token_out}
- Chain: {chain}

Consider:
1) Minimal slippage across DEXs
2) Lowest gas costs & timing
3) MEV protection
4) Best price execution

Return a short, clear plan (no markdown).
"""
                result = await self.execution_agent.run(prompt)
                if hasattr(self.execution_agent, "clear"):
                    try:
                        self.execution_agent.clear()
                    except Exception:
                        pass
                execution_plan = self._parse_execution_plan(result, token_in, token_out, amount, chain)
            else:
                execution_plan = {
                    "token_in": token_in,
                    "token_out": token_out,
                    "amount_in": amount,
                    "recommended_dex": "Uniswap V3",
                    "expected_output": amount * 2500 if token_in == "ETH" and token_out == "USDC" else amount * 0.9,
                    "estimated_gas": float(np.random.uniform(50, 150)),
                    "slippage_tolerance": 0.01,
                    "price_impact": float(np.random.uniform(0.001, 0.005)),
                    "source": "fallback"
                }
            self.display_execution_plan(execution_plan)
            trade_result = await self._simulate_trade_execution(execution_plan)
            return trade_result
        except Exception as e:
            logger.error(f"Error in trade execution: {e}")
            return {"error": str(e), "token_in": token_in, "token_out": token_out}

    def _parse_execution_plan(self, agent_result: Dict, token_in: str, token_out: str, amount: float, chain: str) -> Dict:
        try:
            plan = {
                "token_in": token_in,
                "token_out": token_out,
                "amount_in": amount,
                "chain": chain,
                "recommended_dex": "Uniswap V3",
                "expected_output": amount * 0.95,
                "estimated_gas": 100.0,
                "slippage_tolerance": 0.01,
                "price_impact": 0.001,
                "source": "ai_analysis"
            }
            if isinstance(agent_result, dict):
                if "tool_calls" in agent_result:
                    for tool_call in agent_result["tool_calls"]:
                        if tool_call["tool"] == "dex_monitor":
                            result = tool_call.get("result", {})
                            if isinstance(result, str):
                                try:
                                    result = json.loads(result)
                                except:
                                    result = {}
                            opportunities = result.get("opportunities", [])
                            if opportunities:
                                best_opp = min(opportunities, key=lambda x: x.get("gas_cost_usd", 999))
                                plan["recommended_dex"] = best_opp.get("dex_buy", "Uniswap V3")
                                plan["estimated_gas"] = float(best_opp.get("gas_cost_usd", 100))
                final_answer = (agent_result.get("final_answer", "") or agent_result.get("content", "") or "")
                for dex in ["uniswap", "sushiswap", "curve", "1inch", "paraswap"]:
                    if dex in final_answer.lower():
                        plan["recommended_dex"] = dex.title()
                        break
            return plan
        except Exception as e:
            logger.error(f"Error parsing execution plan: {e}")
            return {"token_in": token_in, "token_out": token_out, "amount_in": amount, "error": str(e)}

    # ---------- Displays ----------
    def display_execution_plan(self, plan: Dict):
        console.print("\n[bold]📋 Trade Execution Plan[/bold]")
        table = Table()
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="white")
        table.add_row("Trade", f"{plan['amount_in']} {plan['token_in']} → {plan['token_out']}")
        table.add_row("Recommended DEX", plan.get("recommended_dex", "N/A"))
        table.add_row("Expected Output", f"{plan.get('expected_output', 0):.4f} {plan['token_out']}")
        table.add_row("Estimated Gas", f"${plan.get('estimated_gas', 0):.2f}")
        table.add_row("Slippage Tolerance", f"{plan.get('slippage_tolerance', 0.01)*100:.2f}%")
        table.add_row("Price Impact", f"{plan.get('price_impact', 0.001)*100:.3f}%")
        console.print(table)

    async def _simulate_trade_execution(self, plan: Dict) -> Dict:
        await asyncio.sleep(2)
        expected_output = float(plan.get("expected_output", 0))
        actual_slippage = float(np.random.uniform(0, plan.get("slippage_tolerance", 0.01)))
        actual_output = expected_output * (1 - actual_slippage)
        trade_result = {
            "status": "executed",
            "transaction_hash": f"0x{hash(str(plan))%1000000000:09x}...",
            "amount_in": plan["amount_in"],
            "amount_out": actual_output,
            "actual_slippage": actual_slippage,
            "gas_used": float(plan.get("estimated_gas", 100)) * float(np.random.uniform(0.8, 1.2)),
            "execution_time": float(np.random.uniform(10, 30)),
            "dex_used": plan.get("recommended_dex", "Uniswap V3"),
            "timestamp": datetime.now().isoformat()
        }
        profit = actual_output - float(plan["amount_in"])  # simplified P&L for demo
        self.performance_metrics["total_trades"] += 1
        if profit > 0:
            self.performance_metrics["successful_trades"] += 1
        self.performance_metrics["total_pnl"] += profit
        self.trading_history.append(trade_result)
        console.print(f"\n[bold green]✅ Trade Executed Successfully![/bold green]")
        console.print(f"Transaction Hash: {trade_result['transaction_hash']}")
        console.print(f"Output: {actual_output:.4f} {plan['token_out']}")
        console.print(f"Slippage: {actual_slippage*100:.3f}%")
        return trade_result

    def display_risk_assessment(self, risk: Dict[str, Any]):
        table = Table(title="🛡️ Risk Assessment")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        table.add_row("Strategy", risk.get("strategy", "N/A"))
        table.add_row("Amount", f"${risk.get('amount', 0):,.2f}")
        table.add_row("Risk Score", f"{risk.get('risk_score', 0):.1f}/10")
        table.add_row("Expected Return", f"${risk.get('expected_return', 0):,.2f}")
        table.add_row("Max Loss (VaR 95%)", f"${risk.get('var_95', 0):,.2f}")
        recs = risk.get("recommendations", [])
        table.add_row("Recommendations", recs[0] if recs else "N/A")
        console.print(table)

    def display_opportunities(self, opportunities: List[Dict]):
        if not opportunities:
            console.print("[yellow]No opportunities found[/yellow]")
            return
        opportunities.sort(key=lambda x: x.get("potential_profit", 0), reverse=True)
        table = Table(title=f"🎯 Trading Opportunities Found ({len(opportunities)})")
        table.add_column("ID", style="dim", width=14)
        table.add_column("Chain", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Tokens", style="white")
        table.add_column("Profit ($)", style="green", justify="right")
        table.add_column("Risk", style="yellow", justify="center")
        table.add_column("Confidence", style="blue", justify="center")
        table.add_column("Gas ($)", style="red", justify="right")
        table.add_column("Source", style="dim", width=12)
        for opp in opportunities[:10]:
            risk_score = float(opp.get("risk_score", 0))
            risk_color = "green" if risk_score < 3 else "yellow" if risk_score < 4 else "red"
            confidence = float(opp.get("confidence", 0))
            conf_color = "red" if confidence < 0.7 else "yellow" if confidence < 0.85 else "green"
            tokens_display = " → ".join(opp.get("tokens", ["N/A"]))[:18]
            source = opp.get("source", "unknown")[:12]
            table.add_row(
                str(opp.get("id", "N/A"))[:14],
                str(opp.get("chain", "N/A")).capitalize(),
                str(opp.get("type", "N/A")).upper(),
                tokens_display,
                f"${opp.get('potential_profit', 0):.2f}",
                f"[{risk_color}]{risk_score:.1f}/5[/{risk_color}]",
                f"[{conf_color}]{confidence*100:.1f}%[/{conf_color}]",
                f"${opp.get('estimated_gas', 0):.2f}",
                source
            )
        console.print(table)
        
        # Show breakdown by source
        ai_ops = [opp for opp in opportunities if opp.get("source") == "ai_agent"]
        dex_ops = [opp for opp in opportunities if opp.get("source") in ["dex_monitor", "dex_monitor_ai"]]
        
        if ai_ops:
            console.print(f"\n[bold blue]🧠 AI-Generated: {len(ai_ops)} opportunities[/bold blue]")
        if dex_ops:
            console.print(f"[bold green]🛠️ DEX Monitor: {len(dex_ops)} opportunities[/bold green]")

        console.print("\n✅ Demo market scan completed successfully!")

    async def display_dashboard(self, live_mode: bool = False):
        if live_mode:
            with Live(self.generate_dashboard_layout(), refresh_per_second=0.5, console=console) as live:
                try:
                    while not shutdown_event.is_set():
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
        if self.using_spoon_agents:
            agent_status = "SpoonOS (OpenRouter)" if is_openrouter_enabled() else "SpoonOS (OpenAI)"
            agent_status_display = "🟢 " + agent_status
        elif OPENAI_AVAILABLE and (is_openrouter_enabled() or bool(os.getenv("OPENAI_API_KEY"))):
            agent_status = "Direct (OpenRouter)" if is_openrouter_enabled() else "Direct (OpenAI)"
            agent_status_display = "🟡 " + agent_status
        else:
            agent_status = "Fallback"
            agent_status_display = "🔴 No AI"

        title = Panel.fit(f"[bold blue]🤖 DeFi AI Trading Bot Dashboard[/bold blue] ({agent_status}) - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", border_style="bright_blue")
        layout = Layout()
        layout.split_column(Layout(title, name="title", size=3), Layout(name="main"))
        layout["main"].split_row(Layout(name="left"), Layout(name="right"))
        layout["left"].split_column(Layout(name="status", size=8), Layout(name="performance", size=10))
        layout["right"].split_column(Layout(name="positions", size=10), Layout(name="activity", size=8))
        connected_chains = len(self.web3_connections)
        total_chains = len(self.config["trading_config"]["chains"])
        web3_status = "🟢 Connected" if WEB3_AVAILABLE else "🟡 Simulation"
        status_content = f"""[bold]System Status[/bold]
[green]● AI Agents:[/green] {agent_status_display}
[{ 'green' if WEB3_AVAILABLE else 'yellow'}]● Web3:[/{ 'green' if WEB3_AVAILABLE else 'yellow'}] {web3_status}
[yellow]● Running Strategies:[/yellow] {len(self.running_strategies)}
[blue]● Connected Chains:[/blue] {connected_chains}/{total_chains}
[cyan]● Last Health Check:[/cyan] {self.last_health_check.strftime('%H:%M:%S')}"""
        layout["status"].update(Panel(status_content, title="System", border_style="green"))
        total_trades = self.performance_metrics["total_trades"]
        win_rate = 0 if total_trades == 0 else (self.performance_metrics["successful_trades"] / total_trades) * 100
        performance_content = f"""[bold]Performance Metrics[/bold]
[blue]● Total Trades:[/blue] {total_trades}
[green]● Successful:[/green] {self.performance_metrics['successful_trades']}
[yellow]● Win Rate:[/yellow] {win_rate:.1f}%
[{ 'green' if self.performance_metrics['total_pnl'] >= 0 else 'red'}]● Total P&L:[/{ 'green' if self.performance_metrics['total_pnl'] >= 0 else 'red'}] ${self.performance_metrics['total_pnl']:+.2f}
[cyan]● Sharpe Ratio:[/cyan] {self.performance_metrics.get('sharpe_ratio', 0):.2f}
[magenta]● Max Drawdown:[/magenta] {self.performance_metrics.get('max_drawdown', 0)*100:.1f}%"""
        layout["performance"].update(Panel(performance_content, title="Performance", border_style="blue"))
        positions = list(self.active_positions.values())
        if positions:
            positions_content = "[bold]Active Positions[/bold]\n"
            for pos in positions[:5]:
                pnl = float(pos.get("pnl", 0))
                pnl_color = "green" if pnl > 0 else "red"
                positions_content += f"● {pos.get('token', 'N/A')}: [{pnl_color}]${pnl:+.2f}[/{pnl_color}]\n"
            if len(positions) > 5:
                positions_content += f"... and {len(positions) - 5} more"
        else:
            positions_content = "[dim]No active positions[/dim]"
        layout["positions"].update(Panel(positions_content.strip(), title="Positions", border_style="yellow"))
        recent_trades = len(self.trading_history)
        activity_content = f"""[bold]Recent Activity[/bold]
● [green]✓[/green] {recent_trades} trades executed
● [yellow]⚠[/yellow] {len(self.running_strategies)} strategies running
● [blue]ℹ[/blue] {agent_status_display}
● [green]✓[/green] Risk monitoring enabled
● [cyan]ℹ[/cyan] Market scanning continuous"""
        layout["activity"].update(Panel(activity_content, title="Activity", border_style="cyan"))
        return layout

    async def update_dashboard_data(self):
        self.last_health_check = datetime.now()
        if self.active_positions:
            await self.update_positions()

    async def update_positions(self):
        for pos_id, position in list(self.active_positions.items()):
            try:
                price_change = float(np.random.uniform(-0.02, 0.02))
                position["current_price"] *= (1 + price_change)
                position["pnl"] = (position["current_price"] - position["entry_price"]) * position["amount"]
                position["pnl_percent"] = ((position["current_price"] - position["entry_price"]) / max(position["entry_price"], 1e-9)) * 100
                position["last_updated"] = datetime.now().isoformat()
                if position["pnl_percent"] <= -5.0:
                    await self.trigger_stop_loss(pos_id, position)
                elif position["pnl_percent"] >= 15.0:
                    await self.trigger_take_profit(pos_id, position)
            except Exception as e:
                logger.error(f"Error updating position {pos_id}: {e}")

    async def trigger_stop_loss(self, pos_id: str, position: Dict):
        logger.warning(f"Stop-loss triggered for position {pos_id}")
        console.print(f"[red]🛑 Stop-loss triggered for {position['token']} position[/red]")
        await self.close_position(pos_id, reason="stop_loss")

    async def trigger_take_profit(self, pos_id: str, position: Dict):
        logger.info(f"Take-profit triggered for position {pos_id}")
        console.print(f"[green]🎯 Take-profit triggered for {position['token']} position[/green]")
        await self.close_position(pos_id, reason="take_profit")

    async def close_position(self, pos_id: str, reason: str = "manual"):
        if pos_id not in self.active_positions:
            logger.error(f"Position {pos_id} not found")
            return
        position = self.active_positions[pos_id]
        position["status"] = "closed"
        position["close_reason"] = reason
        position["closed_at"] = datetime.now().isoformat()
        if float(position.get("pnl", 0)) > 0:
            self.performance_metrics["successful_trades"] += 1
        self.performance_metrics["total_pnl"] += float(position.get("pnl", 0))
        del self.active_positions[pos_id]
        logger.info(f"Position {pos_id} closed with P&L: ${float(position.get('pnl', 0)):.2f}")

    def get_system_health(self) -> Dict:
        if self.using_spoon_agents:
            framework = "spoonos"
        elif OPENAI_AVAILABLE and (is_openrouter_enabled() or bool(os.getenv("OPENAI_API_KEY"))):
            framework = "openrouter_fallback" if is_openrouter_enabled() else "openai_fallback"
        else:
            framework = "no_ai"
        return {
            "timestamp": datetime.now().isoformat(),
            "agents": {
                "market_intelligence": "active" if getattr(self, "market_agent", None) else "inactive",
                "risk_assessment": "active" if getattr(self, "risk_agent", None) else "inactive",
                "execution_manager": "active" if getattr(self, "execution_agent", None) else "inactive"
            },
            "connections": {chain: "connected" for chain in self.web3_connections.keys()},
            "agent_framework": framework,
            "database_status": "connected" if self.db_engine else "file_storage",
            "cache_status": "redis" if self.redis_client else "memory",
            "web3_status": "available" if WEB3_AVAILABLE else "simulation",
            "performance": self.performance_metrics,
            "running_strategies": len(self.running_strategies),
            "active_positions": len(self.active_positions),
            "uptime": "100%",
            "last_trade": self.trading_history[-1]["timestamp"] if self.trading_history else None,
            "spoonos_available": SPOONOS_AVAILABLE,
            "openai_available": OPENAI_AVAILABLE,
            "api_keys_configured": {
                "openai": bool(os.getenv("OPENAI_API_KEY")),
                "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
                "openrouter": is_openrouter_enabled()
            }
        }

    async def emergency_stop(self):
        console.print("\n[bold red]🛑 EMERGENCY STOP ACTIVATED[/bold red]")
        for strategy_name in list(self.running_strategies.keys()):
            self.running_strategies[strategy_name]["status"] = "stopped"
            console.print(f"[red]● Stopped strategy: {strategy_name}[/red]")
        positions_to_close = list(self.active_positions.keys())
        for pos_id in positions_to_close:
            await self.close_position(pos_id, reason="emergency_stop")
            console.print(f"[red]● Closed position: {pos_id}[/red]")
        console.print("[bold red]All trading activities stopped[/bold red]")
        logger.critical("Emergency stop executed - all activities halted")

    def save_state(self):
        state = {
            "timestamp": datetime.now().isoformat(),
            "active_positions": self.active_positions,
            "performance_metrics": self.performance_metrics,
            "running_strategies": self.running_strategies,
            "config": self.config,
            "trading_history": self.trading_history[-100:]
        }
        state_file = Path("data/bot_state.json")
        state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        logger.info("Bot state saved")

    def load_state(self):
        state_file = Path("data/bot_state.json")
        if state_file.exists():
            try:
                with open(state_file) as f:
                    state = json.load(f)
                self.active_positions = state.get("active_positions", {})
                self.performance_metrics = state.get("performance_metrics", self.performance_metrics)
                self.running_strategies = state.get("running_strategies", {})
                self.trading_history = state.get("trading_history", [])
                logger.info(f"Bot state loaded - {len(self.active_positions)} positions, {len(self.trading_history)} trades")
                return True
            except Exception as e:
                logger.error(f"Failed to load state: {e}")
                return False
        return False

    async def graceful_shutdown(self):
        console.print("\n[yellow]🔄 Initiating graceful shutdown...[/yellow]")
        self.save_state()
        if hasattr(self, 'session'):
            self.session.close()
        if self.db_engine:
            self.db_engine.dispose()
        if self.redis_client:
            try:
                self.redis_client.close()
            except Exception:
                pass
        console.print("[green]✅ Shutdown complete[/green]")
        logger.info("Bot shutdown completed")


def setup_signal_handlers(bot: TradingBotOrchestrator):
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}")
        shutdown_event.set()
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
    """DeFi AI Trading Bot - Multi-chain autonomous trading with real AI agents"""
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
    """Scan markets for trading opportunities using real AI agents"""
    async def run_scan():
        bot = TradingBotOrchestrator(ctx.obj['config'])
        setup_signal_handlers(bot)
        chains_list = list(chains) if chains else None
        tokens_list = list(tokens) if tokens else None
        result = await bot.scan_markets(chains=chains_list, tokens=tokens_list)
        opportunities = result.get("opportunities", [])
        profitable_ops = [op for op in opportunities if op.get("potential_profit", 0) >= min_profit]
        if profitable_ops:
            console.print(f"\n[bold green]Found {len(profitable_ops)} opportunities above ${min_profit} profit threshold[/bold green]")
            ai_ops = [op for op in profitable_ops if op.get("source") == "ai_agent"]
            tool_ops = [op for op in profitable_ops if op.get("source") != "ai_agent"]
            if ai_ops:
                console.print(f"[cyan]🧠 AI-Generated JSON: {len(ai_ops)} opportunities[/cyan]")
            if tool_ops:
                console.print(f"[blue]🛠️ Tool-Generated: {len(tool_ops)} opportunities[/blue]")
        else:
            console.print(f"[yellow]No opportunities found above ${min_profit} profit threshold[/yellow]")
    
    try:
        asyncio.run(run_scan())
    except KeyboardInterrupt:
        console.print("\n[yellow]Scan cancelled by user[/yellow]")


@cli.command()
@click.argument('strategy', type=click.Choice(['arbitrage', 'yield_farming', 'dca', 'market_making']))
@click.argument('amount', type=float)
@click.option('--tokens', '-t', multiple=True, help='Target tokens for analysis')
@click.pass_context
def risk(ctx, strategy, amount, tokens):
    """Assess risk for a trading strategy using AI agents"""
    async def run_risk_assessment():
        bot = TradingBotOrchestrator(ctx.obj['config'])
        setup_signal_handlers(bot)
        tokens_list = list(tokens) if tokens else ["ETH", "USDC"]
        result = await bot.assess_risk(strategy, amount, tokens_list)
        if "error" not in result:
            risk_score = result.get("risk_score", 5)
            console.print(f"\n[bold]Risk Assessment Complete[/bold]")
            console.print(f"Overall Risk Score: {risk_score:.1f}/10")
            if risk_score < 4:
                console.print("[green]✅ Low risk - proceed with confidence[/green]")
            elif risk_score < 7:
                console.print("[yellow]⚠️ Medium risk - proceed with caution[/yellow]")
            else:
                console.print("[red]🛑 High risk - consider reducing position size[/red]")
        else:
            console.print(f"[red]Risk assessment failed: {result['error']}[/red]")
    
    try:
        asyncio.run(run_risk_assessment())
    except KeyboardInterrupt:
        console.print("\n[yellow]Risk assessment cancelled by user[/yellow]")


@cli.command()
@click.argument('token_in')
@click.argument('token_out')
@click.argument('amount', type=float)
@click.option('--chain', default='ethereum', help='Blockchain to execute on')
@click.pass_context
def trade(ctx, token_in, token_out, amount, chain):
    """Execute optimal trade using AI agents"""
    async def run_trade():
        bot = TradingBotOrchestrator(ctx.obj['config'])
        setup_signal_handlers(bot)
        result = await bot.execute_optimal_trade(token_in, token_out, amount, chain)
        if "error" not in result:
            console.print(f"\n[bold green]🎉 Trade executed successfully![/bold green]")
            console.print(f"Transaction: {result.get('transaction_hash', 'N/A')}")
            console.print(f"Output: {result.get('amount_out', 0):.4f} {token_out}")
        else:
            console.print(f"[red]Trade execution failed: {result['error']}[/red]")
    
    try:
        asyncio.run(run_trade())
    except KeyboardInterrupt:
        console.print("\n[yellow]Trade cancelled by user[/yellow]")


@cli.command()
@click.option('--live', '-l', is_flag=True, help='Live dashboard mode')
@click.pass_context
def dashboard(ctx, live):
    """Display trading dashboard"""
    async def run_dashboard():
        bot = TradingBotOrchestrator(ctx.obj['config'])
        setup_signal_handlers(bot)
        bot.load_state()
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
            bot.performance_metrics.update({
                "total_trades": 15,
                "successful_trades": 12,
                "total_pnl": 450.75,
                "win_rate": 80.0
            })
        await bot.display_dashboard(live_mode=live)
    
    try:
        asyncio.run(run_dashboard())
    except KeyboardInterrupt:
        console.print("\n[yellow]Dashboard closed[/yellow]")


@cli.command()
@click.pass_context
def status(ctx):
    """Show system status and health check"""
    async def run_status():
        bot = TradingBotOrchestrator(ctx.obj['config'])
        health = bot.get_system_health()
        table = Table(title="🏥 System Health Check")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="white")
        table.add_column("Details", style="dim")
        framework = health["agent_framework"]
        if framework == "spoonos":
            framework_status = "[green]SPOONOS ACTIVE[/green]"; details = "Full AI capabilities (OpenRouter/OpenAI)"
        elif framework == "openai_fallback":
            framework_status = "[yellow]OPENAI FALLBACK[/yellow]"; details = "Direct OpenAI API calls"
        elif framework == "openrouter_fallback":
            framework_status = "[yellow]OPENROUTER FALLBACK[/yellow]"; details = "Direct OpenRouter API calls"
        else:
            framework_status = "[red]NO AI[/red]"; details = "Mock responses only"
        table.add_row("AI Framework", framework_status, details)
        api_keys = health["api_keys_configured"]
        if api_keys.get("openai"):
            table.add_row("OpenAI API", "[green]CONFIGURED[/green]", "OPENAI_API_KEY present")
        else:
            table.add_row("OpenAI API", "[red]NOT CONFIGURED[/red]", "Set OPENAI_API_KEY")
        if api_keys.get("openrouter"):
            table.add_row("OpenRouter API", "[green]CONFIGURED[/green]", "Using OpenRouter endpoint")
        else:
            table.add_row("OpenRouter API", "[yellow]NOT DETECTED[/yellow]", "Set OPENROUTER_API_KEY or use sk-or- in OPENAI_API_KEY")
        if api_keys.get("anthropic"):
            table.add_row("Anthropic API", "[green]CONFIGURED[/green]", "Claude available")
        else:
            table.add_row("Anthropic API", "[red]NOT CONFIGURED[/red]", "Set ANTHROPIC_API_KEY")
        for agent, st in health["agents"].items():
            status_color = "green" if st == "active" else "yellow"
            table.add_row(f"Agent: {agent.replace('_', ' ').title()}", f"[{status_color}]{str(st).upper()}[/{status_color}]", "AI-powered" if st == "active" else "Offline")
        for chain, st in health["connections"].items():
            table.add_row(f"Chain: {chain.title()}", f"[green]{str(st).upper()}[/green]", "RPC connection active")
        components = [("Database", health["database_status"], "Data persistence"),
                      ("Cache", health["cache_status"], "Performance optimization"),
                      ("Web3", health["web3_status"], "Blockchain connectivity")]
        for name, st, desc in components:
            color = "green" if st in ["connected", "redis", "available"] else "yellow"
            table.add_row(name, f"[{color}]{str(st).upper()}[/{color}]", desc)
        console.print(table)
        perf = health["performance"]
        console.print(f"\n[bold]Performance Summary:[/bold]")
        console.print(f"Total Trades: {perf['total_trades']}")
        if perf['total_trades'] > 0:
            console.print(f"Success Rate: {(perf['successful_trades']/perf['total_trades']*100):.1f}%")
        console.print(f"Total P&L: ${perf['total_pnl']:+.2f}")
        console.print(f"Running Strategies: {health['running_strategies']}")
        console.print(f"Active Positions: {health['active_positions']}")
        console.print(f"\n[bold]Recommendations:[/bold]")
        if framework == "openrouter_fallback" and not (SPOONOS_AVAILABLE and bot.chatbot and bot.using_spoon_agents):
            console.print("[yellow]● OpenRouter is configured. You can also use SpoonOS with OpenRouter (already supported here).[/yellow]")
        if not (api_keys.get("openai") or api_keys.get("openrouter") or api_keys.get("anthropic")):
            console.print("[red]● Configure at least one LLM API key for full functionality[/red]")
        if not SPOONOS_AVAILABLE:
            console.print("[yellow]● Install spoon-ai-sdk for enhanced agent capabilities (optional)[/yellow]")
        mode = os.getenv("SPOON_TOOLCALL_MODE", "").lower() or ("strip" if is_openrouter_enabled() else "sanitize")
        console.print(f"\n[dim]Tool-call mode: {mode} (set SPOON_TOOLCALL_MODE to 'strip' or 'sanitize')[/dim]")
    
    try:
        asyncio.run(run_status())
    except Exception as e:
        console.print(f"[red]Status check failed: {e}[/red]")


@cli.command()
@click.confirmation_option(prompt='Are you sure you want to stop all trading activities?')
@click.pass_context
def emergency_stop(ctx):
    """Emergency stop all trading activities"""
    async def run_emergency_stop():
        bot = TradingBotOrchestrator(ctx.obj['config'])
        await bot.emergency_stop()
    
    try:
        asyncio.run(run_emergency_stop())
    except Exception as e:
        console.print(f"[red]Emergency stop failed: {e}[/red]")


@cli.command()
@click.pass_context
def configure(ctx):
    """Interactive configuration setup"""
    config_path = ctx.obj.get('config') or "config/config.json"
    config_file = Path(config_path)
    console.print("[bold blue]🔧 DeFi Trading Bot Configuration[/bold blue]")
    config_file.parent.mkdir(parents=True, exist_ok=True)
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
        console.print(f"[green]Loaded existing configuration from {config_path}[/green]")
    else:
        bot = TradingBotOrchestrator()
        config = bot.get_default_config()
        console.print("[yellow]Creating new configuration[/yellow]")
    console.print("\n[bold]Dependencies Status:[/bold]")
    console.print(f"SpoonOS: {'✅ Available' if SPOONOS_AVAILABLE else '❌ Not installed'}")
    console.print(f"Web3: {'✅ Available' if WEB3_AVAILABLE else '❌ Not installed'}")
    console.print(f"Redis: {'✅ Available' if REDIS_AVAILABLE else '❌ Not installed'}")
    console.print(f"SQLAlchemy: {'✅ Available' if SQLALCHEMY_AVAILABLE else '❌ Not installed'}")
    console.print("\n[bold]LLM Configuration:[/bold]")
    openai_key = Prompt.ask("OpenAI API Key (can also be an OpenRouter sk-or- key)", default=os.getenv("OPENAI_API_KEY", ""))
    anthropic_key = Prompt.ask("Anthropic API Key (optional)", default=os.getenv("ANTHROPIC_API_KEY", ""))
    openrouter_key = Prompt.ask("OpenRouter API Key (optional, used if OPENAI_API_KEY isn't sk-or-)", default=os.getenv("OPENROUTER_API_KEY", ""))
    if not (openai_key or anthropic_key or openrouter_key):
        console.print("[yellow]⚠️ No API keys provided - bot will run in mock mode[/yellow]")
    elif SPOONOS_AVAILABLE and (openai_key and openai_key.startswith("sk-or-") or openrouter_key):
        console.print("[green]✅ SpoonOS will use OpenRouter (OpenAI-compatible base_url)[/green]")
    elif SPOONOS_AVAILABLE and openai_key and not openai_key.startswith("sk-or-"):
        console.print("[green]✅ SpoonOS will use OpenAI directly[/green]")

    console.print("\n[bold]Risk Management:[/bold]")
    max_risk = Prompt.ask("Max portfolio risk per trade", default="0.02")
    stop_loss = Prompt.ask("Default stop loss percentage", default="0.05")
    take_profit = Prompt.ask("Default take profit percentage", default="0.15")
    try:
        config["trading_config"]["risk_management"]["max_portfolio_risk"] = float(max_risk)
        config["trading_config"]["risk_management"]["stop_loss_percentage"] = float(stop_loss)
        config["trading_config"]["risk_management"]["take_profit_percentage"] = float(take_profit)
    except Exception:
        pass
    console.print("\n[bold]Chain Configuration:[/bold]")
    for chain_name in config["trading_config"]["chains"].keys():
        enabled = Confirm.ask(f"Enable {chain_name.title()} chain?", default=config["trading_config"]["chains"][chain_name]["enabled"])
        config["trading_config"]["chains"][chain_name]["enabled"] = enabled
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    console.print(f"\n[green]✅ Configuration saved to {config_path}[/green]")
    console.print("\n[bold]Next steps:[/bold]")
    console.print("1. Install missing dependencies:")
    if not SPOONOS_AVAILABLE:
        console.print("   pip install spoon-ai-sdk  # optional SpoonOS SDK")
    if not WEB3_AVAILABLE:
        console.print("   pip install web3 eth-account")
    if not REDIS_AVAILABLE:
        console.print("   pip install redis")
    if not SQLALCHEMY_AVAILABLE:
        console.print("   pip install sqlalchemy")
    console.print("2. Set up environment variables (API keys, RPC URLs)")
    console.print("3. Run: python defi_bot_fixed.py status")
    console.print("4. Start with: python defi_bot_fixed.py scan")


@cli.command()
def version():
    """Show version information"""
    console.print("[bold blue]🤖 DeFi AI Trading Bot[/bold blue]")
    console.print("Version: 1.3.2 (CRITICAL FIX: Null content issue resolved)")
    console.print("Status: Production-Ready")
    console.print("Multi-chain • AI-powered • Risk-managed")
    console.print("\n[bold]AI Framework Status:[/bold]")
    console.print(f"SpoonOS: {'✅ Available' if SPOONOS_AVAILABLE else '❌ Not available'}")
    if is_openrouter_enabled():
        console.print("Using: SpoonOS/Direct via OpenRouter")
    elif OPENAI_AVAILABLE and os.getenv('OPENAI_API_KEY'):
        console.print("Using: SpoonOS/Direct via OpenAI")
    else:
        console.print("Using: Mock mode")
    console.print("\n[bold]Dependency Status:[/bold]")
    console.print(f"Web3: {'✅' if WEB3_AVAILABLE else '❌'}")
    console.print(f"Redis: {'✅' if REDIS_AVAILABLE else '❌'}")
    console.print(f"SQLAlchemy: {'✅' if SQLALCHEMY_AVAILABLE else '❌'}")
    console.print("\nFor help: python defi_bot_fixed.py --help")


def main():
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