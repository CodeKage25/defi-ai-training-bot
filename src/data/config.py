
import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Centralized configuration manager.
    Singleton pattern to load and access config settings.
    """
    _instance = None
    _config: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        """Load configuration from JSON file and env vars"""
        try:
            # Default path: project_root/config/config.json
            base_path = Path(__file__).parent.parent.parent
            config_path = base_path / "config" / "config.json"
            
            if config_path.exists():
                with open(config_path, "r") as f:
                    self._config = json.load(f)
                logger.info(f"Loaded config from {config_path}")
            else:
                logger.warning(f"Config file not found at {config_path}. Using defaults.")
                self._config = {}
                
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            self._config = {}

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value by key (dot notation supported)"""
        keys = key.split(".")
        value = self._config
        
        try:
            for k in keys:
                value = value.get(k)
                if value is None:
                    return default
            return value
        except AttributeError:
            return default

    # Type-safe getters for specific sections
    
    @property
    def risk_config(self) -> Dict[str, Any]:
        return self.get("trading_config.risk_management", {})

    @property
    def execution_config(self) -> Dict[str, Any]:
        return self.get("trading_config.execution", {})

    @property
    def chain_config(self) -> Dict[str, Any]:
        return self.get("trading_config.chains", {})

    def get_whitelisted_tokens(self, chain: str) -> list:
        # In a real app, this would be in config.json. 
        # For now, we return the hardcoded list if not in config, 
        # allowing us to move the hardcoded list here.
        
        # Check config first
        whitelist = self.get(f"trading_config.chains.{chain}.whitelisted_tokens")
        if whitelist:
            return whitelist
            
        # Fallback defaults (moved from SafeExecutionEngine)
        defaults = {
            "ethereum": ["ETH", "WETH", "USDC", "USDT", "DAI", "WBTC"],
            "polygon": ["MATIC", "WMATIC", "USDC", "USDT", "DAI", "WETH", "WBTC"],
            "bsc": ["BNB", "WBNB", "BUSD", "USDC", "USDT", "ETH", "BTCB"],
            "arbitrum": ["ETH", "WETH", "USDC", "USDT", "DAI", "WBTC", "ARB"]
        }
        return defaults.get(chain, [])

    def get_whitelisted_dexs(self, chain: str) -> list:
        # Check config first (existing config.json has supported_dexs)
        dexs = self.get(f"trading_config.chains.{chain}.supported_dexs")
        if dexs:
            return dexs
            
        # Fallback
        return ["uniswap_v3", "sushiswap", "curve", "pancakeswap", "quickswap", "1inch"]
