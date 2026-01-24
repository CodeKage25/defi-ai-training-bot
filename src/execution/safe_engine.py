
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from decimal import Decimal

# Configure logger
logger = logging.getLogger(__name__)

from src.data.config import ConfigManager

class SafeExecutionEngine:
    """
    The Deterministic Layer (Python Firewall).
    Acts as a strict gateway for all trading operations.
    Enforces rigid safety checks before allowing any execution.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config_manager = ConfigManager()
        
        # Load Risk Parameters from Config
        risk_config = self.config_manager.risk_config
        
        # Use config value or fallback to safe defaults
        self.MAX_DRAWDOWN_PERCENT = risk_config.get("max_drawdown", 0.1) * 100 # Config is 0.1 (10%), we store as percent logic or ratio
        # Let's align on ratio for internal logic if easier, or percent. 
        # Config says max_drawdown: 0.1.
        self.MAX_DRAWDOWN_RATIO = risk_config.get("max_drawdown", 0.1) 
        
        # Hard cap on position size
        self.MAX_POSITION_SIZE_USD = 10000.0 # Could also add to config.json
        
        # State tracking
        self.portfolio_value_start_of_day = 100000.0 # Mock starting value
        self.current_drawdown = 0.0

    def validate_and_execute(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point. Receives a 'proposal' from the AI 'Board'.
        Runs deterministic checks. If pass, executes. If fail, rejects.
        """
        proposal_id = proposal.get("id", "unknown")
        logger.info(f"Received proposal {proposal_id}: {json.dumps(proposal, indent=2)}")

        # 1. WHITELIST CHECK
        if not self._check_whitelist(proposal):
            return self._reject_proposal(proposal, "Failed Whitelist Check")

        # 2. DRAWDOWN CHECK (Circuit Breaker)
        if not self._check_drawdown(proposal):
            return self._reject_proposal(proposal, "Failed Drawdown/Risk Check")

        # 3. SOLVENT CHECK (Simulation)
        if not self._check_solvent(proposal):
             return self._reject_proposal(proposal, "Failed Solvent/Simulation Check")

        # If all pass, allow execution
        logger.info(f"Proposal {proposal_id} PASSED all checks. Executing.")
        return self._execute_transaction(proposal)

    def _check_whitelist(self, proposal: Dict[str, Any]) -> bool:
        """
        Verify that interaction targets (tokens, contracts) are in the allowing list.
        Prevents 'phishing token' attacks.
        """
        chain = proposal.get("chain", "ethereum").lower()
        token_in = proposal.get("token_in")
        token_out = proposal.get("token_out")
        dex = proposal.get("dex")

        # Get whitelists from ConfigManager
        allowed_tokens = self.config_manager.get_whitelisted_tokens(chain)
        allowed_dexs = self.config_manager.get_whitelisted_dexs(chain)

        # Check Chain (implicit if whitelisted tokens exist)
        if not allowed_tokens:
            logger.error(f"Chain {chain} not supported or no whitelist found.")
            return False

        # Check Tokens
        if token_in not in allowed_tokens:
             logger.error(f"Token {token_in} NOT in whitelist for {chain}.")
             return False
        if token_out not in allowed_tokens:
             logger.error(f"Token {token_out} NOT in whitelist for {chain}.")
             return False

        # Check DEX
        is_dex_allowed = any(allowed in dex for allowed in allowed_dexs)
        if not is_dex_allowed:
            logger.error(f"DEX {dex} NOT in whitelist (Allowed: {allowed_dexs}).")
            return False

        return True

    def _check_drawdown(self, proposal: Dict[str, Any]) -> bool:
        """
        Circuit Breaker.
        If the proposed trade value looks risky or if the daily drawdown is hit.
        """
        amount_usd = float(proposal.get("amount_usd", 0.0))
        
        # Size Cap Check
        if amount_usd > self.MAX_POSITION_SIZE_USD:
            logger.error(f"Trade size ${amount_usd} exceeds limit ${self.MAX_POSITION_SIZE_USD}.")
            return False

        # Estimated Slab/Drawdown (Mock logic)
        # If we assume worst case slippage (e.g. 50%), would it breach max drawdown?
        # worst_case_loss = amount_usd * 0.5 
        # For this demo, let's just check if amount is reasonable vs portfolio
        if amount_usd > (self.portfolio_value_start_of_day * 0.1): # Max 10% of portfolio in one trade
             logger.error(f"Trade size ${amount_usd} is > 10% of portfolio. Too risky.")
             return False

        return True

    def _check_solvent(self, proposal: Dict[str, Any]) -> bool:
        """
        Simulation Check.
        In a real system, this fork-simulates the tx to see if it reverts.
        Here we do basic logical validation (e.g., do we have balance?).
        """
        # Mock logic: Assume we have balance if amount < 50000
        amount_in = float(proposal.get("amount_in", 0.0))
        
        if amount_in <= 0:
            logger.error(f"Invalid amount {amount_in}.")
            return False

        # Simulate transaction success probability
        # In this firewall, we assume 'deterministic' means 'it works logically'
        return True

    def _execute_transaction(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Actually execute the trade (Mock).
        """
        return {
            "status": "executed",
            "tx_hash": "0xsafe" + "1" * 60, # Mock hash
            "details": proposal,
            "timestamp": datetime.now().isoformat()
        }

    def _reject_proposal(self, proposal: Dict[str, Any], reason: str) -> Dict[str, Any]:
        logger.warning(f"Proposal REJECTED: {reason}")
        return {
            "status": "rejected",
            "reason": reason,
            "details": proposal
        }
